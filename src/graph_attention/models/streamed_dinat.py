"""M41.3: host-streamed, single-GPU M37 DiNAT-DiT inference.

Unlike M41.1/M41.2, *no* full-graph hidden state, QKV tensor, or MLP
activation is resident on the accelerator. A CPU-owned Jacobi buffer is read
for each partition, with complete incoming one-hop/exact-two-hop halos;
only owned updated nodes are copied back to the next CPU buffer.

Inference-only. The complete static graph and plan currently reside on the
host: topology construction is NOT yet an out-of-core graph builder.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass, replace
import time

import torch

from .dinat_dit import AlternatingDilatedGeometricDiT
from .streamed_stage_profile import StreamedStageProfiler
from .dit_common import _modulate
from .owned_query_dinat import (
    OwnedQueryPartition,
    OwnedQueryPlan,
    OwnedQueryTopology,
    _owned_context,
    build_owned_query_plan,
)


@dataclass(frozen=True)
class StreamedDiNATPlan:
    """Immutable CPU plan used with a separately resident GPU model.

    All graph indices, geometric displacements, and CSR structures remain
    on the host; one partition's metadata moves to the compute device per
    layer. Rebuild after any change of mesh, topology, partitioning or model
    attention configuration.
    """

    cpu_plan: OwnedQueryPlan
    model_device: torch.device
    pinned_host_buffers: bool = False

    @property
    def base(self):
        return self.cpu_plan.base

    @property
    def partitions(self):
        return self.cpu_plan.partitions


@torch.inference_mode(False)
@torch.no_grad()
def build_streamed_dinat_plan(
    model: AlternatingDilatedGeometricDiT,
    *,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    attention_edge_indices: dict[str, torch.Tensor],
    partition_ids: torch.Tensor,
    pin_memory: bool = False,
) -> StreamedDiNATPlan:
    """Prepare CPU-resident halo metadata without moving the mesh to the GPU.

    The incoming graph must already exist on CPU. Reuse exactly the M41.2
    rectangular CSR and M41.1's original-edge/self-loop contracts. A CPU
    model copy is used only to satisfy the original planner's device check;
    the trained accelerator model is never moved or modified.
    """
    if not isinstance(model, AlternatingDilatedGeometricDiT):
        raise TypeError("M41.3 requires AlternatingDilatedGeometricDiT")
    if model.sparse_attention_backend != "torch_sparse":
        raise ValueError("M41.3 requires torch_sparse")
    if coords.device.type != "cpu" or edge_index.device.type != "cpu":
        raise ValueError("M41.3 plan requires CPU coordinates and CPU edges")
    if partition_ids.device.type != "cpu" or any(
        tensor.device.type != "cpu" for tensor in attention_edge_indices.values()
    ):
        raise ValueError("M41.3 graph partition and attention edges must be on CPU")
    parameter = model.input_projection.weight
    if parameter.dtype != torch.float32 or coords.dtype != torch.float32:
        raise TypeError("M41.3 currently requires float32 model and coordinates")
    if pin_memory and parameter.device.type != "cuda":
        raise ValueError("pinned host buffers require a CUDA model")

    # A clean CPU model copy avoids forcing the static mesh and full
    # topology onto GPU. Do not carry cached CUDA CSR graphs into deepcopy.
    # The planning clone is not retained after construction.
    if model._torch_sparse_topology_cache or model._self_attention_edge_cache:
        raise ValueError(
            "Clear the model's cached full-graph attention topologies before "
            "constructing a streamed plan"
        )
    cpu_model = copy.deepcopy(model).to("cpu")
    try:
        cpu_plan = build_owned_query_plan(
            cpu_model,
            coords=coords,
            edge_index=edge_index,
            attention_edge_indices=attention_edge_indices,
            partition_ids=partition_ids,
        )
    finally:
        del cpu_model

    return StreamedDiNATPlan(
        cpu_plan=cpu_plan,
        model_device=parameter.device,
        pinned_host_buffers=bool(pin_memory),
    )


def _host_buffer(plan: StreamedDiNATPlan, channels: int) -> torch.Tensor:
    return torch.empty(
        (plan.base.num_nodes, channels),
        device="cpu",
        dtype=torch.float32,
        pin_memory=plan.pinned_host_buffers,
    )


def _gpu_topology(topology: OwnedQueryTopology, device: torch.device) -> OwnedQueryTopology:
    """Stage one attention CSR; never retain all partition CSRs on GPU."""
    return OwnedQueryTopology(
        pattern=topology.pattern.to(device),
        row_indices=topology.row_indices.to(device),
        col_indices=topology.col_indices.to(device),
        edge_order=topology.edge_order.to(device),
    )


def _initial_owned(
    model: AlternatingDilatedGeometricDiT,
    input_host: torch.Tensor,
    plan: StreamedDiNATPlan,
    part: OwnedQueryPartition,
) -> torch.Tensor:
    device = plan.model_device
    owned = part.nodes.owned_global
    inputs = input_host.index_select(0, owned).to(device)
    coords = plan.base.model_coords.index_select(0, owned).to(device)
    return model.input_projection(inputs) + model.coordinate_projection(coords)


def _phase(
    profile: StreamedStageProfiler | None, name: str,
    *, device: torch.device, layer_index: int | None = None,
):
    """The ordinary numerical path has no CUDA synchronization added."""
    return (
        profile.measure(name, device=device, layer_index=layer_index)
        if profile is not None else nullcontext()
    )


def _partition_iterator(
    partitions,
    profile: StreamedStageProfiler | None,
    *, device: torch.device, layer_index: int | None = None,
):
    """Capture NPZ/CSR materialization without retaining prior partitions."""
    if profile is None:
        yield from partitions
        return
    iterator = iter(partitions)
    while True:
        with profile.measure("partition_metadata_load", device=device, layer_index=layer_index):
            try:
                part = next(iterator)
            except StopIteration:
                return
        yield part


def _update_owned(
    block: torch.nn.Module,
    current_host: torch.Tensor,
    part: OwnedQueryPartition,
    *,
    layer_index: int,
    shifts: tuple[torch.Tensor, ...],
    device: torch.device,
    stage_profiler: StreamedStageProfiler | None = None,
) -> torch.Tensor:
    """Only one partition's hidden/QKV/attention/MLP tensors live on GPU.

    Optional profiling inserts CUDA synchronization around disjoint phases.
    With no profiler these operations and their order match M41.3.
    """
    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = shifts
    nodes = part.nodes
    with _phase(stage_profiler, "cpu_hidden_gather", device=device, layer_index=layer_index):
        host_hidden = current_host.index_select(0, nodes.read_global)
    with _phase(stage_profiler, "h2d_hidden_and_owned_indices", device=device, layer_index=layer_index):
        read_hidden = host_hidden.to(device)
        owned_local = nodes.owned_local.to(device)
    del host_hidden

    with _phase(stage_profiler, "gpu_norm_qkv", device=device, layer_index=layer_index):
        attention_input = _modulate(block.norm1(read_hidden), shift_attn, scale_attn)
        attention_module = block.attention
        qkv = attention_module.qkv(attention_input).reshape(
            read_hidden.shape[0], 3, attention_module.num_heads, attention_module.head_dim
        )
    attention = nodes.local if layer_index % 2 == 0 else nodes.dilated
    topology = part.local if layer_index % 2 == 0 else part.dilated

    # Displacements and rectangular CSR are moved only for this partition;
    # the original M37 edge sets and sparse attention are unchanged.
    with _phase(stage_profiler, "h2d_attention_topology_geometry", device=device, layer_index=layer_index):
        staged_attention = replace(
            attention, displacement=attention.displacement.to(device)
        )
        staged_topology = _gpu_topology(topology, device)
    with _phase(stage_profiler, "gpu_attention_context", device=device, layer_index=layer_index):
        scores = _owned_context(
            attention_module,
            qkv[:, 0].index_select(0, owned_local),
            qkv[:, 1],
            qkv[:, 2],
            staged_attention,
            staged_topology,
        )
    # Before M41.5, the staged CSR was an inline argument to _owned_context
    # and became unreachable immediately after that call. Keep that tensor
    # lifetime (and peak GPU memory) in the optional profiler refactor.
    del staged_topology
    with _phase(stage_profiler, "gpu_attention_out_residual", device=device, layer_index=layer_index):
        owned_hidden = read_hidden.index_select(0, owned_local)
        owned_hidden = owned_hidden + gate_attn * attention_module.out_proj(
            scores.reshape(owned_local.numel(), model_hidden_dim(block))
        )
    with _phase(stage_profiler, "gpu_mlp", device=device, layer_index=layer_index):
        mlp_input = _modulate(block.norm2(owned_hidden), shift_mlp, scale_mlp)
        result = owned_hidden + gate_mlp * block.mlp(mlp_input)
    return result


def model_hidden_dim(block: torch.nn.Module) -> int:
    return block.attention.num_heads * block.attention.head_dim


@torch.no_grad()
def forward_streamed_dinat(
    model: AlternatingDilatedGeometricDiT,
    inputs: torch.Tensor,
    *,
    plan: StreamedDiNATPlan,
    conditioning: torch.Tensor | None,
    return_layer_states: bool = False,
    stage_profiler: StreamedStageProfiler | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Run M37 layer-by-layer with CPU global buffers and GPU-local working sets.

    Returns output on the input's device for compatibility with the original
    FlowMatchingTask sampler. Debug states are always CPU tensors; capturing
    them consumes substantial additional host memory on large graphs.

    No gradients, packed multi-graph batches or multi-GPU execution.
    """
    if not isinstance(model, AlternatingDilatedGeometricDiT):
        raise TypeError("M41.3 requires AlternatingDilatedGeometricDiT")
    if model.training:
        raise ValueError("M41.3 is inference-only; call model.eval()")
    if (
        model.sparse_attention_backend != "torch_sparse"
        or model.spatial_dim != plan.base.spatial_dim
        or model.include_self_attention != plan.base.include_self_attention
        or model.include_local_self_attention != plan.base.include_local_self_attention
        or model.coordinate_normalization != plan.base.coordinate_normalization
        or model.coordinate_normalization_eps != plan.base.coordinate_normalization_eps
    ):
        raise ValueError("M41.3 plan does not match the model attention configuration")
    device = model.input_projection.weight.device
    if device != plan.model_device or plan.base.device.type != "cpu":
        raise ValueError("M41.3 plan and model devices do not match")
    if (
        inputs.ndim != 2
        or inputs.shape != (plan.base.num_nodes, model.in_channels)
        or inputs.dtype != torch.float32
        or model.input_projection.weight.dtype != torch.float32
    ):
        raise ValueError("M41.3 inputs must be [N, in_channels] float32")
    if conditioning is not None:
        if conditioning.dtype != torch.float32:
            raise TypeError("M41.3 conditioning must be float32")
        conditioning = conditioning.to(device)

    # Conditioning is one vector per physical graph; its computation does not
    # require the complete set of physical node features on GPU.
    probe = torch.empty((1, model.in_channels), device=device, dtype=torch.float32)
    condition = model._condition_embedding(
        probe, batch_index=None, conditioning=conditioning
    )
    forward_start = time.perf_counter() if stage_profiler is not None else None
    with _phase(stage_profiler, "host_buffer_initialization", device=device):
        input_host = inputs.detach().to("cpu")
        hidden = _host_buffer(plan, model.hidden_dim)
        next_hidden = _host_buffer(plan, model.hidden_dim)

    for part in _partition_iterator(plan.partitions, stage_profiler, device=device):
        with _phase(stage_profiler, "initial_projection_and_readback", device=device):
            result = _initial_owned(model, input_host, plan, part).to("cpu")
        with _phase(stage_profiler, "initial_cpu_scatter", device=device):
            hidden.index_copy_(0, part.nodes.owned_global, result)
    del input_host

    states: list[torch.Tensor] = []
    for layer_index, block in enumerate(model.blocks):
        layer_start = time.perf_counter() if stage_profiler is not None else None
        with _phase(stage_profiler, "gpu_layer_modulation", device=device, layer_index=layer_index):
            modulation = block.adaLN_modulation(condition)
            shifts = modulation.chunk(6, dim=-1)

        # Jacobi synchronization: ALL partitions read the immutable previous
        # hidden buffer. Nothing reads current-layer output until the next layer.
        # Blocking GPU->CPU readback of owned states is the effective barrier.
        for part in _partition_iterator(
            plan.partitions, stage_profiler, device=device, layer_index=layer_index
        ):
            predicted = _update_owned(
                block, hidden, part, layer_index=layer_index,
                shifts=shifts, device=device, stage_profiler=stage_profiler,
            )
            with _phase(stage_profiler, "d2h_owned_readback", device=device, layer_index=layer_index):
                updated = predicted.to("cpu")
            with _phase(stage_profiler, "cpu_owned_scatter", device=device, layer_index=layer_index):
                next_hidden.index_copy_(0, part.nodes.owned_global, updated)
        with _phase(stage_profiler, "jacobi_buffer_swap", device=device, layer_index=layer_index):
            hidden, next_hidden = next_hidden, hidden
        if return_layer_states:
            states.append(hidden.clone())
        if stage_profiler is not None:
            stage_profiler.layer_wall_seconds[layer_index] = time.perf_counter() - layer_start

    output_host = _host_buffer(plan, model.out_channels)
    for part in _partition_iterator(plan.partitions, stage_profiler, device=device):
        owned = part.nodes.owned_global
        with _phase(stage_profiler, "final_projection_and_readback", device=device):
            owned_hidden = hidden.index_select(0, owned).to(device)
            updated = model.final_layer(owned_hidden, condition, batch_index=None)
            result = updated.to("cpu")
        with _phase(stage_profiler, "final_cpu_scatter", device=device):
            output_host.index_copy_(0, owned, result)
    # The original sampler retains global *physical* states, not hidden states.
    with _phase(stage_profiler, "output_return_to_input_device", device=device):
        output = output_host.to(inputs.device)
    if stage_profiler is not None:
        stage_profiler._sync(device)
        stage_profiler.forward_wall_seconds = time.perf_counter() - forward_start
    return (output, states) if return_layer_states else output
