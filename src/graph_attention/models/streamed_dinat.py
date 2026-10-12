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
from dataclasses import dataclass, replace

import torch

from .dinat_dit import AlternatingDilatedGeometricDiT
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


def _update_owned(
    block: torch.nn.Module,
    current_host: torch.Tensor,
    part: OwnedQueryPartition,
    *,
    layer_index: int,
    shifts: tuple[torch.Tensor, ...],
    device: torch.device,
) -> torch.Tensor:
    """Only one partition's hidden/QKV/attention/MLP tensors live on GPU."""
    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = shifts
    nodes = part.nodes
    read_hidden = current_host.index_select(0, nodes.read_global).to(device)
    owned_local = nodes.owned_local.to(device)

    attention_input = _modulate(block.norm1(read_hidden), shift_attn, scale_attn)
    attention_module = block.attention
    qkv = attention_module.qkv(attention_input).reshape(
        read_hidden.shape[0], 3, attention_module.num_heads, attention_module.head_dim
    )
    attention = nodes.local if layer_index % 2 == 0 else nodes.dilated
    topology = part.local if layer_index % 2 == 0 else part.dilated
    # Edge indices are only used for their edge count in _owned_context.
    # Transfer displacements and rectangular CSR, but not a global graph.
    staged_attention = replace(
        attention, displacement=attention.displacement.to(device)
    )
    scores = _owned_context(
        attention_module,
        qkv[:, 0].index_select(0, owned_local),
        qkv[:, 1],
        qkv[:, 2],
        staged_attention,
        _gpu_topology(topology, device),
    )
    owned_hidden = read_hidden.index_select(0, owned_local)
    owned_hidden = owned_hidden + gate_attn * attention_module.out_proj(
        scores.reshape(owned_local.numel(), model_hidden_dim(block))
    )
    mlp_input = _modulate(block.norm2(owned_hidden), shift_mlp, scale_mlp)
    return owned_hidden + gate_mlp * block.mlp(mlp_input)


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
    input_host = inputs.detach().to("cpu")
    hidden = _host_buffer(plan, model.hidden_dim)
    next_hidden = _host_buffer(plan, model.hidden_dim)

    for part in plan.partitions:
        result = _initial_owned(model, input_host, plan, part).to("cpu")
        hidden.index_copy_(0, part.nodes.owned_global, result)
    del input_host

    states: list[torch.Tensor] = []
    for layer_index, block in enumerate(model.blocks):
        modulation = block.adaLN_modulation(condition)
        shifts = modulation.chunk(6, dim=-1)

        # Jacobi synchronization: ALL partitions read the immutable previous
        # hidden buffer. Nothing reads current-layer output until the next layer.
        for part in plan.partitions:
            updated = _update_owned(
                block, hidden, part, layer_index=layer_index,
                shifts=shifts, device=device,
            ).to("cpu")
            next_hidden.index_copy_(0, part.nodes.owned_global, updated)
        hidden, next_hidden = next_hidden, hidden
        if return_layer_states:
            states.append(hidden.clone())

    output_host = _host_buffer(plan, model.out_channels)
    for part in plan.partitions:
        owned = part.nodes.owned_global
        owned_hidden = hidden.index_select(0, owned).to(device)
        updated = model.final_layer(owned_hidden, condition, batch_index=None)
        output_host.index_copy_(0, owned, updated.to("cpu"))
    # The original sampler retains global *physical* states, not hidden states.
    output = output_host.to(inputs.device)
    return (output, states) if return_layer_states else output
