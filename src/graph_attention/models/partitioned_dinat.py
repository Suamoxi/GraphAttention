"""M41.1: exact layerwise halo inference for a single DiNAT-DiT physical graph.

Reuses trained DiT blocks unchanged. Each partition updates only owned nodes
using all original-graph incoming one-hop / exact-two-hop attention edges.
All partitions read the same previous-layer hidden tensor.

M41.1 keeps the global hidden tensor on one GPU. It does not yet implement
out-of-core storage, multiple GPUs, partitioned gradients or training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from graph_attention.geometry import edge_relative_displacement
from graph_attention.geometry.partitioning import validate_partition_ids

from .dinat_dit import AlternatingDilatedGeometricDiT
from .dit_common import _normalize_coordinates_by_graph
from .sparse_transformer import (
    _TorchSparseTopology,
    _build_torch_sparse_topology,
    _validate_edge_index,
    _validate_model_inputs,
)


@dataclass(frozen=True)
class PartitionAttention:
    """Original incoming edges for owned targets, using partition-local IDs."""

    edge_index: torch.Tensor
    displacement: torch.Tensor
    sparse_topology: _TorchSparseTopology | None


@dataclass(frozen=True)
class OwnedHaloPartition:
    partition_id: int
    owned_global: torch.Tensor
    read_global: torch.Tensor
    owned_local: torch.Tensor
    local: PartitionAttention
    dilated: PartitionAttention

    @property
    def num_halo_nodes(self) -> int:
        return int(self.read_global.numel() - self.owned_global.numel())


@dataclass(frozen=True)
class DiNATPartitionPlan:
    """Static mesh/attention geometry cached across all Flow Matching steps."""

    num_nodes: int
    spatial_dim: int
    device: torch.device
    model_coords: torch.Tensor
    backend: str
    include_self_attention: bool
    include_local_self_attention: bool
    coordinate_normalization: str
    coordinate_normalization_eps: float
    partitions: tuple[OwnedHaloPartition, ...]


def _attention_for_owned_targets(
    global_edges: torch.Tensor,
    *,
    owner: torch.Tensor,
    partition_id: int,
    read_global: torch.Tensor,
    coords: torch.Tensor,
    backend: str,
) -> PartitionAttention:
    # Edge convention in GraphAttention: edge[0] = key/source,
    # edge[1] = query/target. Every owned target keeps ALL incoming edges.
    selected = global_edges[:, owner[global_edges[1]] == partition_id]
    local_edges = torch.searchsorted(read_global, selected).contiguous()
    displacement = edge_relative_displacement(
        coords.index_select(0, read_global), local_edges
    )
    topology = None
    if backend == "torch_sparse" and local_edges.shape[1]:
        # Cached CSR must be normal tensors, not inference-only tensors.
        with torch.inference_mode(False):
            topology = _build_torch_sparse_topology(
                local_edges, num_nodes=int(read_global.numel())
            )
    return PartitionAttention(local_edges, displacement, topology)


def build_dinat_partition_plan(
    model: AlternatingDilatedGeometricDiT,
    *,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    attention_edge_indices: dict[str, torch.Tensor],
    partition_ids: torch.Tensor,
) -> DiNATPartitionPlan:
    """Build one partition plan from the ORIGINAL global graph and coordinates.

    The graph must be a single physical sample. Construct this plan once,
    outside torch.inference_mode(), and reuse it for every model evaluation.
    """
    if not isinstance(model, AlternatingDilatedGeometricDiT):
        raise TypeError("M41.1 supports AlternatingDilatedGeometricDiT only")
    if coords.ndim != 2 or coords.shape[1] != model.spatial_dim:
        raise ValueError("coords have the wrong spatial shape")
    if not coords.is_floating_point() or not bool(torch.isfinite(coords).all()):
        raise ValueError("coords must be finite floating-point values")
    if (
        coords.device != model.input_projection.weight.device
        or coords.dtype != model.input_projection.weight.dtype
    ):
        raise ValueError("coords and model parameters must share device/dtype")

    num_nodes = int(coords.shape[0])
    _validate_edge_index(edge_index, num_nodes=num_nodes, device=coords.device)
    if "dilated" not in attention_edge_indices:
        raise ValueError("attention_edge_indices['dilated'] is required")
    dilated = attention_edge_indices["dilated"]
    _validate_edge_index(dilated, num_nodes=num_nodes, device=coords.device)

    owner = partition_ids.to(device=coords.device)
    partition_count = validate_partition_ids(owner, num_nodes)

    # Exactly match M37/M40 self-loop configuration; do not add periodic edges.
    local_edges = model._attention_edges_with_optional_self(
        "local", edge_index, num_nodes=num_nodes
    )
    dilated_edges = model._attention_edges_with_optional_self(
        "dilated", dilated, num_nodes=num_nodes
    )

    model_coords = coords
    if model.coordinate_normalization == "centered_bbox":
        model_coords = _normalize_coordinates_by_graph(
            coords,
            batch_index=None,
            eps=model.coordinate_normalization_eps,
        )

    partitions: list[OwnedHaloPartition] = []
    for partition_id in range(partition_count):
        owned_global = torch.nonzero(owner == partition_id).flatten()
        onehop_keys = local_edges[0, owner[local_edges[1]] == partition_id]
        twohop_keys = dilated_edges[0, owner[dilated_edges[1]] == partition_id]
        read_global = torch.unique(
            torch.cat((owned_global, onehop_keys, twohop_keys)), sorted=True
        )
        owned_local = torch.searchsorted(read_global, owned_global)
        partitions.append(
            OwnedHaloPartition(
                partition_id=partition_id,
                owned_global=owned_global,
                read_global=read_global,
                owned_local=owned_local,
                local=_attention_for_owned_targets(
                    local_edges,
                    owner=owner,
                    partition_id=partition_id,
                    read_global=read_global,
                    coords=coords,
                    backend=model.sparse_attention_backend,
                ),
                dilated=_attention_for_owned_targets(
                    dilated_edges,
                    owner=owner,
                    partition_id=partition_id,
                    read_global=read_global,
                    coords=coords,
                    backend=model.sparse_attention_backend,
                ),
            )
        )

    return DiNATPartitionPlan(
        num_nodes=num_nodes,
        spatial_dim=model.spatial_dim,
        device=coords.device,
        model_coords=model_coords,
        backend=model.sparse_attention_backend,
        include_self_attention=model.include_self_attention,
        include_local_self_attention=model.include_local_self_attention,
        coordinate_normalization=model.coordinate_normalization,
        coordinate_normalization_eps=model.coordinate_normalization_eps,
        partitions=tuple(partitions),
    )


@torch.no_grad()
def forward_partitioned_dinat(
    model: AlternatingDilatedGeometricDiT,
    inputs: torch.Tensor,
    *,
    plan: DiNATPartitionPlan,
    conditioning: torch.Tensor | None,
    return_layer_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Layerwise Jacobi-style halo refresh, exact original owned-node attention.

    This is a single-physical-graph inference API (no artificial partition
    batch_index). For debugging, return_layer_states adds one global hidden
    tensor per transformer layer and should be disabled in production.
    """
    if not isinstance(model, AlternatingDilatedGeometricDiT):
        raise TypeError("M41.1 supports AlternatingDilatedGeometricDiT only")
    if model.training:
        raise ValueError("M41.1 is inference-only; call model.eval()")
    if (
        model.spatial_dim != plan.spatial_dim
        or model.sparse_attention_backend != plan.backend
        or model.include_self_attention != plan.include_self_attention
        or model.include_local_self_attention != plan.include_local_self_attention
        or model.coordinate_normalization != plan.coordinate_normalization
        or model.coordinate_normalization_eps != plan.coordinate_normalization_eps
    ):
        raise ValueError("partition plan does not match the model configuration")
    _validate_model_inputs(
        inputs,
        expected_channels=model.in_channels,
        parameter=model.input_projection.weight,
    )
    if inputs.shape[0] != plan.num_nodes or inputs.device != plan.device:
        raise ValueError("input count or device differs from the partition plan")

    condition = model._condition_embedding(
        inputs, batch_index=None, conditioning=conditioning
    )
    hidden = torch.empty(
        (plan.num_nodes, model.hidden_dim), device=inputs.device, dtype=inputs.dtype
    )
    for part in plan.partitions:
        core = part.owned_global
        projected = (
            model.input_projection(inputs.index_select(0, core))
            + model.coordinate_projection(plan.model_coords.index_select(0, core))
        )
        hidden.index_copy_(0, core, projected)

    states: list[torch.Tensor] = []
    for layer_index, block in enumerate(model.blocks):
        # NEVER mutate the current layer in place: all partitions read the
        # same layer-l hidden states. Halo copies refresh after each layer.
        next_hidden = torch.empty_like(hidden)
        for part in plan.partitions:
            read_hidden = hidden.index_select(0, part.read_global)
            attention = part.local if layer_index % 2 == 0 else part.dilated
            kwargs: dict[str, Any] = {
                "edge_displacement": attention.displacement,
            }
            if attention.sparse_topology is not None:
                kwargs["sparse_adj"] = attention.sparse_topology
            updated = block(
                read_hidden,
                condition,
                edge_index=attention.edge_index,
                batch_index=None,
                attention_kwargs=kwargs,
            )
            next_hidden.index_copy_(
                0,
                part.owned_global,
                updated.index_select(0, part.owned_local),
            )
        hidden = next_hidden
        if return_layer_states:
            states.append(hidden.clone())

    result = torch.empty(
        (plan.num_nodes, model.out_channels), device=inputs.device, dtype=inputs.dtype
    )
    for part in plan.partitions:
        core = part.owned_global
        result.index_copy_(
            0,
            core,
            model.final_layer(
                hidden.index_select(0, core), condition, batch_index=None
            ),
        )
    return (result, states) if return_layer_states else result
