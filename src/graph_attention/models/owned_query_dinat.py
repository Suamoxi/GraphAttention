"""M41.2: inference-only DiNAT with owned queries and full halo keys/values.

The validated M41.1 full-block executor is left unchanged. This executor
reuses M37's trained modules and original global attention edges, but avoids
computing halo-node attention outputs, residual updates, and MLP updates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from graph_attention.geometry.partitioning import validate_partition_ids

from .dinat_dit import AlternatingDilatedGeometricDiT
from .dit_common import _modulate
from .partitioned_dinat import (
    DiNATPartitionPlan,
    OwnedHaloPartition,
    PartitionAttention,
    build_dinat_partition_plan,
)
from .sparse_transformer import _validate_model_inputs


@dataclass(frozen=True)
class OwnedQueryTopology:
    """Rectangular target-owned (rows) by owned-plus-halo (columns) CSR."""

    pattern: torch.Tensor
    row_indices: torch.Tensor
    col_indices: torch.Tensor
    edge_order: torch.Tensor

    @property
    def num_queries(self) -> int:
        return self.pattern.shape[0]

    @property
    def num_keys(self) -> int:
        return self.pattern.shape[1]


@dataclass(frozen=True)
class OwnedQueryPartition:
    nodes: OwnedHaloPartition
    local: OwnedQueryTopology
    dilated: OwnedQueryTopology


@dataclass(frozen=True)
class OwnedQueryPlan:
    """An immutable M41.1 ownership plan plus rectangular attention patterns."""

    base: DiNATPartitionPlan
    partitions: tuple[OwnedQueryPartition, ...]


def _rectangular_topology(
    attention: PartitionAttention,
    *,
    owned_local: torch.Tensor,
    num_read_nodes: int,
    dtype: torch.dtype,
) -> OwnedQueryTopology:
    source, target = attention.edge_index
    # Every retained target must be owned. Source indices remain in the
    # complete owned-plus-halo read set, including cross-partition sources.
    target_row = torch.searchsorted(owned_local, target)
    if bool((target_row >= owned_local.numel()).any()) or not torch.equal(
        owned_local.index_select(0, target_row), target
    ):
        raise ValueError("partition attention contains a non-owned target")
    edge_order = torch.argsort(target_row * num_read_nodes + source)
    row_indices = target_row[edge_order]
    col_indices = source[edge_order]
    row_counts = torch.bincount(row_indices, minlength=owned_local.numel())
    crow_indices = torch.cat(
        (row_counts.new_zeros(1), row_counts.cumsum(0))
    )
    if int(crow_indices[-1]) != source.numel():
        raise ValueError("incomplete CSR attention rows")
    if source.numel() > 1:
        sorted_linear = row_indices * num_read_nodes + col_indices
        if bool((sorted_linear[1:] == sorted_linear[:-1]).any()):
            raise ValueError("rectangular CSR attention requires unique directed edges")

    # Cached CSR metadata must be normal tensors even when a caller builds a
    # plan inside torch.inference_mode(). The pattern is immutable thereafter.
    pattern = torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        torch.ones(source.numel(), dtype=dtype, device=source.device),
        size=(owned_local.numel(), num_read_nodes),
        device=source.device,
    )
    return OwnedQueryTopology(pattern, row_indices, col_indices, edge_order)


@torch.inference_mode(False)
@torch.no_grad()
def build_owned_query_plan(
    model: AlternatingDilatedGeometricDiT,
    *,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    attention_edge_indices: dict[str, torch.Tensor],
    partition_ids: torch.Tensor,
) -> OwnedQueryPlan:
    """Build one plan per immutable graph, reusing M41.1's ownership contract.

    The original square M41.1 CSR caches are dropped from the returned plan:
    only rectangular owned-query patterns are needed at inference. This
    leaves the M41.1 plan constructor and executor behavior unchanged.
    """
    if model.sparse_attention_backend != "torch_sparse":
        raise ValueError("M41.2 owned-query execution requires torch_sparse")
    base = build_dinat_partition_plan(
        model,
        coords=coords,
        edge_index=edge_index,
        attention_edge_indices=attention_edge_indices,
        partition_ids=partition_ids,
    )
    validate_partition_ids(partition_ids.to(base.device), base.num_nodes)
    optimized = []
    stripped = []
    for part in base.partitions:
        optimized.append(
            OwnedQueryPartition(
                nodes=replace(
                    part,
                    local=replace(part.local, sparse_topology=None),
                    dilated=replace(part.dilated, sparse_topology=None),
                ),
                local=_rectangular_topology(
                    part.local,
                    owned_local=part.owned_local,
                    num_read_nodes=part.read_global.numel(),
                    dtype=base.model_coords.dtype,
                ),
                dilated=_rectangular_topology(
                    part.dilated,
                    owned_local=part.owned_local,
                    num_read_nodes=part.read_global.numel(),
                    dtype=base.model_coords.dtype,
                ),
            )
        )
        stripped.append(optimized[-1].nodes)
    return OwnedQueryPlan(
        base=replace(base, partitions=tuple(stripped)),
        partitions=tuple(optimized),
    )


def _owned_context(
    attention_module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention: PartitionAttention,
    topology: OwnedQueryTopology,
) -> torch.Tensor:
    """Aggregate only owned rows; Q/K/V have already been projected globally."""
    if attention.edge_index.shape[1] == 0:
        return torch.zeros_like(query)

    # Native sampled CSR Q_owners @ K_read^T, with shape [owned, read].
    # The CSR is target-major; geometry bias uses the identical edge_order.
    scores_by_head = []
    for head in range(attention_module.num_heads):
        sampled = torch.sparse.sampled_addmm(
            topology.pattern,
            query[:, head, :].contiguous(),
            key[:, head, :].transpose(0, 1).contiguous(),
            beta=0.0,
            alpha=attention_module.scale,
        )
        scores_by_head.append(sampled.values())
    scores = torch.stack(scores_by_head, dim=1)
    scores = scores + attention_module.geometry_mlp(attention.displacement)[topology.edge_order]

    target = topology.row_indices
    by_head = target[:, None].expand(-1, attention_module.num_heads)
    maximum = torch.full(
        (query.shape[0], attention_module.num_heads),
        -torch.inf,
        device=scores.device,
        dtype=scores.dtype,
    )
    maximum.scatter_reduce_(0, by_head, scores, reduce="amax", include_self=True)
    exponent = torch.exp(scores - maximum[target])
    denominator = torch.zeros_like(maximum)
    denominator.scatter_add_(0, by_head, exponent)
    weights = exponent / denominator[target]

    # Inference only. Preserve M37's bounded index_add accumulation while
    # creating results exclusively for the owned query rows.
    aggregated = torch.zeros_like(query)
    for start in range(0, weights.shape[0], 65536):
        stop = min(start + 65536, weights.shape[0])
        messages = weights[start:stop].to(value.dtype).unsqueeze(-1) * value[
            topology.col_indices[start:stop]
        ]
        aggregated.index_add_(0, target[start:stop], messages)
    return aggregated


@torch.no_grad()
def forward_owned_query_dinat(
    model: AlternatingDilatedGeometricDiT,
    inputs: torch.Tensor,
    *,
    plan: OwnedQueryPlan,
    conditioning: torch.Tensor | None,
    return_layer_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Execute exact layerwise halo inference without halo-target DiT outputs.

    The model, original attention topology, global coordinate normalization,
    and complete incoming-edge softmax are unchanged. This does not support
    gradients, multi-graph packing, or DGL/scatter backends.
    """
    if not isinstance(model, AlternatingDilatedGeometricDiT):
        raise TypeError("M41.2 supports AlternatingDilatedGeometricDiT only")
    if model.training:
        raise ValueError("M41.2 is inference-only; call model.eval()")
    base = plan.base
    if (
        model.spatial_dim != base.spatial_dim
        or model.sparse_attention_backend != base.backend
        or model.sparse_attention_backend != "torch_sparse"
        or model.include_self_attention != base.include_self_attention
        or model.include_local_self_attention != base.include_local_self_attention
        or model.coordinate_normalization != base.coordinate_normalization
        or model.coordinate_normalization_eps != base.coordinate_normalization_eps
    ):
        raise ValueError("owned-query plan does not match model configuration")
    _validate_model_inputs(
        inputs,
        expected_channels=model.in_channels,
        parameter=model.input_projection.weight,
    )
    if (
        inputs.shape[0] != base.num_nodes
        or inputs.device != base.device
        or inputs.dtype != base.model_coords.dtype
    ):
        raise ValueError("inputs do not match owned-query plan shape/device/dtype")
    if inputs.dtype != torch.float32:
        raise TypeError("M41.2 torch_sparse attention requires float32")

    condition = model._condition_embedding(inputs, batch_index=None, conditioning=conditioning)
    hidden = model.input_projection(inputs) + model.coordinate_projection(base.model_coords)
    states: list[torch.Tensor] = []

    for layer_idx, block in enumerate(model.blocks):
        # Every physical node is owned exactly once. Compute strictly nodewise
        # work (norm1, QKV, out projection and MLP) globally ONCE, in exactly
        # M37's batch shape. Partition only the attention reduction itself.
        modulation = block.adaLN_modulation(condition)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            modulation.chunk(6, dim=-1)
        )
        attention_input = _modulate(block.norm1(hidden), shift_attn, scale_attn)
        attn = block.attention
        qkv = attn.qkv(attention_input).reshape(
            base.num_nodes, 3, attn.num_heads, attn.head_dim
        )
        all_query = qkv[:, 0]
        all_keyvalue = qkv[:, 1:]
        context = torch.empty_like(all_query)

        for p in plan.partitions:
            part = p.nodes
            attention = part.local if layer_idx % 2 == 0 else part.dilated
            topology = p.local if layer_idx % 2 == 0 else p.dilated
            # Gather exactly the owned query rows and their complete key/value
            # halo. Every partition reads the same previous-layer global QKV.
            key, value = all_keyvalue.index_select(0, part.read_global).unbind(dim=1)
            owned = _owned_context(
                attn,
                all_query.index_select(0, part.owned_global),
                key,
                value,
                attention,
                topology,
            )
            context.index_copy_(0, part.owned_global, owned)

        # Preserve M37's full-sized out-projection, residual and adaLN MLP.
        # Only context computation is partitioned, so halo outputs and
        # duplicated per-partition MLP work are eliminated.
        hidden = hidden + gate_attn * attn.out_proj(
            context.reshape(base.num_nodes, model.hidden_dim)
        )
        mlp_input = _modulate(block.norm2(hidden), shift_mlp, scale_mlp)
        hidden = hidden + gate_mlp * block.mlp(mlp_input)
        if return_layer_states:
            states.append(hidden.clone())

    result = model.final_layer(hidden, condition, batch_index=None)
    return (result, states) if return_layer_states else result
