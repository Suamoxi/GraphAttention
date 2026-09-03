"""Budget-aware packed batching for variable node graphs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from operator import index as operator_index
from typing import Any

import torch

from .contracts import ReferenceScales, RegimeParameters, Sample


@dataclass(frozen=True, slots=True)
class MicrobatchBudget:
    """Hard computational limits for one packed microbatch."""

    max_nodes: int
    max_edges: int

    def __post_init__(self) -> None:
        if isinstance(self.max_nodes, bool) or isinstance(self.max_edges, bool):
            raise TypeError("microbatch budgets must be integer counts, not bool values")
        try:
            max_nodes = operator_index(self.max_nodes)
            max_edges = operator_index(self.max_edges)
        except TypeError as exc:
            raise TypeError("microbatch budgets must be integer counts") from exc
        if max_nodes <= 0 or max_edges <= 0:
            raise ValueError("microbatch max_nodes and max_edges must both be positive")
        object.__setattr__(self, "max_nodes", max_nodes)
        object.__setattr__(self, "max_edges", max_edges)


@dataclass(frozen=True, slots=True)
class PackedBatch:
    """Disconnected packed representation of multiple physical CFD samples.

    Named node fields remain separate so M4 does not choose task channel order.
    The later task layer may concatenate an explicit ordered subset into model
    state while retaining field semantics.
    """

    coords: torch.Tensor
    edge_index: torch.Tensor
    batch_index: torch.Tensor
    ptr: torch.Tensor
    fields: Mapping[str, torch.Tensor]
    node_weights: torch.Tensor | None
    sample_ids: tuple[str, ...]
    mesh_ids: tuple[str | None, ...]
    case_ids: tuple[str | None, ...]
    reference_scales: tuple[ReferenceScales, ...]
    regime_parameters: tuple[RegimeParameters, ...]
    sample_metadata: tuple[Mapping[str, Any], ...]
    mesh_metadata: tuple[Mapping[str, Any], ...]

    @property
    def num_graphs(self) -> int:
        return len(self.sample_ids)

    @property
    def num_nodes(self) -> int:
        return self.coords.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


def partition_samples_by_budget(
    samples: Iterable[Sample],
    budget: MicrobatchBudget,
) -> tuple[tuple[Sample, ...], ...]:
    """Partition an already selected sample sequence without reordering it.

    Sampling policy remains external. This function only creates contiguous
    computational groups whose current node and edge counts satisfy ``budget``.
    """

    selected = tuple(samples)
    if not selected:
        return ()

    groups: list[tuple[Sample, ...]] = []
    current: list[Sample] = []
    current_nodes = 0
    current_edges = 0

    for sample in selected:
        nodes = sample.mesh.num_nodes
        edges = sample.mesh.num_edges
        _validate_nonempty_sample(sample)
        if nodes > budget.max_nodes or edges > budget.max_edges:
            raise ValueError(
                f"sample '{sample.sample_id}' exceeds microbatch budget: "
                f"nodes={nodes} (max {budget.max_nodes}), "
                f"edges={edges} (max {budget.max_edges})"
            )

        exceeds_with_current = (
            current_nodes + nodes > budget.max_nodes or current_edges + edges > budget.max_edges
        )
        if current and exceeds_with_current:
            groups.append(tuple(current))
            current = []
            current_nodes = 0
            current_edges = 0

        current.append(sample)
        current_nodes += nodes
        current_edges += edges

    if current:
        groups.append(tuple(current))
    return tuple(groups)


def pack_samples(
    samples: Iterable[Sample],
    *,
    node_field_names: Iterable[str] = (),
) -> PackedBatch:
    """Pack selected node graphs into one disconnected graph.

    ``node_field_names`` is explicit so the packer never infers field support or
    task semantics from tensor shape. Each selected field is concatenated only
    along its node dimension and remains named in the returned mapping.
    """

    selected = tuple(samples)
    if not selected:
        raise ValueError("cannot pack an empty sample collection")
    if isinstance(node_field_names, str):
        raise TypeError("node_field_names must be an iterable of names, not one string")

    field_names = tuple(node_field_names)
    if any(not isinstance(name, str) or not name.strip() for name in field_names):
        raise ValueError("node_field_names must contain only non-empty strings")
    if len(set(field_names)) != len(field_names):
        raise ValueError("node_field_names must not contain duplicates")

    first = selected[0]
    _validate_nonempty_sample(first)
    spatial_dim = first.mesh.spatial_dim
    coords_dtype = first.mesh.coords.dtype
    device = first.mesh.coords.device

    if first.mesh.edge_index.device != device:
        raise ValueError(
            f"sample '{first.sample_id}' coords and edge_index must be on the same device"
        )

    has_node_weights = first.mesh.node_weights is not None
    node_weight_dtype = first.mesh.node_weights.dtype if has_node_weights else None

    field_trailing_shapes: dict[str, torch.Size] = {}
    field_dtypes: dict[str, torch.dtype] = {}
    for name in field_names:
        value = _require_node_field(first, name)
        if value.device != device:
            raise ValueError(
                f"sample '{first.sample_id}' field '{name}' must be on device {device}"
            )
        field_trailing_shapes[name] = value.shape[1:]
        field_dtypes[name] = value.dtype

    node_counts: list[int] = []
    offsets: list[int] = []
    running_nodes = 0

    for sample in selected:
        _validate_nonempty_sample(sample)
        mesh = sample.mesh
        if mesh.spatial_dim != spatial_dim:
            raise ValueError(
                "all packed samples must share one coordinate dimension; "
                f"sample '{sample.sample_id}' has D={mesh.spatial_dim}, "
                f"expected D={spatial_dim}"
            )
        if mesh.coords.dtype != coords_dtype:
            raise TypeError(
                "all packed coordinate tensors must share one dtype; "
                f"sample '{sample.sample_id}' has {mesh.coords.dtype}, "
                f"expected {coords_dtype}"
            )
        if mesh.coords.device != device:
            raise ValueError(
                "all packed tensors must share one device; "
                f"sample '{sample.sample_id}' coords are on {mesh.coords.device}, "
                f"expected {device}"
            )
        if mesh.edge_index.device != device:
            raise ValueError(
                f"sample '{sample.sample_id}' coords and edge_index must be on the same device"
            )

        sample_has_weights = mesh.node_weights is not None
        if sample_has_weights != has_node_weights:
            raise ValueError(
                "cannot mix samples with and without node_weights in one packed batch; "
                "an explicit task-level weighting policy is required"
            )
        if mesh.node_weights is not None:
            if mesh.node_weights.device != device:
                raise ValueError(
                    f"sample '{sample.sample_id}' node_weights must be on device {device}"
                )
            if mesh.node_weights.dtype != node_weight_dtype:
                raise TypeError(
                    "all packed node_weights tensors must share one dtype; "
                    f"sample '{sample.sample_id}' has {mesh.node_weights.dtype}, "
                    f"expected {node_weight_dtype}"
                )

        for name in field_names:
            value = _require_node_field(sample, name)
            if value.shape[1:] != field_trailing_shapes[name]:
                raise ValueError(
                    f"node field '{name}' must have consistent trailing shape; "
                    f"sample '{sample.sample_id}' has {tuple(value.shape[1:])}, "
                    f"expected {tuple(field_trailing_shapes[name])}"
                )
            if value.dtype != field_dtypes[name]:
                raise TypeError(
                    f"node field '{name}' must have one dtype across the packed batch; "
                    f"sample '{sample.sample_id}' has {value.dtype}, "
                    f"expected {field_dtypes[name]}"
                )
            if value.device != device:
                raise ValueError(
                    f"sample '{sample.sample_id}' field '{name}' must be on device {device}"
                )

        offsets.append(running_nodes)
        node_counts.append(mesh.num_nodes)
        running_nodes += mesh.num_nodes

    coords = torch.cat([sample.mesh.coords for sample in selected], dim=0)
    edge_parts = [
        sample.mesh.edge_index + offset
        for sample, offset in zip(selected, offsets, strict=True)
        if sample.mesh.num_edges > 0
    ]
    edge_index = (
        torch.cat(edge_parts, dim=1)
        if edge_parts
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )

    counts_tensor = torch.tensor(node_counts, dtype=torch.long, device=device)
    batch_index = torch.repeat_interleave(
        torch.arange(len(selected), dtype=torch.long, device=device),
        counts_tensor,
    )
    ptr = torch.empty(len(selected) + 1, dtype=torch.long, device=device)
    ptr[0] = 0
    ptr[1:] = counts_tensor.cumsum(dim=0)

    fields = {
        name: torch.cat([sample.fields[name] for sample in selected], dim=0) for name in field_names
    }
    node_weights = (
        torch.cat([sample.mesh.node_weights for sample in selected], dim=0)
        if has_node_weights
        else None
    )

    return PackedBatch(
        coords=coords,
        edge_index=edge_index,
        batch_index=batch_index,
        ptr=ptr,
        fields=fields,
        node_weights=node_weights,
        sample_ids=tuple(sample.sample_id for sample in selected),
        mesh_ids=tuple(sample.mesh.mesh_id for sample in selected),
        case_ids=tuple(sample.case_id for sample in selected),
        reference_scales=tuple(sample.reference_scales for sample in selected),
        regime_parameters=tuple(sample.regime_parameters for sample in selected),
        sample_metadata=tuple(sample.metadata for sample in selected),
        mesh_metadata=tuple(sample.mesh.metadata for sample in selected),
    )


def _validate_nonempty_sample(sample: Sample) -> None:
    if sample.mesh.num_nodes <= 0:
        raise ValueError(f"sample '{sample.sample_id}' has no mesh nodes and cannot be packed")


def _require_node_field(sample: Sample, name: str) -> torch.Tensor:
    try:
        value = sample.fields[name]
    except KeyError as exc:
        raise KeyError(f"sample '{sample.sample_id}' does not contain node field '{name}'") from exc
    if value.ndim == 0 or value.shape[0] != sample.mesh.num_nodes:
        raise ValueError(
            f"selected node field '{name}' in sample '{sample.sample_id}' must have leading "
            f"dimension N={sample.mesh.num_nodes}, got shape {tuple(value.shape)}"
        )
    return value
