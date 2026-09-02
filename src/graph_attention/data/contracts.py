"""Core M2 scientific data contracts.

These contracts describe what exists in a CFD sample. They deliberately avoid
model, task, batching, and preprocessing behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any

import torch


class FieldSupport(StrEnum):
    """Spatial support of a physical or metadata quantity."""

    NODE = "node"
    CELL = "cell"
    FACE = "face"
    GLOBAL = "global"


class FieldRole(StrEnum):
    """Scientific role of a field in the source data."""

    PRIMARY_STATE = "primary_state"
    SPECIES_STATE = "species_state"
    AUXILIARY_PHYSICAL = "auxiliary_physical"
    DERIVED_PHYSICAL = "derived_physical"
    GEOMETRY_BOUNDARY = "geometry_boundary"
    DIAGNOSTIC = "diagnostic"
    COMPUTATIONAL_METADATA = "computational_metadata"
    FORCING_INTERNAL = "forcing_internal"
    GLOBAL_METADATA = "global_metadata"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Semantic description of one supported CFD quantity."""

    name: str
    support: FieldSupport
    role: FieldRole
    source_path: str | None = None
    components: tuple[str, ...] = ()
    units: str | None = None
    provenance: str | None = None
    stored: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("field name must be non-empty")
        if self.source_path == "":
            raise ValueError("source_path must be non-empty when provided")
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"field '{self.name}' has duplicate component names")
        if any(not component for component in self.components):
            raise ValueError(f"field '{self.name}' has an empty component name")

    @property
    def component_count(self) -> int:
        """Number of explicit components, or one for a scalar field."""

        return len(self.components) or 1


class FieldCatalog:
    """Ordered catalogue of supported fields keyed by canonical name."""

    def __init__(self, fields: tuple[FieldSpec, ...] | list[FieldSpec]) -> None:
        self._fields = tuple(fields)
        names = [field.name for field in self._fields]
        if len(set(names)) != len(names):
            duplicates = sorted(name for name in set(names) if names.count(name) > 1)
            raise ValueError(f"duplicate field names: {duplicates}")
        self._by_name = {field.name: field for field in self._fields}

    def __iter__(self):
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def get(self, name: str) -> FieldSpec:
        """Return one declared field or fail with a semantic error."""

        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown field '{name}'") from exc

    def require(self, names: list[str] | tuple[str, ...]) -> tuple[FieldSpec, ...]:
        """Resolve field names while preserving the requested order."""

        return tuple(self.get(name) for name in names)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self._fields)


@dataclass(frozen=True, slots=True)
class ReferenceScale:
    """One case-level physical reference quantity."""

    name: str
    value: float
    definition: str
    provenance: str
    units: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("reference scale name must be non-empty")
        if not self.definition:
            raise ValueError(f"reference scale '{self.name}' needs a definition")
        if not self.provenance:
            raise ValueError(f"reference scale '{self.name}' needs provenance")
        if not isfinite(self.value):
            raise ValueError(f"reference scale '{self.name}' must be finite")


class ReferenceScales:
    """Named case-level reference quantities with explicit semantics."""

    def __init__(self, scales: tuple[ReferenceScale, ...] | list[ReferenceScale] = ()) -> None:
        self._scales = tuple(scales)
        names = [scale.name for scale in self._scales]
        if len(set(names)) != len(names):
            duplicates = sorted(name for name in set(names) if names.count(name) > 1)
            raise ValueError(f"duplicate reference scale names: {duplicates}")
        self._by_name = {scale.name: scale for scale in self._scales}

    def __iter__(self):
        return iter(self._scales)

    def __len__(self) -> int:
        return len(self._scales)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def get(self, name: str) -> ReferenceScale:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown reference scale '{name}'") from exc

    def value(self, name: str) -> float:
        return self.get(name).value


@dataclass(slots=True)
class Mesh:
    """Canonical M2 representation of one native node-based CFD mesh."""

    coords: torch.Tensor
    edge_index: torch.Tensor
    mesh_id: str | None = None
    node_weights: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.coords.ndim != 2:
            raise ValueError("coords must have shape [N, D]")
        if not self.coords.is_floating_point():
            raise TypeError("coords must use a floating-point dtype")
        if self.coords.numel() and not torch.isfinite(self.coords).all().item():
            raise ValueError("coords must be finite")

        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long indices")

        if self.edge_index.numel():
            min_index = int(self.edge_index.min())
            max_index = int(self.edge_index.max())
            if min_index < 0 or max_index >= self.num_nodes:
                raise ValueError(
                    "edge_index contains node indices outside the valid "
                    f"[0, {self.num_nodes}) range"
                )

        if self.node_weights is not None:
            if self.node_weights.ndim != 1 or self.node_weights.shape[0] != self.num_nodes:
                raise ValueError("node_weights must have shape [N]")
            if not self.node_weights.is_floating_point():
                raise TypeError("node_weights must use a floating-point dtype")
            if self.node_weights.numel() and not torch.isfinite(self.node_weights).all().item():
                raise ValueError("node_weights must be finite")

    @property
    def num_nodes(self) -> int:
        return self.coords.shape[0]

    @property
    def spatial_dim(self) -> int:
        return self.coords.shape[1]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


@dataclass(slots=True)
class Sample:
    """One CFD sample plus its native mesh and case-level metadata."""

    sample_id: str
    mesh: Mesh
    fields: Mapping[str, torch.Tensor]
    reference_scales: ReferenceScales = field(default_factory=ReferenceScales)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        self.fields = dict(self.fields)

    def validate_against(self, catalog: FieldCatalog) -> None:
        """Validate loaded tensors against declared semantic field contracts."""

        for name, tensor in self.fields.items():
            spec = catalog.get(name)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"field '{name}' must be a torch.Tensor")

            if spec.support == FieldSupport.NODE:
                self._validate_node_field(spec, tensor)

    def _validate_node_field(self, spec: FieldSpec, tensor: torch.Tensor) -> None:
        if tensor.ndim == 0:
            raise ValueError(f"node field '{spec.name}' must have a node dimension")
        if tensor.shape[0] != self.mesh.num_nodes:
            raise ValueError(
                f"node field '{spec.name}' has {tensor.shape[0]} nodes, "
                f"expected {self.mesh.num_nodes}"
            )

        component_count = spec.component_count
        if component_count == 1:
            if tensor.ndim not in {1, 2}:
                raise ValueError(
                    f"scalar node field '{spec.name}' must have shape [N] or [N, 1]"
                )
            if tensor.ndim == 2 and tensor.shape[1] != 1:
                raise ValueError(f"scalar node field '{spec.name}' must have one component")
        else:
            if tensor.ndim != 2 or tensor.shape[1] != component_count:
                raise ValueError(
                    f"node field '{spec.name}' must have shape "
                    f"[N, {component_count}] for components {spec.components}"
                )
