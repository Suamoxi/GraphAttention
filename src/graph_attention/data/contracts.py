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
    SOLVER_METADATA = "solver_metadata"
    FORCING_INTERNAL = "forcing_internal"
    GLOBAL_METADATA = "global_metadata"


class ReferenceScope(StrEnum):
    """Physical scope at which a reference quantity is defined."""

    CASE = "case"
    OPERATING_CONDITION = "operating_condition"
    SNAPSHOT = "snapshot"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Semantic description of one canonical field."""

    name: str
    support: FieldSupport
    role: FieldRole
    components: tuple[str, ...] = ("value",)
    source_path: str | None = None
    units: str | None = None
    provenance: str | None = None
    stored: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("FieldSpec.name must be non-empty.")
        if not self.components:
            raise ValueError(f"Field '{self.name}' must define at least one component.")
        if any(not component.strip() for component in self.components):
            raise ValueError(f"Field '{self.name}' contains an empty component name.")
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"Field '{self.name}' contains duplicate component names.")


@dataclass(frozen=True, slots=True)
class FieldCatalog:
    """Immutable collection of uniquely named field specifications."""

    fields: tuple[FieldSpec, ...]
    _by_name: Mapping[str, FieldSpec] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_name: dict[str, FieldSpec] = {}
        for spec in self.fields:
            if spec.name in by_name:
                raise ValueError(f"Duplicate field name '{spec.name}' in catalog.")
            by_name[spec.name] = spec
        object.__setattr__(self, "_by_name", by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> FieldSpec:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown field '{name}'.") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.fields)

    def require(self, names: tuple[str, ...] | list[str]) -> tuple[FieldSpec, ...]:
        """Return requested fields in the exact requested order."""

        return tuple(self[name] for name in names)


@dataclass(frozen=True, slots=True)
class ReferenceScale:
    """One physical reference quantity and its explicit semantics."""

    name: str
    value: float
    definition: str
    provenance: str | None = None
    units: str | None = None
    scope: ReferenceScope = ReferenceScope.CASE
    inference_available: bool | None = None
    derivation: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ReferenceScale.name must be non-empty.")
        if not self.definition.strip():
            raise ValueError(f"Reference scale '{self.name}' needs an explicit definition.")
        if not isfinite(self.value):
            raise ValueError(f"Reference scale '{self.name}' must be finite.")
        if self.provenance is not None and not self.provenance.strip():
            raise ValueError(f"Reference scale '{self.name}' has empty provenance.")
        if self.units is not None and not self.units.strip():
            raise ValueError(f"Reference scale '{self.name}' has empty units.")
        if self.derivation is not None and not self.derivation.strip():
            raise ValueError(f"Reference scale '{self.name}' has empty derivation.")
        try:
            scope = ReferenceScope(self.scope)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Reference scale '{self.name}' has invalid scope '{self.scope}'."
            ) from exc
        object.__setattr__(self, "scope", scope)
        if self.inference_available is not None and not isinstance(
            self.inference_available, bool
        ):
            raise TypeError(
                f"Reference scale '{self.name}' inference_available must be a bool or None."
            )


@dataclass(frozen=True, slots=True)
class ReferenceScales:
    """Collection of uniquely named physical reference quantities."""

    scales: tuple[ReferenceScale, ...] = ()
    scheme: str | None = None
    _by_name: Mapping[str, ReferenceScale] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.scheme is not None and not self.scheme.strip():
            raise ValueError("ReferenceScales.scheme must be non-empty when provided.")
        by_name: dict[str, ReferenceScale] = {}
        for scale in self.scales:
            if scale.name in by_name:
                raise ValueError(f"Duplicate reference scale '{scale.name}'.")
            by_name[scale.name] = scale
        object.__setattr__(self, "_by_name", by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> ReferenceScale:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown reference scale '{name}'.") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(scale.name for scale in self.scales)

    def require(self, names: tuple[str, ...] | list[str]) -> tuple[ReferenceScale, ...]:
        """Return requested references in the exact requested order."""

        return tuple(self[name] for name in names)


@dataclass(slots=True)
class Mesh:
    """Canonical node-based CFD mesh representation for one sample."""

    coords: torch.Tensor
    edge_index: torch.Tensor
    mesh_id: str | None = None
    node_weights: torch.Tensor | None = None
    cell_connectivity: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.coords.ndim != 2:
            raise ValueError("Mesh.coords must have shape [N, D].")
        if not self.coords.is_floating_point():
            raise TypeError("Mesh.coords must use a floating-point dtype.")
        if not torch.isfinite(self.coords).all():
            raise ValueError("Mesh.coords contains NaN or Inf values.")

        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("Mesh.edge_index must have shape [2, E].")
        if self.edge_index.dtype != torch.long:
            raise TypeError("Mesh.edge_index must use torch.long indices.")

        if self.edge_index.numel() > 0:
            if int(self.edge_index.min()) < 0:
                raise ValueError("Mesh.edge_index contains a negative node index.")
            if int(self.edge_index.max()) >= self.num_nodes:
                raise ValueError("Mesh.edge_index references a node outside coords.")

        if self.node_weights is not None:
            if self.node_weights.ndim != 1 or self.node_weights.shape[0] != self.num_nodes:
                raise ValueError("Mesh.node_weights must have shape [N].")
            if not self.node_weights.is_floating_point():
                raise TypeError("Mesh.node_weights must use a floating-point dtype.")
            if not torch.isfinite(self.node_weights).all():
                raise ValueError("Mesh.node_weights contains NaN or Inf values.")

        if self.cell_connectivity is not None:
            if self.cell_connectivity.ndim != 2:
                raise ValueError("Mesh.cell_connectivity must have shape [C, K].")
            if self.cell_connectivity.dtype != torch.long:
                raise TypeError("Mesh.cell_connectivity must use torch.long indices.")
            if self.cell_connectivity.numel() > 0:
                if int(self.cell_connectivity.min()) < 0:
                    raise ValueError("Mesh.cell_connectivity contains a negative node index.")
                if int(self.cell_connectivity.max()) >= self.num_nodes:
                    raise ValueError("Mesh.cell_connectivity references a node outside coords.")

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
    """One physical CFD sample with named fields and one native mesh."""

    sample_id: str
    mesh: Mesh
    fields: Mapping[str, torch.Tensor]
    reference_scales: ReferenceScales = field(default_factory=ReferenceScales)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("Sample.sample_id must be non-empty.")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("Sample field names must be unique.")
        for name, value in self.fields.items():
            if not name.strip():
                raise ValueError("Sample contains an empty field name.")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Sample field '{name}' must be a torch.Tensor.")

    def validate_against(self, catalog: FieldCatalog) -> None:
        """Validate loaded fields against their declared spatial support.

        M2 can validate node-supported fields because the canonical mesh node
        count is known. Cell/face support sizes are intentionally not guessed.
        """

        for name, value in self.fields.items():
            spec = catalog[name]
            if spec.support is FieldSupport.NODE:
                if value.ndim == 0 or value.shape[0] != self.mesh.num_nodes:
                    raise ValueError(
                        f"Node field '{name}' must have leading dimension N="
                        f"{self.mesh.num_nodes}, got shape {tuple(value.shape)}."
                    )
            expected_components = len(spec.components)
            if expected_components > 1:
                if value.ndim < 2 or value.shape[-1] != expected_components:
                    raise ValueError(
                        f"Field '{name}' declares {expected_components} components "
                        f"but has shape {tuple(value.shape)}."
                    )
