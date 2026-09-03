"""Deterministic node-regression task contracts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from graph_attention.data import (
    ConvectiveNondimensionalizer,
    FieldCatalog,
    FieldSpec,
    FieldSupport,
    PackedBatch,
    Sample,
    pack_samples,
)


@dataclass(frozen=True, slots=True)
class NodeRegressionBatch:
    """Task-facing node-regression tensors with preserved packed provenance."""

    source: PackedBatch
    coords: torch.Tensor
    inputs: torch.Tensor
    targets: torch.Tensor
    conditioning: torch.Tensor
    input_channels: tuple[str, ...]
    target_channels: tuple[str, ...]
    conditioning_names: tuple[str, ...]

    @property
    def edge_index(self) -> torch.Tensor:
        return self.source.edge_index

    @property
    def batch_index(self) -> torch.Tensor:
        return self.source.batch_index

    @property
    def ptr(self) -> torch.Tensor:
        return self.source.ptr

    @property
    def node_weights(self) -> torch.Tensor | None:
        return self.source.node_weights

    @property
    def num_graphs(self) -> int:
        return self.source.num_graphs


class NodeRegressionTask:
    """Explicit named-field deterministic node-regression task.

    M5 selects complete declared field groups. Partial selection from within one
    multi-component field is intentionally deferred until a real task requires
    it; separate stored scalar components remain independently selectable.
    """

    def __init__(
        self,
        input_fields: Iterable[str],
        target_fields: Iterable[str],
        conditioning_parameters: Iterable[str] = (),
        physical_nondimensionalization: bool = False,
    ) -> None:
        self.input_fields = _validated_names(
            input_fields,
            "input_fields",
            require_nonempty=True,
        )
        self.target_fields = _validated_names(
            target_fields,
            "target_fields",
            require_nonempty=True,
        )
        self.conditioning_parameters = _validated_names(
            conditioning_parameters,
            "conditioning_parameters",
            require_nonempty=False,
        )
        if not isinstance(physical_nondimensionalization, bool):
            raise TypeError("physical_nondimensionalization must be a bool")
        self.physical_nondimensionalization = physical_nondimensionalization

    @property
    def required_node_fields(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.input_fields + self.target_fields))

    def pack_and_prepare(
        self,
        samples: Iterable[Sample],
        catalog: FieldCatalog,
    ) -> NodeRegressionBatch:
        """Pack selected samples and prepare the explicit task representation."""

        packed = pack_samples(samples, node_field_names=self.required_node_fields)
        return self.prepare(packed, catalog)

    def prepare(self, packed: PackedBatch, catalog: FieldCatalog) -> NodeRegressionBatch:
        """Prepare inputs, targets, coordinates, and global conditioning."""

        specs = self._require_node_specs(catalog)
        fields, coords = self._preprocess_physical(packed)

        inputs, input_channels = _concatenate_fields(
            self.input_fields,
            fields,
            specs,
            role="input",
        )
        targets, target_channels = _concatenate_fields(
            self.target_fields,
            fields,
            specs,
            role="target",
        )
        if inputs.device != targets.device:
            raise ValueError("task inputs and targets must be on the same device")
        if inputs.dtype != targets.dtype:
            raise TypeError(
                "task inputs and targets must share one dtype, "
                f"got {inputs.dtype} and {targets.dtype}"
            )

        conditioning = _build_conditioning(
            packed,
            self.conditioning_parameters,
            dtype=inputs.dtype,
            device=inputs.device,
        )

        return NodeRegressionBatch(
            source=packed,
            coords=coords,
            inputs=inputs,
            targets=targets,
            conditioning=conditioning,
            input_channels=input_channels,
            target_channels=target_channels,
            conditioning_names=self.conditioning_parameters,
        )

    def _require_node_specs(self, catalog: FieldCatalog) -> dict[str, FieldSpec]:
        specs: dict[str, FieldSpec] = {}
        for name in self.required_node_fields:
            spec = catalog[name]
            if spec.support is not FieldSupport.NODE:
                raise ValueError(
                    f"task field '{name}' has support '{spec.support}', expected node support"
                )
            specs[name] = spec
        return specs

    def _preprocess_physical(
        self,
        packed: PackedBatch,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        fields = {name: packed.fields[name] for name in self.required_node_fields}
        if not self.physical_nondimensionalization:
            return fields, packed.coords

        field_parts: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.required_node_fields
        }
        coord_parts: list[torch.Tensor] = []

        for graph_index in range(packed.num_graphs):
            start = int(packed.ptr[graph_index])
            stop = int(packed.ptr[graph_index + 1])
            transform = ConvectiveNondimensionalizer(packed.reference_scales[graph_index])
            graph_fields = {
                name: packed.fields[name][start:stop] for name in self.required_node_fields
            }
            transformed = transform.nondimensionalize(graph_fields)
            for name in self.required_node_fields:
                field_parts[name].append(transformed[name])
            coord_parts.append(
                transform.nondimensionalize_coordinates(packed.coords[start:stop])
            )

        return (
            {name: torch.cat(parts, dim=0) for name, parts in field_parts.items()},
            torch.cat(coord_parts, dim=0),
        )


def _validated_names(
    values: Iterable[str],
    label: str,
    *,
    require_nonempty: bool,
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{label} must be an iterable of names, not one string")
    names = tuple(values)
    if require_nonempty and not names:
        raise ValueError(f"{label} must contain at least one field name")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{label} must contain only non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not contain duplicates")
    return names


def _concatenate_fields(
    names: Sequence[str],
    fields: dict[str, torch.Tensor],
    specs: dict[str, FieldSpec],
    *,
    role: str,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    matrices: list[torch.Tensor] = []
    channels: list[str] = []
    dtype: torch.dtype | None = None
    device: torch.device | None = None

    for name in names:
        try:
            value = fields[name]
        except KeyError as exc:
            raise KeyError(f"packed batch does not contain task field '{name}'") from exc
        spec = specs[name]
        matrix = _as_channel_matrix(value, spec)
        if not matrix.is_floating_point():
            raise TypeError(f"task {role} field '{name}' must use a floating-point dtype")
        if dtype is None:
            dtype = matrix.dtype
            device = matrix.device
        elif matrix.dtype != dtype:
            raise TypeError(
                f"task {role} fields must share one dtype; field '{name}' has {matrix.dtype}, "
                f"expected {dtype}"
            )
        elif matrix.device != device:
            raise ValueError(
                f"task {role} fields must share one device; field '{name}' is on "
                f"{matrix.device}, expected {device}"
            )
        matrices.append(matrix)
        channels.extend(f"{name}.{component}" for component in spec.components)

    return torch.cat(matrices, dim=1), tuple(channels)


def _as_channel_matrix(value: torch.Tensor, spec: FieldSpec) -> torch.Tensor:
    components = len(spec.components)
    if value.ndim == 1:
        if components != 1:
            raise ValueError(
                f"field '{spec.name}' declares {components} components but has shape "
                f"{tuple(value.shape)}"
            )
        return value.unsqueeze(1)
    if value.ndim == 2 and value.shape[1] == components:
        return value
    raise ValueError(
        f"task field '{spec.name}' must have shape [N] for one component or [N, C] "
        f"matching its {components} declared components, got {tuple(value.shape)}"
    )


def _build_conditioning(
    packed: PackedBatch,
    names: tuple[str, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not names:
        return torch.empty((packed.num_graphs, 0), dtype=dtype, device=device)

    definitions: dict[str, str] = {}
    rows: list[list[float]] = []
    for graph_index, parameters in enumerate(packed.regime_parameters):
        row: list[float] = []
        for name in names:
            try:
                parameter = parameters[name]
            except KeyError as exc:
                raise KeyError(
                    f"graph {graph_index} does not define requested conditioning "
                    f"parameter '{name}'"
                ) from exc
            if parameter.inference_available is not True:
                raise ValueError(
                    f"conditioning parameter '{name}' for graph {graph_index} is not declared "
                    "available at inference"
                )
            previous_definition = definitions.setdefault(name, parameter.definition)
            if parameter.definition != previous_definition:
                raise ValueError(
                    f"conditioning parameter '{name}' has inconsistent definitions across graphs"
                )
            row.append(parameter.value)
        rows.append(row)

    return torch.tensor(rows, dtype=dtype, device=device)
