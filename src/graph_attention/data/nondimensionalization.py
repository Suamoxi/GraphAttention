"""Explicit M3.3 physical nondimensionalization for supported CFD fields."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import torch

from .contracts import ReferenceScale, ReferenceScales, ReferenceScope

_FIELD_REFERENCE_POWERS: dict[str, tuple[tuple[str, int], ...]] = {
    "rho": (("rho_ref", 1),),
    "rhou": (("rho_ref", 1), ("U_ref", 1)),
    "rhov": (("rho_ref", 1), ("U_ref", 1)),
    "rhow": (("rho_ref", 1), ("U_ref", 1)),
    "rhoE": (("rho_ref", 1), ("U_ref", 2)),
    "pressure": (("rho_ref", 1), ("U_ref", 2)),
    "temperature": (("T_ref", 1),),
}


@dataclass(frozen=True, slots=True)
class ConvectiveNondimensionalizer:
    """Apply the frozen M3.3 convective reference-state convention.

    The class transforms only explicitly supported canonical field names. It
    never infers a transformation from tensor shape, units, or numerical range.
    """

    reference_scales: ReferenceScales

    def __post_init__(self) -> None:
        if self.reference_scales.scheme is None:
            raise ValueError("physical nondimensionalization requires an explicit reference scheme")

    def field_scale(self, field_name: str) -> float:
        """Return the dimensional multiplicative scale for one canonical field."""

        try:
            powers = _FIELD_REFERENCE_POWERS[field_name]
        except KeyError as exc:
            raise KeyError(
                f"No M3.3 nondimensionalization is defined for field '{field_name}'."
            ) from exc

        scale = 1.0
        for reference_name, power in powers:
            reference = self._validated_reference(field_name, reference_name)
            scale *= reference.value**power
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"Field '{field_name}' has an invalid derived dimensional scale {scale}."
            )
        return scale

    def nondimensionalize(
        self,
        fields: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return dimensionless copies of the supplied named physical fields."""

        return self._transform(fields, inverse=False)

    def dimensionalize(
        self,
        fields: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Invert M3.3 nondimensionalization for supplied named fields."""

        return self._transform(fields, inverse=True)

    def _transform(
        self,
        fields: Mapping[str, torch.Tensor],
        *,
        inverse: bool,
    ) -> dict[str, torch.Tensor]:
        transformed: dict[str, torch.Tensor] = {}
        for name, tensor in fields.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Field '{name}' must be a torch.Tensor.")
            if not tensor.is_floating_point():
                raise TypeError(
                    f"Physical field '{name}' must be floating-point for nondimensionalization."
                )
            scale = self.field_scale(name)
            transformed[name] = tensor * scale if inverse else tensor / scale
        return transformed

    def _validated_reference(
        self,
        field_name: str,
        reference_name: str,
    ) -> ReferenceScale:
        try:
            reference = self.reference_scales[reference_name]
        except KeyError as exc:
            raise ValueError(
                f"Field '{field_name}' requires missing reference '{reference_name}'."
            ) from exc

        if reference.value <= 0.0:
            raise ValueError(
                f"Reference '{reference_name}' must be strictly positive for field '{field_name}'."
            )
        if reference.units is None:
            raise ValueError(
                f"Reference '{reference_name}' requires explicit units before preprocessing."
            )
        if reference.provenance is None:
            raise ValueError(
                f"Reference '{reference_name}' requires explicit provenance before preprocessing."
            )
        if reference.inference_available is not True:
            raise ValueError(
                f"Reference '{reference_name}' must be explicitly marked available at "
                f"inference for field '{field_name}'."
            )
        if reference.scope is ReferenceScope.SNAPSHOT:
            raise ValueError(
                f"Snapshot-scoped reference '{reference_name}' is not supported by the "
                "baseline M3.3 convention."
            )
        return reference
