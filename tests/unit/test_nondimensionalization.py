import pytest
import torch

from graph_attention.data import (
    ConvectiveNondimensionalizer,
    ReferenceScale,
    ReferenceScales,
    ReferenceScope,
)


def _scale(
    name: str,
    value: float,
    definition: str,
    units: str,
    *,
    scope: ReferenceScope = ReferenceScope.CASE,
    inference_available: bool = True,
    provenance: str | None = "test_case_definition",
) -> ReferenceScale:
    return ReferenceScale(
        name=name,
        value=value,
        definition=definition,
        provenance=provenance,
        units=units,
        scope=scope,
        inference_available=inference_available,
    )


def _references() -> ReferenceScales:
    return ReferenceScales(
        (
            _scale("rho_ref", 2.0, "reference_density", "kg/m^3"),
            _scale("U_ref", 5.0, "characteristic_velocity", "m/s"),
            _scale("L_ref", 4.0, "characteristic_length", "m"),
            _scale("T_ref", 300.0, "reference_temperature", "K"),
        ),
        scheme="test_convective_reference",
    )


def test_convective_nondimensionalization_matches_frozen_field_scales() -> None:
    transform = ConvectiveNondimensionalizer(_references())
    fields = {
        "rho": torch.tensor([2.0], dtype=torch.float64),
        "rhou": torch.tensor([10.0], dtype=torch.float64),
        "rhov": torch.tensor([-20.0], dtype=torch.float64),
        "rhow": torch.tensor([5.0], dtype=torch.float64),
        "rhoE": torch.tensor([100.0], dtype=torch.float64),
        "pressure": torch.tensor([50.0], dtype=torch.float64),
        "temperature": torch.tensor([600.0], dtype=torch.float64),
        "vis_lam": torch.tensor([40.0], dtype=torch.float64),
        "vis_turb": torch.tensor([20.0], dtype=torch.float64),
    }

    result = transform.nondimensionalize(fields)

    expected = {
        "rho": 1.0,
        "rhou": 1.0,
        "rhov": -2.0,
        "rhow": 0.5,
        "rhoE": 2.0,
        "pressure": 1.0,
        "temperature": 2.0,
        "vis_lam": 1.0,
        "vis_turb": 0.5,
    }
    for name, value in expected.items():
        assert result[name].dtype == torch.float64
        assert torch.allclose(result[name], torch.tensor([value], dtype=torch.float64))


def test_convective_nondimensionalization_round_trip() -> None:
    transform = ConvectiveNondimensionalizer(_references())
    fields = {
        "rho": torch.tensor([0.9, 1.2], dtype=torch.float64),
        "rhou": torch.tensor([-2.0, 7.5], dtype=torch.float64),
        "rhoE": torch.tensor([1000.0, 1200.0], dtype=torch.float64),
    }

    restored = transform.dimensionalize(transform.nondimensionalize(fields))

    for name, tensor in fields.items():
        assert torch.allclose(restored[name], tensor, rtol=1e-12, atol=1e-12)


def test_only_references_required_by_requested_fields_are_needed() -> None:
    references = ReferenceScales(
        (_scale("rho_ref", 2.0, "reference_density", "kg/m^3"),),
        scheme="density_only_test",
    )
    transform = ConvectiveNondimensionalizer(references)

    result = transform.nondimensionalize({"rho": torch.tensor([4.0])})

    assert torch.equal(result["rho"], torch.tensor([2.0]))


def test_missing_reference_fails_without_fallback() -> None:
    references = ReferenceScales(
        (_scale("rho_ref", 2.0, "reference_density", "kg/m^3"),),
        scheme="incomplete_test",
    )
    transform = ConvectiveNondimensionalizer(references)

    with pytest.raises(ValueError, match="missing reference 'U_ref'"):
        transform.nondimensionalize({"rhou": torch.tensor([1.0])})


def test_reference_scheme_is_required_for_physical_preprocessing() -> None:
    references = ReferenceScales(
        (_scale("rho_ref", 2.0, "reference_density", "kg/m^3"),)
    )

    with pytest.raises(ValueError, match="explicit reference scheme"):
        ConvectiveNondimensionalizer(references)


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        (
            _scale(
                "rho_ref",
                2.0,
                "reference_density",
                "kg/m^3",
                inference_available=False,
            ),
            "not available at inference",
        ),
        (
            _scale(
                "rho_ref",
                2.0,
                "instantaneous_density_scale",
                "kg/m^3",
                scope=ReferenceScope.SNAPSHOT,
            ),
            "Snapshot-scoped reference",
        ),
        (
            _scale(
                "rho_ref",
                2.0,
                "reference_density",
                "kg/m^3",
                provenance=None,
            ),
            "requires explicit provenance",
        ),
    ],
)
def test_invalid_reference_semantics_fail_before_transform(
    reference: ReferenceScale,
    message: str,
) -> None:
    transform = ConvectiveNondimensionalizer(
        ReferenceScales((reference,), scheme="invalid_reference_test")
    )

    with pytest.raises(ValueError, match=message):
        transform.nondimensionalize({"rho": torch.tensor([1.0])})


def test_nonpositive_reference_is_rejected_when_used() -> None:
    reference = _scale("rho_ref", 0.0, "reference_density", "kg/m^3")
    transform = ConvectiveNondimensionalizer(
        ReferenceScales((reference,), scheme="zero_reference_test")
    )

    with pytest.raises(ValueError, match="strictly positive"):
        transform.nondimensionalize({"rho": torch.tensor([1.0])})


def test_unsupported_or_nonfloating_fields_do_not_silently_pass_through() -> None:
    transform = ConvectiveNondimensionalizer(_references())

    with pytest.raises(KeyError, match="No M3.3 nondimensionalization"):
        transform.nondimensionalize({"mpi_rank": torch.tensor([0.0])})

    with pytest.raises(TypeError, match="must be floating-point"):
        transform.nondimensionalize({"rho": torch.tensor([1], dtype=torch.int64)})
