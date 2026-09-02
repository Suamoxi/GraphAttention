from pathlib import Path

import pytest

from graph_attention.data import ReferenceScope, load_case_definition, load_case_definitions


def _write_case(path: Path, *, case_id: str = "case-a") -> None:
    path.write_text(
        f"""case_id: {case_id}
reference_scheme: hit_forcing_reference
references:
  rho_ref:
    value: 1.2
    units: kg/m^3
    definition: prescribed_reference_density
    provenance: simulation_setup
    inference_available: true
  U_ref:
    value: 69.44
    units: m/s
    definition: prescribed_reference_velocity
    provenance: simulation_setup
    inference_available: true
    scope: operating_condition
    derivation: Ma_ref * sqrt(gamma * R * T_ref)
regime:
  Re:
    value: 50000.0
    definition: rho_ref_U_ref_L_ref_over_mu_ref
    provenance: simulation_setup
    inference_available: true
    derivation: rho_ref * U_ref * L_ref / mu_ref
  Ma:
    value: 0.2
    definition: U_ref_over_a_ref
    provenance: simulation_setup
    inference_available: true
"""
    )


def test_case_definition_loads_explicit_values_and_derivation_as_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case.yaml"
    _write_case(path)

    case = load_case_definition(path)

    assert case.case_id == "case-a"
    assert case.source_path == path.resolve()
    assert case.reference_scales.scheme == "hit_forcing_reference"
    assert case.reference_scales["rho_ref"].value == 1.2
    assert case.reference_scales["rho_ref"].scope is ReferenceScope.CASE
    assert case.reference_scales["U_ref"].value == 69.44
    assert case.reference_scales["U_ref"].scope is ReferenceScope.OPERATING_CONDITION
    assert case.reference_scales["U_ref"].derivation == "Ma_ref * sqrt(gamma * R * T_ref)"
    assert case.regime_parameters.names == ("Re", "Ma")
    assert case.regime_parameters["Re"].value == 50000.0
    assert case.regime_parameters["Re"].derivation == "rho_ref * U_ref * L_ref / mu_ref"
    assert case.regime_parameters["Ma"].value == 0.2


def test_case_definition_requires_literal_numeric_reference_values(tmp_path: Path) -> None:
    path = tmp_path / "case.yaml"
    path.write_text(
        """case_id: case-a
reference_scheme: test
references:
  U_ref:
    value: ${oc.env:U_REF}
    units: m/s
    definition: prescribed_velocity
    provenance: simulation_setup
    inference_available: true
"""
    )

    with pytest.raises(TypeError, match="explicit numeric literal"):
        load_case_definition(path)


def test_case_definition_requires_literal_numeric_regime_values(tmp_path: Path) -> None:
    path = tmp_path / "case.yaml"
    path.write_text(
        """case_id: case-a
reference_scheme: test
references:
  U_ref:
    value: 10.0
    units: m/s
    definition: prescribed_velocity
    provenance: simulation_setup
    inference_available: true
regime:
  Ma:
    value: ${oc.env:MACH}
    definition: U_ref_over_a_ref
    provenance: simulation_setup
    inference_available: true
"""
    )

    with pytest.raises(TypeError, match="explicit numeric literal"):
        load_case_definition(path)


def test_case_definition_requires_explicit_inference_availability(tmp_path: Path) -> None:
    path = tmp_path / "case.yaml"
    path.write_text(
        """case_id: case-a
reference_scheme: test
references:
  rho_ref:
    value: 1.0
    units: kg/m^3
    definition: prescribed_density
    provenance: simulation_setup
"""
    )

    with pytest.raises(ValueError, match="inference_available"):
        load_case_definition(path)


def test_case_definition_rejects_unknown_regime_keys(tmp_path: Path) -> None:
    path = tmp_path / "case.yaml"
    path.write_text(
        """case_id: case-a
reference_scheme: test
references:
  U_ref:
    value: 10.0
    units: m/s
    definition: prescribed_velocity
    provenance: simulation_setup
    inference_available: true
regime:
  Ma:
    value: 0.2
    definition: U_ref_over_a_ref
    provenance: simulation_setup
    inference_available: true
    units: dimensionless
"""
    )

    with pytest.raises(ValueError, match="unsupported keys"):
        load_case_definition(path)


def test_case_definition_mapping_key_must_match_declared_case_id(tmp_path: Path) -> None:
    path = tmp_path / "case.yaml"
    _write_case(path, case_id="declared-case")

    with pytest.raises(ValueError, match="does not match declared case_id"):
        load_case_definitions({"configured-case": path})
