import pytest
import torch

from graph_attention.data import (
    FieldCatalog,
    FieldRole,
    FieldSpec,
    FieldSupport,
    Mesh,
    ReferenceScale,
    ReferenceScales,
    ReferenceScope,
    RegimeParameter,
    RegimeParameters,
    Sample,
)


def test_field_catalog_preserves_semantics_and_requested_order() -> None:
    rho = FieldSpec(
        name="rho",
        support=FieldSupport.NODE,
        role=FieldRole.PRIMARY_STATE,
        source_path="/GaseousPhase/rho",
        units="kg/m^3",
    )
    momentum = FieldSpec(
        name="momentum",
        support=FieldSupport.NODE,
        role=FieldRole.PRIMARY_STATE,
        components=("x", "y", "z"),
        provenance="stored as rhou/rhov/rhow components",
    )
    catalog = FieldCatalog((rho, momentum))

    assert catalog.names == ("rho", "momentum")
    assert catalog.require(["momentum", "rho"]) == (momentum, rho)


def test_field_catalog_rejects_duplicate_names() -> None:
    spec = FieldSpec("rho", FieldSupport.NODE, FieldRole.PRIMARY_STATE)
    with pytest.raises(ValueError, match="Duplicate field name"):
        FieldCatalog((spec, spec))


def test_reference_scales_preserve_definition_separately_from_value() -> None:
    scales = ReferenceScales(
        (
            ReferenceScale(
                name="U_ref",
                value=12.5,
                definition="bulk_velocity",
                provenance="case_metadata",
                units="m/s",
                scope=ReferenceScope.CASE,
            ),
        ),
        scheme="bulk_flow_reference",
    )

    assert scales["U_ref"].value == 12.5
    assert scales["U_ref"].definition == "bulk_velocity"
    assert scales["U_ref"].units == "m/s"
    assert scales["U_ref"].scope is ReferenceScope.CASE
    assert scales.scheme == "bulk_flow_reference"
    assert scales.names == ("U_ref",)


def test_reference_scale_accepts_string_scope_and_rejects_invalid_scope() -> None:
    scale = ReferenceScale("U_ref", 1.0, "bulk_velocity", scope="operating_condition")
    assert scale.scope is ReferenceScope.OPERATING_CONDITION

    with pytest.raises(ValueError, match="invalid scope"):
        ReferenceScale("U_ref", 1.0, "bulk_velocity", scope="unknown")


def test_regime_parameters_preserve_explicit_dimensionless_semantics() -> None:
    reynolds = RegimeParameter(
        name="Re",
        value=50000.0,
        definition="rho_ref_U_ref_L_ref_over_mu_ref",
        provenance="simulation_setup",
        inference_available=True,
        derivation="rho_ref * U_ref * L_ref / mu_ref",
    )
    mach = RegimeParameter(
        name="Ma",
        value=0.2,
        definition="U_ref_over_a_ref",
        provenance="simulation_setup",
        inference_available=True,
    )
    parameters = RegimeParameters((reynolds, mach))

    assert parameters.names == ("Re", "Ma")
    assert parameters["Re"].value == 50000.0
    assert parameters["Re"].derivation == "rho_ref * U_ref * L_ref / mu_ref"
    assert parameters.require(["Ma", "Re"]) == (mach, reynolds)


def test_regime_parameters_reject_duplicate_names() -> None:
    parameter = RegimeParameter("Ma", 0.1, "U_ref_over_a_ref")

    with pytest.raises(ValueError, match="Duplicate regime parameter"):
        RegimeParameters((parameter, parameter))


def test_mesh_validates_canonical_node_graph_shapes() -> None:
    mesh = Mesh(
        coords=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        node_weights=torch.tensor([0.2, 0.3, 0.5]),
    )

    assert mesh.num_nodes == 3
    assert mesh.num_edges == 3
    assert mesh.spatial_dim == 2


def test_mesh_rejects_out_of_range_connectivity() -> None:
    with pytest.raises(ValueError, match="outside coords"):
        Mesh(
            coords=torch.zeros((2, 3)),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        )


def test_mesh_validates_optional_native_cell_connectivity() -> None:
    connectivity = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    mesh = Mesh(
        coords=torch.zeros((4, 3)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        cell_connectivity=connectivity,
    )

    assert mesh.cell_connectivity is connectivity

    with pytest.raises(ValueError, match="cell_connectivity references a node outside coords"):
        Mesh(
            coords=torch.zeros((4, 3)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            cell_connectivity=torch.tensor([[0, 1, 2, 4]], dtype=torch.long),
        )


def test_sample_validates_node_support_and_components_against_catalog() -> None:
    catalog = FieldCatalog(
        (
            FieldSpec("rho", FieldSupport.NODE, FieldRole.PRIMARY_STATE),
            FieldSpec(
                "momentum",
                FieldSupport.NODE,
                FieldRole.PRIMARY_STATE,
                components=("x", "y", "z"),
            ),
        )
    )
    mesh = Mesh(
        coords=torch.zeros((4, 3)),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
    )
    regime_parameters = RegimeParameters(
        (RegimeParameter("Ma", 0.2, "U_ref_over_a_ref", provenance="case"),)
    )
    sample = Sample(
        sample_id="case-a/snapshot-0001",
        mesh=mesh,
        fields={"rho": torch.ones(4), "momentum": torch.ones((4, 3))},
        case_id="case-a",
        regime_parameters=regime_parameters,
    )

    assert sample.case_id == "case-a"
    assert sample.regime_parameters is regime_parameters
    sample.validate_against(catalog)

    bad = Sample(
        sample_id="case-a/snapshot-0002",
        mesh=mesh,
        fields={"momentum": torch.ones((4, 2))},
    )
    with pytest.raises(ValueError, match="declares 3 components"):
        bad.validate_against(catalog)


def test_sample_rejects_invalid_case_id() -> None:
    mesh = Mesh(torch.zeros((1, 3)), torch.empty((2, 0), dtype=torch.long))

    with pytest.raises(ValueError, match="case_id"):
        Sample("sample", mesh, {"rho": torch.ones(1)}, case_id="")


def test_sample_rejects_unknown_field_when_validated() -> None:
    catalog = FieldCatalog(())
    mesh = Mesh(torch.zeros((1, 3)), torch.empty((2, 0), dtype=torch.long))
    sample = Sample("sample", mesh, {"mystery": torch.ones(1)})

    with pytest.raises(KeyError, match="Unknown field"):
        sample.validate_against(catalog)
