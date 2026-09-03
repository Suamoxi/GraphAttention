from dataclasses import replace

import pytest
import torch

from graph_attention.data import (
    AVBP_FIELD_CATALOG,
    FieldCatalog,
    FieldRole,
    FieldSpec,
    FieldSupport,
    Mesh,
    ReferenceScale,
    ReferenceScales,
    RegimeParameter,
    RegimeParameters,
    Sample,
    SyntheticMeshDataset,
)
from graph_attention.tasks import NodeRegressionTask


def _references(rho_ref: float, u_ref: float, l_ref: float) -> ReferenceScales:
    return ReferenceScales(
        (
            ReferenceScale(
                name="rho_ref",
                value=rho_ref,
                definition="test_density",
                units="kg/m^3",
                provenance="unit test",
                inference_available=True,
            ),
            ReferenceScale(
                name="U_ref",
                value=u_ref,
                definition="test_velocity",
                units="m/s",
                provenance="unit test",
                inference_available=True,
            ),
            ReferenceScale(
                name="L_ref",
                value=l_ref,
                definition="test_length",
                units="m",
                provenance="unit test",
                inference_available=True,
            ),
        ),
        scheme="test_reference",
    )


def _physical_sample(
    sample_id: str,
    *,
    rho_ref: float,
    u_ref: float,
    l_ref: float,
) -> Sample:
    coords = torch.tensor([[0.0, 0.0], [l_ref, 0.0]], dtype=torch.float64)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    rho = torch.tensor([rho_ref, 2.0 * rho_ref], dtype=torch.float64)
    rhou = torch.tensor([rho_ref * u_ref, 1.5 * rho_ref * u_ref], dtype=torch.float64)
    return Sample(
        sample_id=sample_id,
        case_id=f"case-{sample_id}",
        mesh=Mesh(coords=coords, edge_index=edge_index),
        fields={"rho": rho, "rhou": rhou},
        reference_scales=_references(rho_ref, u_ref, l_ref),
    )


def _regime_parameters(
    *,
    re_value: float,
    ma_value: float,
    re_definition: str = "test_reynolds_number",
    available: bool = True,
) -> RegimeParameters:
    return RegimeParameters(
        (
            RegimeParameter(
                name="Re",
                value=re_value,
                definition=re_definition,
                provenance="unit test",
                inference_available=available,
            ),
            RegimeParameter(
                name="Ma",
                value=ma_value,
                definition="test_mach_number",
                provenance="unit test",
                inference_available=True,
            ),
        )
    )


def test_regression_task_builds_explicit_channel_order() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=3, seed=13)
    task = NodeRegressionTask(
        input_fields=("rho", "momentum"),
        target_fields=("momentum", "rho"),
    )
    samples = [dataset[0], dataset[1]]

    batch = task.pack_and_prepare(samples, dataset.field_catalog)

    assert batch.input_channels == (
        "rho.value",
        "momentum.x",
        "momentum.y",
        "momentum.z",
    )
    assert batch.target_channels == (
        "momentum.x",
        "momentum.y",
        "momentum.z",
        "rho.value",
    )
    expected_inputs = torch.cat(
        (
            batch.source.fields["rho"].unsqueeze(1),
            batch.source.fields["momentum"],
        ),
        dim=1,
    )
    expected_targets = torch.cat(
        (
            batch.source.fields["momentum"],
            batch.source.fields["rho"].unsqueeze(1),
        ),
        dim=1,
    )
    torch.testing.assert_close(batch.inputs, expected_inputs)
    torch.testing.assert_close(batch.targets, expected_targets)
    assert batch.conditioning.shape == (2, 0)


def test_regression_task_applies_case_specific_physical_nondimensionalization() -> None:
    first = _physical_sample("a", rho_ref=2.0, u_ref=3.0, l_ref=4.0)
    second = _physical_sample("b", rho_ref=5.0, u_ref=2.0, l_ref=10.0)
    task = NodeRegressionTask(
        input_fields=("rhou",),
        target_fields=("rho",),
        physical_nondimensionalization=True,
    )

    batch = task.pack_and_prepare([first, second], AVBP_FIELD_CATALOG)

    torch.testing.assert_close(
        batch.inputs[:, 0],
        torch.tensor([1.0, 1.5, 1.0, 1.5], dtype=torch.float64),
    )
    torch.testing.assert_close(
        batch.targets[:, 0],
        torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        batch.coords,
        torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        batch.source.fields["rho"],
        torch.tensor([2.0, 4.0, 5.0, 10.0], dtype=torch.float64),
    )


def test_regression_task_builds_ordered_inference_available_conditioning() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, seed=3)
    first = replace(
        dataset[0],
        regime_parameters=_regime_parameters(re_value=10.0, ma_value=0.1),
    )
    second = replace(
        dataset[1],
        regime_parameters=_regime_parameters(re_value=20.0, ma_value=0.2),
    )
    task = NodeRegressionTask(
        input_fields=("momentum",),
        target_fields=("rho",),
        conditioning_parameters=("Ma", "Re"),
    )

    batch = task.pack_and_prepare([first, second], dataset.field_catalog)

    assert batch.conditioning_names == ("Ma", "Re")
    torch.testing.assert_close(
        batch.conditioning,
        torch.tensor([[0.1, 10.0], [0.2, 20.0]], dtype=batch.inputs.dtype),
    )


def test_regression_task_rejects_unavailable_or_inconsistent_conditioning() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, seed=3)
    unavailable = replace(
        dataset[0],
        regime_parameters=_regime_parameters(
            re_value=10.0,
            ma_value=0.1,
            available=False,
        ),
    )
    task = NodeRegressionTask(
        input_fields=("momentum",),
        target_fields=("rho",),
        conditioning_parameters=("Re",),
    )

    with pytest.raises(ValueError, match="available at inference"):
        task.pack_and_prepare([unavailable], dataset.field_catalog)

    first = replace(
        dataset[0],
        regime_parameters=_regime_parameters(re_value=10.0, ma_value=0.1),
    )
    second = replace(
        dataset[1],
        regime_parameters=_regime_parameters(
            re_value=20.0,
            ma_value=0.2,
            re_definition="different_reynolds_number",
        ),
    )
    with pytest.raises(ValueError, match="inconsistent definitions"):
        task.pack_and_prepare([first, second], dataset.field_catalog)


def test_regression_task_rejects_non_node_or_nonfloating_fields() -> None:
    dataset = SyntheticMeshDataset(num_samples=1)
    sample = dataset[0]
    global_catalog = FieldCatalog(
        (
            FieldSpec(
                name="proxy",
                support=FieldSupport.GLOBAL,
                role=FieldRole.GLOBAL_METADATA,
            ),
            dataset.field_catalog["rho"],
        )
    )
    proxy_sample = replace(
        sample,
        fields={**sample.fields, "proxy": torch.ones(sample.mesh.num_nodes)},
    )
    task = NodeRegressionTask(input_fields=("proxy",), target_fields=("rho",))

    with pytest.raises(ValueError, match="expected node support"):
        task.pack_and_prepare([proxy_sample], global_catalog)

    integer_sample = replace(
        sample,
        fields={**sample.fields, "rho": torch.ones(sample.mesh.num_nodes, dtype=torch.long)},
    )
    integer_task = NodeRegressionTask(input_fields=("rho",), target_fields=("rho",))
    with pytest.raises(TypeError, match="floating-point"):
        integer_task.pack_and_prepare([integer_sample], dataset.field_catalog)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_fields": (), "target_fields": ("rho",)},
        {"input_fields": ("rho",), "target_fields": ()},
        {"input_fields": ("rho", "rho"), "target_fields": ("rho",)},
        {"input_fields": "rho", "target_fields": ("rho",)},
    ],
)
def test_regression_task_rejects_invalid_field_selection(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        NodeRegressionTask(**kwargs)  # type: ignore[arg-type]
