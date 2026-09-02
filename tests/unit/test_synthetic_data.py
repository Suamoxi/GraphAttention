import pytest
import torch

from graph_attention.data import SyntheticMeshDataset


def test_synthetic_dataset_exercises_variable_mesh_contracts() -> None:
    dataset = SyntheticMeshDataset(
        num_samples=6,
        min_nodes=4,
        max_nodes=9,
        spatial_dim=3,
        seed=7,
    )
    samples = [dataset[index] for index in range(len(dataset))]

    assert {sample.metadata["topology"] for sample in samples} == {"chain", "cycle", "star"}
    assert len({sample.mesh.num_nodes for sample in samples}) > 1
    assert len({sample.mesh.num_edges for sample in samples}) > 1

    for sample in samples:
        sample.validate_against(dataset.field_catalog)
        assert sample.mesh.spatial_dim == 3
        assert sample.fields["rho"].shape == (sample.mesh.num_nodes,)
        assert sample.fields["momentum"].shape == (sample.mesh.num_nodes, 3)
        torch.testing.assert_close(sample.mesh.node_weights.sum(), torch.tensor(1.0))


def test_synthetic_dataset_is_deterministic_by_index() -> None:
    dataset = SyntheticMeshDataset(num_samples=5, seed=123)

    first = dataset[3]
    _ = dataset[1]
    second = dataset[3]

    assert first.sample_id == second.sample_id
    assert first.metadata == second.metadata
    torch.testing.assert_close(first.mesh.coords, second.mesh.coords)
    torch.testing.assert_close(first.mesh.edge_index, second.mesh.edge_index)
    torch.testing.assert_close(first.fields["rho"], second.fields["rho"])
    torch.testing.assert_close(first.fields["momentum"], second.fields["momentum"])


def test_synthetic_dataset_does_not_advance_global_torch_rng() -> None:
    dataset = SyntheticMeshDataset(num_samples=3, seed=5)

    torch.manual_seed(99)
    expected = torch.rand(4)

    torch.manual_seed(99)
    _ = dataset[0]
    actual = torch.rand(4)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_samples": 0},
        {"min_nodes": 2},
        {"min_nodes": 5, "max_nodes": 4},
        {"spatial_dim": 1},
        {"seed": -1},
    ],
)
def test_synthetic_dataset_rejects_invalid_configuration(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        SyntheticMeshDataset(**kwargs)
