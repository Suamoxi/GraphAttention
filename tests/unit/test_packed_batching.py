from dataclasses import replace

import pytest
import torch

from graph_attention.data import (
    Mesh,
    MicrobatchBudget,
    Sample,
    SyntheticMeshDataset,
    pack_samples,
    partition_samples_by_budget,
)


def test_pack_samples_builds_disconnected_variable_graph_batch() -> None:
    dataset = SyntheticMeshDataset(
        num_samples=3,
        min_nodes=4,
        max_nodes=6,
        spatial_dim=2,
        seed=11,
    )
    samples = [dataset[index] for index in range(3)]

    batch = pack_samples(samples, node_field_names=("rho", "momentum"))

    assert batch.num_graphs == 3
    assert batch.num_nodes == 15
    assert batch.num_edges == sum(sample.mesh.num_edges for sample in samples)
    assert batch.sample_ids == tuple(sample.sample_id for sample in samples)
    assert batch.mesh_ids == tuple(sample.mesh.mesh_id for sample in samples)
    assert batch.case_ids == (None, None, None)
    torch.testing.assert_close(batch.ptr, torch.tensor([0, 4, 9, 15]))
    torch.testing.assert_close(
        batch.batch_index,
        torch.tensor([0] * 4 + [1] * 5 + [2] * 6),
    )
    torch.testing.assert_close(batch.coords, torch.cat([sample.mesh.coords for sample in samples]))
    torch.testing.assert_close(
        batch.fields["rho"],
        torch.cat([sample.fields["rho"] for sample in samples]),
    )
    torch.testing.assert_close(
        batch.fields["momentum"],
        torch.cat([sample.fields["momentum"] for sample in samples]),
    )

    offsets = (0, 4, 9)
    expected_edges = torch.cat(
        [
            sample.mesh.edge_index + offset
            for sample, offset in zip(samples, offsets, strict=True)
        ],
        dim=1,
    )
    torch.testing.assert_close(batch.edge_index, expected_edges)
    source, target = batch.edge_index
    torch.testing.assert_close(batch.batch_index[source], batch.batch_index[target])

    assert batch.node_weights is not None
    for graph_index in range(batch.num_graphs):
        start = int(batch.ptr[graph_index])
        stop = int(batch.ptr[graph_index + 1])
        torch.testing.assert_close(
            batch.node_weights[start:stop].sum(),
            torch.tensor(1.0),
        )

    for graph_index, sample in enumerate(samples):
        assert batch.reference_scales[graph_index] is sample.reference_scales
        assert batch.regime_parameters[graph_index] is sample.regime_parameters
        assert batch.sample_metadata[graph_index] is sample.metadata
        assert batch.mesh_metadata[graph_index] is sample.mesh.metadata


def test_pack_samples_supports_explicit_geometry_only_batch() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, min_nodes=4, max_nodes=5)
    samples = [dataset[0], dataset[1]]

    batch = pack_samples(samples)

    assert batch.fields == {}
    assert batch.num_nodes == 9
    assert batch.ptr.tolist() == [0, 4, 9]


def test_partition_samples_by_budget_preserves_selection_order() -> None:
    dataset = SyntheticMeshDataset(num_samples=3, min_nodes=4, max_nodes=6)
    samples = [dataset[index] for index in range(3)]

    groups = partition_samples_by_budget(
        samples,
        MicrobatchBudget(max_nodes=9, max_edges=16),
    )

    assert [[sample.sample_id for sample in group] for group in groups] == [
        [samples[0].sample_id, samples[1].sample_id],
        [samples[2].sample_id],
    ]
    assert sum(sample.mesh.num_nodes for sample in groups[0]) == 9
    assert sum(sample.mesh.num_edges for sample in groups[0]) == 16


def test_partition_samples_by_budget_can_be_edge_limited() -> None:
    dataset = SyntheticMeshDataset(num_samples=3, min_nodes=4, max_nodes=6)
    samples = [dataset[index] for index in range(3)]

    groups = partition_samples_by_budget(
        samples,
        MicrobatchBudget(max_nodes=100, max_edges=10),
    )

    assert tuple(len(group) for group in groups) == (1, 1, 1)


def test_partition_samples_by_budget_rejects_oversized_graph() -> None:
    sample = SyntheticMeshDataset(num_samples=3, min_nodes=4, max_nodes=6)[2]

    with pytest.raises(ValueError, match=r"nodes=6 \(max 5\).+edges=10 \(max 9\)"):
        partition_samples_by_budget(
            [sample],
            MicrobatchBudget(max_nodes=5, max_edges=9),
        )


@pytest.mark.parametrize(
    ("max_nodes", "max_edges", "error_type"),
    [
        (0, 1, ValueError),
        (1, 0, ValueError),
        (-1, 1, ValueError),
        (1, -1, ValueError),
        (True, 1, TypeError),
        (1.5, 2, TypeError),
    ],
)
def test_microbatch_budget_rejects_invalid_limits(
    max_nodes: object,
    max_edges: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        MicrobatchBudget(max_nodes=max_nodes, max_edges=max_edges)  # type: ignore[arg-type]


def test_partition_samples_by_budget_accepts_empty_selection() -> None:
    budget = MicrobatchBudget(max_nodes=10, max_edges=10)
    assert partition_samples_by_budget([], budget) == ()


def test_pack_samples_rejects_unknown_or_duplicate_node_fields() -> None:
    sample = SyntheticMeshDataset(num_samples=1)[0]

    with pytest.raises(KeyError, match="temperature"):
        pack_samples([sample], node_field_names=("temperature",))
    with pytest.raises(ValueError, match="duplicates"):
        pack_samples([sample], node_field_names=("rho", "rho"))


def test_pack_samples_rejects_inconsistent_node_field_shape() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, min_nodes=4, max_nodes=5, spatial_dim=2)
    first = dataset[0]
    second = dataset[1]
    second = replace(
        second,
        fields={
            **second.fields,
            "momentum": torch.zeros((second.mesh.num_nodes, 3)),
        },
    )

    with pytest.raises(ValueError, match="consistent trailing shape"):
        pack_samples([first, second], node_field_names=("momentum",))


def test_pack_samples_rejects_mixed_node_weight_availability() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, min_nodes=4, max_nodes=5)
    first = dataset[0]
    second = dataset[1]
    second = replace(second, mesh=replace(second.mesh, node_weights=None))

    with pytest.raises(ValueError, match="cannot mix samples with and without node_weights"):
        pack_samples([first, second], node_field_names=("rho",))


def test_pack_samples_rejects_coordinate_dimension_mismatch() -> None:
    sample_2d = SyntheticMeshDataset(num_samples=1, spatial_dim=2)[0]
    sample_3d = SyntheticMeshDataset(num_samples=1, spatial_dim=3)[0]

    with pytest.raises(ValueError, match="coordinate dimension"):
        pack_samples([sample_2d, sample_3d], node_field_names=("rho",))


def test_pack_samples_rejects_empty_graph() -> None:
    sample = Sample(
        sample_id="empty",
        mesh=Mesh(
            coords=torch.empty((0, 2)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        ),
        fields={"rho": torch.empty((0,))},
    )

    with pytest.raises(ValueError, match="has no mesh nodes"):
        pack_samples([sample], node_field_names=("rho",))
