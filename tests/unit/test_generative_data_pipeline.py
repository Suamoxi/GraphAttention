from __future__ import annotations

import torch

from graph_attention.data import (
    FieldCatalog,
    FieldRole,
    FieldSpec,
    FieldSupport,
    Mesh,
    Sample,
    SyntheticMeshDataset,
)
from graph_attention.geometry import (
    build_attention_edge_indices,
    cartesian_4_neighbor_edge_index,
)
from graph_attention.tasks import DiffusionDenoisingTask
from graph_attention.training.data_pipeline import GraphTaskCollator


def test_generic_collator_preserves_disconnected_variable_graphs() -> None:
    dataset = SyntheticMeshDataset(
        num_samples=3,
        min_nodes=4,
        max_nodes=7,
        spatial_dim=2,
        seed=17,
    )
    task = DiffusionDenoisingTask(
        state_fields=("rho", "momentum"),
        timesteps=10,
    )
    collator = GraphTaskCollator(
        task,
        dataset.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )

    batch = collator([dataset[0], dataset[2]])

    assert batch.num_graphs == 2
    assert int(batch.ptr[1] - batch.ptr[0]) != int(batch.ptr[2] - batch.ptr[1])
    assert torch.all(batch.batch_index[batch.edge_index[0]] == batch.batch_index[batch.edge_index[1]])

    dilated = batch.attention_edge_indices["dilated"]
    assert torch.all(batch.batch_index[dilated[0]] == batch.batch_index[dilated[1]])


def test_fixed_grid_collation_matches_previous_shared_topology_construction() -> None:
    grid_shape = (2, 3)
    num_nodes = grid_shape[0] * grid_shape[1]
    coords = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    catalog = FieldCatalog(
        (
            FieldSpec(
                name="rho",
                support=FieldSupport.NODE,
                role=FieldRole.PRIMARY_STATE,
            ),
        )
    )
    samples = [
        Sample(
            sample_id=f"slice_{index}",
            mesh=Mesh(
                coords=coords,
                edge_index=torch.empty((2, 0), dtype=torch.long),
                mesh_id="shared-grid",
                metadata={"grid_shape_2d": grid_shape},
            ),
            fields={"rho": torch.arange(num_nodes, dtype=torch.float32) + index},
        )
        for index in range(2)
    ]
    task = DiffusionDenoisingTask(state_fields=("rho",), timesteps=10)
    collator = GraphTaskCollator(
        task,
        catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )

    batch = collator(samples)

    local = cartesian_4_neighbor_edge_index(grid_shape)
    expected_local = torch.cat((local, local + num_nodes), dim=1)
    torch.testing.assert_close(batch.edge_index, expected_local)

    one_dilated = build_attention_edge_indices(
        local,
        num_nodes,
        {"dilated": "exact_two_hop"},
    )["dilated"]
    expected_dilated = torch.cat((one_dilated, one_dilated + num_nodes), dim=1)
    torch.testing.assert_close(batch.attention_edge_indices["dilated"], expected_dilated)
