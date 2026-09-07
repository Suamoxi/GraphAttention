import pytest
import torch

from graph_attention.geometry import cartesian_4_neighbor_edge_index


def test_cartesian_4_neighbor_edges_match_2_by_3_grid() -> None:
    edge_index = cartesian_4_neighbor_edge_index((2, 3))

    assert edge_index.shape == (2, 14)
    edges = {tuple(edge) for edge in edge_index.t().tolist()}
    expected = {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (3, 4),
        (4, 3),
        (4, 5),
        (5, 4),
        (0, 3),
        (3, 0),
        (1, 4),
        (4, 1),
        (2, 5),
        (5, 2),
    }
    assert edges == expected


def test_cartesian_4_neighbor_single_node_has_no_edges() -> None:
    edge_index = cartesian_4_neighbor_edge_index((1, 1))

    assert torch.equal(edge_index, torch.empty((2, 0), dtype=torch.long))


@pytest.mark.parametrize("shape", [(), (2,), (2, 3, 4), (0, 2), (2, -1)])
def test_cartesian_4_neighbor_rejects_invalid_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        cartesian_4_neighbor_edge_index(shape)
