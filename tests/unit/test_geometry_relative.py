import pytest
import torch

from graph_attention.geometry import edge_relative_displacement


def test_edge_relative_displacement_matches_target_to_source_vector() -> None:
    coords = torch.tensor([[0.0, 0.0], [2.0, 1.0], [3.0, -1.0]])
    edge_index = torch.tensor([[1, 0, 2, 1], [0, 1, 1, 2]], dtype=torch.long)

    displacement = edge_relative_displacement(coords, edge_index)

    expected = torch.tensor([[2.0, 1.0], [-2.0, -1.0], [1.0, -2.0], [-1.0, 2.0]])
    torch.testing.assert_close(displacement, expected)


def test_edge_relative_displacement_is_translation_invariant() -> None:
    coords = torch.tensor([[0.2, -0.4, 1.0], [1.2, 0.1, -0.5], [-0.3, 0.7, 0.4]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    translation = torch.tensor([3.0, -2.0, 0.5])

    reference = edge_relative_displacement(coords, edge_index)
    translated = edge_relative_displacement(coords + translation, edge_index)

    torch.testing.assert_close(translated, reference, rtol=1e-6, atol=1e-6)


def test_reverse_edges_have_opposite_relative_displacement() -> None:
    coords = torch.tensor([[0.0, 0.0], [1.5, -2.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    displacement = edge_relative_displacement(coords, edge_index)

    torch.testing.assert_close(displacement[0], -displacement[1])


@pytest.mark.parametrize(
    ("coords", "edge_index", "error"),
    [
        (torch.zeros(3), torch.empty((2, 0), dtype=torch.long), ValueError),
        (torch.zeros((3, 2), dtype=torch.long), torch.empty((2, 0), dtype=torch.long), TypeError),
        (torch.zeros((3, 2)), torch.zeros((2, 1), dtype=torch.int32), TypeError),
        (torch.zeros((3, 2)), torch.tensor([[0], [3]], dtype=torch.long), ValueError),
    ],
)
def test_edge_relative_displacement_rejects_invalid_inputs(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        edge_relative_displacement(coords, edge_index)
