import torch

from graph_attention.geometry import exact_two_hop_edge_index


def _bidirectional(pairs: list[tuple[int, int]]) -> torch.Tensor:
    directed = pairs + [(target, source) for source, target in pairs]
    return torch.tensor(directed, dtype=torch.long).T.contiguous()


def _edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(map(tuple, edge_index.T.tolist()))


def test_exact_two_hop_chain_excludes_local_and_self_edges() -> None:
    local = _bidirectional([(0, 1), (1, 2), (2, 3)])

    dilated = exact_two_hop_edge_index(local, num_nodes=4)

    assert _edge_set(dilated) == {(0, 2), (2, 0), (1, 3), (3, 1)}
    assert _edge_set(local).isdisjoint(_edge_set(dilated))
    assert all(source != target for source, target in _edge_set(dilated))


def test_exact_two_hop_triangle_removes_pairs_that_are_already_direct() -> None:
    local = _bidirectional([(0, 1), (1, 2), (2, 0)])

    dilated = exact_two_hop_edge_index(local, num_nodes=3)

    assert dilated.shape == (2, 0)


def test_exact_two_hop_preserves_disconnected_packed_components() -> None:
    local = _bidirectional([(0, 1), (1, 2), (3, 4), (4, 5)])

    dilated = exact_two_hop_edge_index(local, num_nodes=6)

    assert _edge_set(dilated) == {(0, 2), (2, 0), (3, 5), (5, 3)}
    assert not any((source < 3) != (target < 3) for source, target in _edge_set(dilated))


def test_exact_two_hop_is_invariant_to_duplicate_length_two_paths() -> None:
    # Nodes 0 and 3 are linked through both 1 and 2, but only one 0->3 edge
    # should remain in the Boolean support of A^2.
    local = _bidirectional([(0, 1), (1, 3), (0, 2), (2, 3)])

    dilated = exact_two_hop_edge_index(local, num_nodes=4)

    edges = _edge_set(dilated)
    assert (0, 3) in edges
    assert (3, 0) in edges
    assert len([edge for edge in edges if edge == (0, 3)]) == 1
