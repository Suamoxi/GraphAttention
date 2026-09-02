import torch

from graph_attention.data import Mesh
from graph_attention.geometry import hex_connectivity_to_edge_index, mesh_with_hex_edge_index


def test_one_hex_produces_24_directed_edges_without_self_loops() -> None:
    connectivity = torch.arange(8, dtype=torch.long).reshape(1, 8)

    edge_index = hex_connectivity_to_edge_index(connectivity, num_nodes=8)

    assert edge_index.shape == (2, 24)
    assert not torch.any(edge_index[0] == edge_index[1])
    edges = {tuple(edge) for edge in edge_index.transpose(0, 1).tolist()}
    assert (0, 1) in edges
    assert (1, 0) in edges
    assert (0, 4) in edges
    assert (4, 0) in edges


def test_shared_cell_edges_are_deduplicated() -> None:
    first = torch.arange(8, dtype=torch.long)
    second = torch.tensor([1, 8, 9, 2, 5, 10, 11, 6], dtype=torch.long)
    connectivity = torch.stack((first, second))

    edge_index = hex_connectivity_to_edge_index(connectivity, num_nodes=12)
    edges = edge_index.transpose(0, 1)

    assert torch.unique(edges, dim=0).shape[0] == edges.shape[0]


def test_mesh_transform_preserves_native_connectivity_and_metadata() -> None:
    connectivity = torch.arange(8, dtype=torch.long).reshape(1, 8)
    mesh = Mesh(
        coords=torch.zeros((8, 3)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        cell_connectivity=connectivity,
        metadata={"cell_type": "hex"},
    )

    transformed = mesh_with_hex_edge_index(mesh)

    assert transformed.num_edges == 24
    assert transformed.cell_connectivity is connectivity
    assert transformed.metadata == mesh.metadata
