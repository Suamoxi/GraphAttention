"""Deterministic graph connectivity derived from supplied mesh topology."""

from __future__ import annotations

import torch

from graph_attention.data.contracts import Mesh

_HEX_LOCAL_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def hex_connectivity_to_edge_index(
    cell_connectivity: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Convert zero-based hexahedral cell connectivity to directed sparse edges.

    Each physical hexahedral edge is represented in both directions. Duplicate
    edges shared by adjacent cells are removed. Self-loops are not introduced.
    """

    if cell_connectivity.ndim != 2 or cell_connectivity.shape[1] != 8:
        raise ValueError("hex cell connectivity must have shape [C, 8]")
    if cell_connectivity.dtype != torch.long:
        raise TypeError("hex cell connectivity must use torch.long indices")
    if cell_connectivity.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=cell_connectivity.device)
    if int(cell_connectivity.min()) < 0 or int(cell_connectivity.max()) >= num_nodes:
        raise ValueError("hex cell connectivity contains an out-of-range node index")

    local_edges = torch.tensor(
        _HEX_LOCAL_EDGES,
        dtype=torch.long,
        device=cell_connectivity.device,
    )
    pairs = cell_connectivity[:, local_edges].reshape(-1, 2)
    pairs = torch.sort(pairs, dim=1).values
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs = torch.unique(pairs, dim=0)

    directed = torch.cat((pairs, pairs.flip(dims=(1,))), dim=0)
    key = directed[:, 0] * max(num_nodes, 1) + directed[:, 1]
    directed = directed[torch.argsort(key)]
    return directed.transpose(0, 1).contiguous()


def mesh_with_hex_edge_index(mesh: Mesh) -> Mesh:
    """Return a mesh with graph edges constructed from native hex connectivity."""

    if mesh.cell_connectivity is None:
        raise ValueError("mesh has no cell_connectivity to convert")
    edge_index = hex_connectivity_to_edge_index(mesh.cell_connectivity, mesh.num_nodes)
    return Mesh(
        coords=mesh.coords,
        edge_index=edge_index,
        mesh_id=mesh.mesh_id,
        node_weights=mesh.node_weights,
        cell_connectivity=mesh.cell_connectivity,
        metadata=mesh.metadata,
    )
