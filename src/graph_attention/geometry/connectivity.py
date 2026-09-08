"""Deterministic graph connectivity derived from supplied mesh topology."""

from __future__ import annotations

from operator import index as operator_index

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


def exact_two_hop_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Return directed edges whose shortest supplied-graph distance is exactly two.

    The result contains the Boolean support of length-two walks after removing
    self-loops and every pair already present in ``edge_index``. Duplicate
    length-two paths therefore produce one directed edge. The construction stays
    sparse and never materializes a dense ``[num_nodes, num_nodes]`` adjacency.

    For disconnected packed graphs, the result remains disconnected because a
    length-two walk cannot cross between components when the supplied edges do not.
    """

    node_count = _validated_num_nodes(num_nodes)
    _validate_edge_index(edge_index, node_count)
    if edge_index.shape[1] == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    source, target = edge_index

    # Group incoming and outgoing edges by their shared intermediate node. For
    # each node k, every incoming i->k is paired with every outgoing k->j.
    incoming_order = torch.argsort(target)
    outgoing_order = torch.argsort(source)
    incoming_sources = source[incoming_order]
    outgoing_targets = target[outgoing_order]

    incoming_counts = torch.bincount(target, minlength=node_count)
    outgoing_counts = torch.bincount(source, minlength=node_count)
    path_counts = incoming_counts * outgoing_counts
    total_paths = int(path_counts.sum())
    if total_paths == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    intermediate = torch.repeat_interleave(
        torch.arange(node_count, device=edge_index.device),
        path_counts,
    )
    group_starts = torch.cumsum(path_counts, dim=0) - path_counts
    offset_in_group = torch.arange(total_paths, device=edge_index.device) - torch.repeat_interleave(
        group_starts,
        path_counts,
    )

    incoming_starts = torch.cumsum(incoming_counts, dim=0) - incoming_counts
    outgoing_starts = torch.cumsum(outgoing_counts, dim=0) - outgoing_counts
    outgoing_count_for_path = outgoing_counts[intermediate]

    incoming_position = incoming_starts[intermediate] + torch.div(
        offset_in_group,
        outgoing_count_for_path,
        rounding_mode="floor",
    )
    outgoing_position = outgoing_starts[intermediate] + torch.remainder(
        offset_in_group,
        outgoing_count_for_path,
    )

    path_source = incoming_sources[incoming_position]
    path_target = outgoing_targets[outgoing_position]
    nonself = path_source != path_target
    if not bool(nonself.any()):
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    path_keys = path_source[nonself] * node_count + path_target[nonself]
    path_keys = torch.unique(path_keys, sorted=True)

    # Exact two-hop means that direct one-hop neighbours are excluded even when
    # another length-two path connects the same pair (for example in triangles).
    direct_keys = torch.unique(source * node_count + target, sorted=True)
    insertion = torch.searchsorted(direct_keys, path_keys)
    bounded = insertion.clamp(max=direct_keys.numel() - 1)
    is_direct = (insertion < direct_keys.numel()) & (direct_keys[bounded] == path_keys)
    exact_keys = path_keys[~is_direct]

    return torch.stack(
        (
            torch.div(exact_keys, node_count, rounding_mode="floor"),
            torch.remainder(exact_keys, node_count),
        ),
        dim=0,
    ).contiguous()


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


def _validated_num_nodes(num_nodes: int) -> int:
    if isinstance(num_nodes, bool):
        raise TypeError("num_nodes must be an integer")
    try:
        node_count = operator_index(num_nodes)
    except TypeError as exc:
        raise TypeError("num_nodes must be an integer") from exc
    if node_count <= 0:
        raise ValueError("num_nodes must be positive")
    return node_count


def _validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must use torch.long indices")
    if edge_index.numel() == 0:
        return
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
        raise ValueError("edge_index contains an out-of-range node index")
