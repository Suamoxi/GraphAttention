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
    candidates = _composed_nonself_keys(edge_index, edge_index, node_count)
    direct = torch.unique(edge_index[0]*node_count+edge_index[1],sorted=True)
    return _keys_to_edges(_subtract_keys(candidates,direct),node_count)


def exact_three_hop_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Directed shortest-path distance exactly three, without dense adjacency."""
    two = exact_two_hop_edge_index(edge_index,num_nodes)
    return _exact_three_hop_from_two(edge_index,two,num_nodes)


def _exact_three_hop_from_two(edge_index, two_hop, num_nodes):
    candidates = _composed_nonself_keys(two_hop,edge_index,num_nodes)
    shorter = torch.unique(torch.cat((
        edge_index[0]*num_nodes+edge_index[1],
        two_hop[0]*num_nodes+two_hop[1],
    )),sorted=True)
    return _keys_to_edges(_subtract_keys(candidates,shorter),num_nodes)


def _composed_nonself_keys(lhs, rhs, node_count):
    """Sparse Boolean composition by shared intermediate node."""
    left_source,left_target = lhs
    right_source,right_target = rhs
    incoming_sources = left_source[torch.argsort(left_target)]
    outgoing_targets = right_target[torch.argsort(right_source)]
    incoming_counts = torch.bincount(left_target,minlength=node_count)
    outgoing_counts = torch.bincount(right_source,minlength=node_count)
    path_counts = incoming_counts*outgoing_counts
    total_paths = int(path_counts.sum())
    if total_paths == 0:
        return lhs.new_empty((0,))
    intermediate = torch.repeat_interleave(torch.arange(node_count,device=lhs.device),path_counts)
    group_starts = torch.cumsum(path_counts,dim=0)-path_counts
    offset = torch.arange(total_paths,device=lhs.device)-torch.repeat_interleave(group_starts,path_counts)
    incoming_starts = torch.cumsum(incoming_counts,dim=0)-incoming_counts
    outgoing_starts = torch.cumsum(outgoing_counts,dim=0)-outgoing_counts
    count = outgoing_counts[intermediate]
    source = incoming_sources[incoming_starts[intermediate]+torch.div(offset,count,rounding_mode="floor")]
    target = outgoing_targets[outgoing_starts[intermediate]+torch.remainder(offset,count)]
    nonself = source != target
    return torch.unique(source[nonself]*node_count+target[nonself],sorted=True)


def _subtract_keys(keys, excluded):
    if not keys.numel() or not excluded.numel():
        return keys
    insertion = torch.searchsorted(excluded,keys)
    bounded = insertion.clamp(max=excluded.numel()-1)
    return keys[~((insertion<excluded.numel()) & (excluded[bounded]==keys))]


def _keys_to_edges(keys, node_count):
    return torch.stack((torch.div(keys,node_count,rounding_mode="floor"),
                        torch.remainder(keys,node_count)),dim=0).contiguous()


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
