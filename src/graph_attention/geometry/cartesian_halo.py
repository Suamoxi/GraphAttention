"""Candidate direct Cartesian halo edges for a *qualified* AVBP hex mesh.

Use only after qualify_m41_3_cartesian_mesh.py proves every hex is a
unit Cartesian cell with M37's 12 canonical physical edges. A count
of N=257**3 nodes alone is NOT a sufficient qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

ONE_HOP_OFFSETS = tuple(v for v in product((-1, 0, 1), repeat=3) if sum(abs(a) for a in v) == 1)
EXACT_TWO_HOP_OFFSETS = tuple(
    v for v in product((-2, -1, 0, 1, 2), repeat=3)
    if sum(abs(a) for a in v) == 2
)


@dataclass(frozen=True)
class CartesianPartitionEdges:
    owned_global: np.ndarray
    read_global: np.ndarray
    local_edge_index: np.ndarray
    dilated_edge_index: np.ndarray

    @property
    def halo_nodes(self) -> int:
        return len(self.read_global) - len(self.owned_global)


def cartesian_block_bounds(side: int, blocks: tuple[int, int, int]):
    """Yield (block_id, inclusive lower, exclusive upper), in x/y/z order."""
    if side < 3 or len(blocks) != 3 or any(n < 1 or n > side for n in blocks):
        raise ValueError("side>=3 and 1<=blocks[a]<=side are required")
    cuts = tuple(np.linspace(0, side, k + 1, dtype=np.int64) for k in blocks)
    for block_id, indices in enumerate(product(*(range(k) for k in blocks))):
        yield (
            block_id,
            tuple(int(cuts[a][i]) for a, i in enumerate(indices)),
            tuple(int(cuts[a][i + 1]) for a, i in enumerate(indices)),
        )


def _linear_box(side: int, lo: tuple[int, int, int], hi: tuple[int, int, int]) -> np.ndarray:
    a = np.arange(lo[0], hi[0], dtype=np.int64)
    b = np.arange(lo[1], hi[1], dtype=np.int64)
    c = np.arange(lo[2], hi[2], dtype=np.int64)
    return ((a[:, None, None] * side + b[None, :, None]) * side + c[None, None, :]).ravel()


def _owned_edges(
    side: int,
    linear_to_global: np.ndarray,
    owned_linear: np.ndarray,
    offsets: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    """All (global source, global owned target) directed edges, sorted M37-style."""
    x = owned_linear // (side * side)
    y = (owned_linear // side) % side
    z = owned_linear % side
    target = linear_to_global[owned_linear]
    pieces = []
    for dx, dy, dz in offsets:
        sx, sy, sz = x + dx, y + dy, z + dz
        valid = (
            (sx >= 0) & (sx < side) &
            (sy >= 0) & (sy < side) &
            (sz >= 0) & (sz < side)
        )
        if not np.any(valid):
            continue
        neighbors = (sx[valid] * side + sy[valid]) * side + sz[valid]
        pieces.append(np.stack((linear_to_global[neighbors], target[valid]), axis=0))
    if not pieces:
        return np.empty((2, 0), dtype=np.int64)
    edges = np.concatenate(pieces, axis=1).astype(np.int64, copy=False)
    order = np.lexsort((edges[1], edges[0]))
    return np.ascontiguousarray(edges[:, order])


def partition_cartesian_edges(
    linear_to_global: np.ndarray,
    *,
    side: int,
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
) -> CartesianPartitionEdges:
    """Build partition-local directed 1-hop and exact-2-hop graph.

    No global graph is built. One partition's edges alone are materialized.
    HDF5 node ordering may be arbitrary; the qualified inverse permutation
    maps Cartesian coordinates to the original HDF5 global node IDs.
    """
    if side < 3 or linear_to_global.shape != (side**3,):
        raise ValueError("linear_to_global must have [side**3] entries")
    if any(not (0 <= lower[a] < upper[a] <= side) for a in range(3)):
        raise ValueError("invalid Cartesian owned block bounds")
    # Strictly require a bijection and no sentinel -1 on small meshes;
    # the 257^3 path uses qualified mmap data proven complete by the scanner.
    if linear_to_global.dtype.kind not in "iu":
        raise TypeError("linear_to_global must contain integer global indices")
    owned_linear = _linear_box(side, lower, upper)
    halo_lower = tuple(max(0, a - 2) for a in lower)
    halo_upper = tuple(min(side, b + 2) for b in upper)
    read_linear = _linear_box(side, halo_lower, halo_upper)
    owned_global = np.sort(np.asarray(linear_to_global[owned_linear], dtype=np.int64))
    read_global = np.sort(np.asarray(linear_to_global[read_linear], dtype=np.int64))
    return CartesianPartitionEdges(
        owned_global=owned_global,
        read_global=read_global,
        local_edge_index=_owned_edges(side, linear_to_global, owned_linear, ONE_HOP_OFFSETS),
        dilated_edge_index=_owned_edges(side, linear_to_global, owned_linear, EXACT_TWO_HOP_OFFSETS),
    )
