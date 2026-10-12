"""Deterministic ownership partitioning for halo-based graph execution.

Computational partitions do not define new physical graphs. Every global node
has exactly one owner. Only original-graph connectivity determines attention.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def coordinate_block_partition_ids(
    coords: torch.Tensor,
    *,
    blocks: Sequence[int],
) -> torch.Tensor:
    """Assign nodes to spatial blocks; works with arbitrarily ordered nodes."""
    if coords.ndim != 2 or not coords.is_floating_point():
        raise ValueError("coords must have floating shape [N, spatial_dim]")
    if coords.shape[0] == 0 or not bool(torch.isfinite(coords).all()):
        raise ValueError("coords must be nonempty and finite")
    counts = tuple(blocks)
    if len(counts) != coords.shape[1] or any(
        not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in counts
    ):
        raise ValueError("blocks must contain one positive integer per spatial axis")

    lo = coords.amin(dim=0)
    span = coords.amax(dim=0) - lo
    if bool(torch.any(span <= 0)):
        raise ValueError("coordinate-block partitioning requires nonzero axis extents")
    num = torch.tensor(counts, device=coords.device, dtype=coords.dtype)
    bins = torch.floor(((coords - lo) / span) * num).long()
    bins = torch.minimum(bins, torch.tensor(counts, dtype=torch.long, device=coords.device) - 1)
    partition_ids = bins[:, 0]
    for axis in range(1, len(counts)):
        partition_ids = partition_ids * counts[axis] + bins[:, axis]
    # Unstructured meshes can leave spatial bins empty. Compact occupied bins
    # in spatial order so the ownership validator never sees missing labels.
    return torch.unique(partition_ids, sorted=True, return_inverse=True)[1]


def validate_partition_ids(partition_ids: torch.Tensor, num_nodes: int) -> int:
    """Require contiguous partition labels 0..P-1 and no empty partitions."""
    if partition_ids.ndim != 1 or partition_ids.shape[0] != num_nodes:
        raise ValueError("partition_ids must have one entry per graph node")
    if partition_ids.dtype != torch.long:
        raise TypeError("partition_ids must use torch.long")
    if num_nodes < 1:
        raise ValueError("graph must contain at least one node")
    unique = torch.unique(partition_ids, sorted=True)
    if not torch.equal(
        unique, torch.arange(unique.numel(), device=unique.device, dtype=torch.long)
    ):
        raise ValueError("partition IDs must be contiguous from 0 with no gaps")
    return int(unique.numel())
