"""Canonical Cartesian-grid connectivity transforms."""

from __future__ import annotations

from collections.abc import Sequence
from operator import index as operator_index

import torch


def cartesian_4_neighbor_edge_index(grid_shape: Sequence[int]) -> torch.Tensor:
    """Return bidirectional non-periodic 4-neighbour edges for a row-major 2-D grid.

    Node ordering follows ``node_id = first_index * n_second + second_index``,
    matching diffusion4avbp's fixed slice artifact convention where the first
    in-plane coordinate varies slowest.
    """

    if isinstance(grid_shape, (str, bytes)) or len(grid_shape) != 2:
        raise ValueError("grid_shape must contain exactly two dimensions")
    rows = _positive_int(grid_shape[0], "grid_shape[0]")
    cols = _positive_int(grid_shape[1], "grid_shape[1]")

    node_ids = torch.arange(rows * cols, dtype=torch.long).reshape(rows, cols)
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    if cols > 1:
        left = node_ids[:, :-1].reshape(-1)
        right = node_ids[:, 1:].reshape(-1)
        sources.extend((left, right))
        targets.extend((right, left))
    if rows > 1:
        upper = node_ids[:-1, :].reshape(-1)
        lower = node_ids[1:, :].reshape(-1)
        sources.extend((upper, lower))
        targets.extend((lower, upper))

    if not sources:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack((torch.cat(sources), torch.cat(targets)), dim=0)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result
