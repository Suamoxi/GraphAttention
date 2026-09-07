"""Deterministic relative geometry derived from node coordinates and sparse edges."""

from __future__ import annotations

import torch


def edge_relative_displacement(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Return target-to-source displacement for every directed edge.

    For ``edge_index[0, e] = j`` and ``edge_index[1, e] = i``, the returned
    row is ``coords[j] - coords[i]``. This matches the M8/M9 convention where
    node ``i`` is the attention query and node ``j`` supplies key/value data.
    """

    if coords.ndim != 2:
        raise ValueError("coords must have shape [N, D]")
    if not coords.is_floating_point():
        raise TypeError("coords must use a floating-point dtype")
    if not torch.isfinite(coords).all():
        raise ValueError("coords contain NaN or Inf")

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must use torch.long indices")
    if edge_index.device != coords.device:
        raise ValueError("edge_index and coords must be on the same device")
    if edge_index.numel() == 0:
        return coords.new_empty((0, coords.shape[1]))
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= coords.shape[0]:
        raise ValueError("edge_index contains an out-of-range node index")

    source = edge_index[0]
    target = edge_index[1]
    return coords[source] - coords[target]
