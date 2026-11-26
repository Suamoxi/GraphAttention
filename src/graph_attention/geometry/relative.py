"""Deterministic relative geometry derived from node coordinates and sparse edges."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isclose, isfinite

import torch


def periodic_length_from_mesh_metadata(
    metadata: Sequence[Mapping[str, object]],
) -> float | None:
    """Extract the shared nondimensional box length for packed periodic graphs.

    Absence preserves all historical open-mesh training/generation paths.
    Periodic batches must have the same scalar box length; no assumption is
    made about the case's dimensional L_ref.
    """
    if not metadata:
        return None
    values = [m.get("periodic_length_nondimensional") for m in metadata]
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError("mixed periodic and nonperiodic graphs in one packed batch")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError("periodic length metadata must be a finite scalar number")
    lengths = [float(v) for v in values]
    if any(not isfinite(v) or v <= 0 for v in lengths):
        raise ValueError("periodic length metadata must be finite and positive")
    if any(not isclose(v, lengths[0], rel_tol=1e-6, abs_tol=0.0) for v in lengths[1:]):
        raise ValueError("unequal nondimensional periodic lengths in packed batch")
    return lengths[0]


def edge_relative_displacement(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    periodic_length: float | None = None,
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
    displacement = coords[source] - coords[target]
    if periodic_length is None:
        return displacement
    length = float(periodic_length)
    if not isfinite(length) or length <= 0:
        raise ValueError("periodic_length must be finite and positive")
    half = length / 2.0
    return torch.remainder(displacement + half, length) - half
