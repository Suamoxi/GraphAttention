"""Input-state Jacobian diagnostics for one learned velocity evaluation.

All sensitivities are squared derivatives in standardized state space.
No full N*C by N*C Jacobian, attention rollout, or ODE differentiation.
"""
from __future__ import annotations

import numpy as np
import torch


def query_jacobian(
    velocity: torch.Tensor,
    state: torch.Tensor,
    query: int,
    *,
    retain_graph: bool = False,
) -> np.ndarray:
    """Return [input node, output channel, input channel] for one query."""
    if state.ndim != 2 or velocity.shape != state.shape:
        raise ValueError("expected matching [N,C] state and velocity")
    if not state.requires_grad or not 0 <= query < len(state):
        raise ValueError("invalid differentiable state or query index")
    rows = []
    for c in range(velocity.shape[1]):
        (gradient,) = torch.autograd.grad(
            velocity[query, c], state,
            retain_graph=retain_graph or c + 1 < velocity.shape[1],
            create_graph=False,
        )
        rows.append(gradient.detach().to(device="cpu", dtype=torch.float64).numpy())
    jacobian = np.stack(rows, axis=1)
    if not np.isfinite(jacobian).all():
        raise ValueError("nonfinite input-state derivative")
    return jacobian


def qualify_open_cartesian_edges(
    ijk: np.ndarray, shape: tuple[int, int, int], edges: np.ndarray,
) -> None:
    """Prove the local directed graph is the full open axial-neighbor lattice."""
    xyz = np.asarray(ijk)
    n = int(np.prod(shape))
    if xyz.shape != (n, 3) or xyz.dtype.kind not in "iu":
        raise ValueError("expected integer Cartesian indices")
    if np.unique(np.ravel_multi_index(xyz.T, shape)).size != n:
        raise ValueError("Cartesian indices are not bijective")
    e = np.asarray(edges)
    expected = 2 * sum((shape[a] - 1) * shape[(a + 1) % 3] * shape[(a + 2) % 3]
                       for a in range(3))
    if (e.shape != (2, expected) or e.dtype.kind not in "iu"
            or np.any(e < 0) or np.any(e >= n)):
        raise ValueError("unexpected local edge count or indices")
    if np.any(np.abs(xyz[e[0]] - xyz[e[1]]).sum(axis=1) != 1):
        raise ValueError("local graph has non-axial or wrapping edges")
    if np.unique(e[0].astype(np.int64) * n + e[1]).size != expected:
        raise ValueError("local graph contains repeated directed edges")


def shell_statistics(jacobian: np.ndarray, distances: np.ndarray) -> dict:
    """Reduce squared derivatives by graph distance, preserving channel pairs.

    Radii refer to total shell energy (not per-node averages). An identically
    insensitive output has undefined normalized fractions/radii, not radius 0.
    """
    j = np.asarray(jacobian, dtype=np.float64)
    d = np.asarray(distances)
    if (j.ndim != 3 or j.shape[0] == 0 or j.shape[1] == 0 or j.shape[2] == 0
            or d.shape != (len(j),) or d.dtype.kind not in "iu"
            or np.any(d < 0) or not np.isfinite(j).all()):
        raise ValueError("invalid Jacobian or nonnegative integer distances")
    energy = np.square(j)
    if not np.isfinite(energy).all():
        raise ValueError("nonfinite squared derivatives")
    counts = np.bincount(d)
    per_node = energy.sum(axis=(1, 2))
    pairs = np.empty((len(counts), j.shape[1], j.shape[2]), dtype=np.float64)
    for a in range(j.shape[1]):
        for b in range(j.shape[2]):
            pairs[:, a, b] = np.bincount(d, weights=energy[:, a, b], minlength=len(counts))
    shell = pairs.sum(axis=(1, 2))
    mean = np.divide(shell, counts, out=np.zeros_like(shell), where=counts > 0)
    total = float(shell.sum())
    fractions = shell / total if total > 0 else None
    cumulative = np.cumsum(shell) / total if total > 0 else None
    if cumulative is not None:
        cumulative[-1] = 1.0
    radii = {
        f"R{int(p * 100)}": int(np.searchsorted(cumulative, p)) if cumulative is not None else None
        for p in (0.50, 0.90, 0.95)
    }
    return dict(counts=counts, per_node=per_node, pairs=pairs, shell=shell,
                mean=mean, total=total, fractions=fractions, cumulative=cumulative,
                radii=radii)
