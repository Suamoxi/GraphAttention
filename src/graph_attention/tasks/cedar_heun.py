"""Heun ODE integration for the M48 CEDAR E1/E2 inference ablation.

This module does not implement CEDAR. The caller supplies the original
full-domain projection as a callable; no cropped-domain projection is valid.
Both strategies return the *raw* terminal state so that the caller can apply
one final float64 CEDAR projection during physical reconstruction.
"""
from __future__ import annotations

from collections.abc import Callable

import torch


@torch.no_grad()
def integrate_cedar_heun(
    initial: torch.Tensor,
    predict_velocity: Callable[[torch.Tensor, float], torch.Tensor],
    project_velocity: Callable[[torch.Tensor], torch.Tensor],
    *,
    steps: int,
    strategy: str,
    observer: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """E1 projects both Heun stages; E2 does not project intermediate velocities.

    Initial must be a globally CEDAR-projected Gaussian. `predict_velocity`
    returns a complete assembled fine-grid velocity, even when evaluated on
    exact-halo graph tiles. The terminal constraint projection is deliberately
    outside this integrator and is required for BOTH strategies.
    """
    if strategy not in ("E1", "E2"):
        raise ValueError("strategy must be E1 (each velocity) or E2 (terminal only)")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(initial, torch.Tensor) or not initial.is_floating_point():
        raise TypeError("initial must be a floating-point tensor")
    if not torch.isfinite(initial).all():
        raise ValueError("nonfinite initial state")

    state = initial.clone()
    dt = 1.0 / steps
    if observer is not None:
        observer(0, state)
    for index in range(steps):
        t = index * dt
        k1 = predict_velocity(state, t)
        if k1.shape != state.shape or k1.device != state.device or k1.dtype != state.dtype:
            raise ValueError("model must return a complete velocity matching the state")
        if strategy == "E1":
            k1 = project_velocity(k1)
        predictor = state + dt * k1
        k2 = predict_velocity(predictor, (index + 1) * dt)
        if k2.shape != state.shape or k2.device != state.device or k2.dtype != state.dtype:
            raise ValueError("model must return a complete velocity matching the state")
        if strategy == "E1":
            k2 = project_velocity(k2)
        state = state + (0.5 * dt) * (k1 + k2)
        if not torch.isfinite(state).all():
            raise ValueError("nonfinite Heun trajectory")
        if observer is not None:
            observer(index + 1, state)
    return state
