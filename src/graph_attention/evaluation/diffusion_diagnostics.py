"""Numerical diagnostics for epsilon-prediction diffusion models."""

from __future__ import annotations

import torch


def reconstruct_x0_from_epsilon(
    noisy_state: torch.Tensor,
    epsilon: torch.Tensor,
    alpha_bar_t: torch.Tensor | float,
) -> torch.Tensor:
    """Reconstruct clean standardized state from an epsilon prediction."""

    if noisy_state.shape != epsilon.shape:
        raise ValueError("noisy_state and epsilon must have identical shape")
    alpha = torch.as_tensor(
        alpha_bar_t,
        device=noisy_state.device,
        dtype=noisy_state.dtype,
    )
    if alpha.numel() != 1:
        raise ValueError("alpha_bar_t must be scalar for fixed-timestep diagnostics")
    if not bool(torch.isfinite(alpha)) or float(alpha) <= 0.0 or float(alpha) > 1.0:
        raise ValueError("alpha_bar_t must be finite and lie in (0, 1]")
    return (
        noisy_state - torch.sqrt(1.0 - alpha) * epsilon
    ) / torch.sqrt(alpha)


def graph_channel_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ptr: torch.Tensor,
) -> torch.Tensor:
    """Return per-graph, per-channel MSE with equal graph semantics."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have identical shape [total_nodes, C]")
    _validate_ptr(ptr, prediction.shape[0])
    rows = []
    squared = (prediction - target).square()
    for start, stop in zip(ptr[:-1].tolist(), ptr[1:].tolist(), strict=True):
        rows.append(squared[int(start) : int(stop)].mean(dim=0))
    return torch.stack(rows, dim=0)


def graph_channel_moments(
    values: torch.Tensor,
    ptr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-graph channel mean, population std, and max absolute value."""

    if values.ndim != 2:
        raise ValueError("values must have shape [total_nodes, C]")
    _validate_ptr(ptr, values.shape[0])
    means = []
    stds = []
    maxima = []
    for start, stop in zip(ptr[:-1].tolist(), ptr[1:].tolist(), strict=True):
        graph = values[int(start) : int(stop)]
        means.append(graph.mean(dim=0))
        stds.append(graph.std(dim=0, unbiased=False))
        maxima.append(graph.abs().amax(dim=0))
    return torch.stack(means), torch.stack(stds), torch.stack(maxima)


def epsilon_to_x0_mse_factor(alpha_bar_t: float) -> float:
    """Return the exact MSE amplification factor (1-alpha_bar)/alpha_bar."""

    alpha = float(alpha_bar_t)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha_bar_t must lie in (0, 1]")
    return (1.0 - alpha) / alpha


def _validate_ptr(ptr: torch.Tensor, total_nodes: int) -> None:
    if ptr.ndim != 1 or ptr.numel() < 2:
        raise ValueError("ptr must be one-dimensional with at least two entries")
    if ptr.dtype not in (torch.int32, torch.int64):
        raise TypeError("ptr must use an integer dtype")
    if int(ptr[0]) != 0 or int(ptr[-1]) != total_nodes:
        raise ValueError("ptr must span exactly all nodes")
    if bool(torch.any(ptr[1:] <= ptr[:-1])):
        raise ValueError("ptr must define non-empty strictly increasing graph ranges")
