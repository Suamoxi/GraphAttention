"""Linear-beta DDPM schedule ablation for packed node fields.

This module keeps the M13 epsilon-prediction task, loss, time conditioning,
and reverse sampler unchanged while replacing only the discrete forward-noise
schedule with the original DDPM-style linear beta schedule.
"""

from __future__ import annotations

import torch

from .diffusion import DiffusionDenoisingTask


class LinearBetaDiffusionDenoisingTask(DiffusionDenoisingTask):
    """M13-compatible DDPM task with a finite linear beta schedule.

    The default experiment uses ``T=1000`` and
    ``beta_t = linspace(1e-4, 2e-2, T)``.  Unlike the cosine baseline, no beta
    clipping is applied: the configured endpoint itself is the maximum beta.
    All other M13 task and sampling semantics are inherited unchanged.
    """

    def __init__(
        self,
        *args: object,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        if not 0.0 < self.beta_start <= self.beta_end < 1.0:
            raise ValueError("linear beta schedule requires 0 < beta_start <= beta_end < 1")

        self._alpha_bar_cpu = _linear_discrete_alpha_bar(
            self.timesteps,
            self.beta_start,
            self.beta_end,
        )
        self._alpha_bar_cache.clear()

    @property
    def noise_schedule(self) -> str:
        return "linear_beta"


def _linear_discrete_alpha_bar(
    timesteps: int,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return torch.cat((torch.ones(1, dtype=torch.float64), alpha_bar), dim=0)
