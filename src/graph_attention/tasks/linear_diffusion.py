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

    def continuous_marginal_coefficients(
        self,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed the discrete DDPM schedule in continuous normalized time.

        log(alpha_bar(t)) is linearly interpolated between exact DDPM grid
        values, so every trained marginal is recovered at t=i/T.
        """

        _validate_continuous_times(times)
        schedule = self._alpha_bar_cpu.to(device=times.device, dtype=times.dtype)
        scaled = times * float(self.timesteps)
        interval = torch.ceil(scaled).to(dtype=torch.long).clamp(1, self.timesteps)
        fraction = scaled - (interval - 1).to(dtype=times.dtype)

        log_previous = torch.log(schedule[interval - 1])
        log_ratio = torch.log(schedule[interval] / schedule[interval - 1])
        log_alpha_bar = log_previous + fraction * log_ratio
        alpha_bar = torch.exp(log_alpha_bar)
        alpha = torch.sqrt(alpha_bar)
        sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0))
        return alpha, sigma

    def continuous_beta_rate(self, times: torch.Tensor) -> torch.Tensor:
        """Return the continuous VP beta rate implied by the DDPM schedule."""

        _validate_continuous_times(times)
        schedule = self._alpha_bar_cpu.to(device=times.device, dtype=times.dtype)
        scaled = times * float(self.timesteps)
        interval = torch.ceil(scaled).to(dtype=torch.long).clamp(1, self.timesteps)
        log_ratio = torch.log(schedule[interval] / schedule[interval - 1])
        return -float(self.timesteps) * log_ratio

    @torch.no_grad()
    def reverse_sde_drift(
        self,
        model: torch.nn.Module,
        batch: object,
        state: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Return the reverse VP-SDE drift induced by the trained DDPM schedule."""

        from .diffusion import _model_epsilon

        if state.ndim != 2 or state.shape != batch.inputs.shape:
            raise ValueError("reverse-SDE state must match batch input shape")
        if times.shape != (batch.num_graphs,):
            raise ValueError(f"times must have shape [{batch.num_graphs}]")
        if bool(torch.any(times <= 0.0)) or bool(torch.any(times > 1.0)):
            raise ValueError("reverse-SDE times must lie in (0, 1]")

        epsilon_hat = _model_epsilon(model, batch, state, times)
        _, sigma = self.continuous_marginal_coefficients(times)
        if bool(torch.any(sigma <= 0.0)):
            raise ValueError("reverse-SDE score requires strictly positive sigma")
        beta = self.continuous_beta_rate(times)
        node_beta = beta[batch.batch_index].unsqueeze(1)
        node_sigma = sigma[batch.batch_index].unsqueeze(1)
        return -0.5 * node_beta * state + node_beta * epsilon_hat / node_sigma

    @torch.no_grad()
    def sample_standardized(
        self,
        model: torch.nn.Module,
        batch: object,
        *,
        steps: int,
        eta: float | None = None,
        sampling_seed: int = 5678,
        sampling_keys: object = None,
        start_timestep: int | None = None,
        method: str | None = None,
        solver: str | None = None,
        sampling_eps: float | None = None,
        final_denoise: bool | None = None,
    ) -> torch.Tensor:
        """Sample the legacy discrete chain or the matched continuous reverse SDE."""

        if method is None:
            return super().sample_standardized(
                model,
                batch,
                steps=steps,
                eta=0.0 if eta is None else eta,
                sampling_seed=sampling_seed,
                sampling_keys=sampling_keys,
                start_timestep=start_timestep,
            )

        if method != "reverse_sde":
            raise ValueError("linear-beta DDPM generic sampling supports reverse_sde only")
        if solver != "euler_maruyama":
            raise ValueError("linear-beta DDPM reverse SDE requires euler_maruyama")
        if eta is not None or start_timestep is not None:
            raise ValueError("eta/start_timestep are legacy discrete-sampler options")
        if sampling_eps is None:
            sampling_eps = 1.0 / float(self.timesteps)
        endpoint = float(sampling_eps)
        if not torch.isfinite(torch.tensor(endpoint)) or not 0.0 < endpoint < 1.0:
            raise ValueError("sampling_eps must lie strictly between 0 and 1")
        if endpoint < 1.0 / float(self.timesteps):
            raise ValueError("sampling_eps must not lie below the first trained DDPM time")
        if final_denoise is None:
            final_denoise = True
        if not isinstance(final_denoise, bool):
            raise TypeError("final_denoise must be boolean")

        from .diffusion import (
            _model_epsilon,
            _node_counts,
            _nonnegative_int,
            _positive_int,
            _sample_generator,
        )

        step_count = _positive_int(steps, "steps")
        seed = _nonnegative_int(sampling_seed, "sampling_seed")
        keys = tuple(batch.source.sample_ids) if sampling_keys is None else tuple(sampling_keys)
        if len(keys) != batch.num_graphs:
            raise ValueError("sampling_keys must contain one key per graph")
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("sampling_keys must contain non-empty strings")

        generators = [
            _sample_generator(batch.inputs.device, seed, "ddpm_reverse_sde_sampling", key)
            for key in keys
        ]
        node_counts = _node_counts(batch)
        state = torch.cat(
            [
                torch.randn(
                    (node_count, batch.inputs.shape[1]),
                    device=batch.inputs.device,
                    dtype=batch.inputs.dtype,
                    generator=generator,
                )
                for node_count, generator in zip(node_counts, generators, strict=True)
            ],
            dim=0,
        )
        times = torch.linspace(
            1.0,
            endpoint,
            steps=step_count + 1,
            device=batch.inputs.device,
            dtype=batch.inputs.dtype,
        )

        for index in range(step_count):
            current = times[index]
            following = times[index + 1]
            dt = following - current
            graph_time = current.expand(batch.num_graphs)
            drift = self.reverse_sde_drift(model, batch, state, graph_time)
            beta = self.continuous_beta_rate(graph_time)
            node_diffusion = torch.sqrt(beta)[batch.batch_index].unsqueeze(1)
            noise = torch.cat(
                [
                    torch.randn(
                        (node_count, batch.inputs.shape[1]),
                        device=batch.inputs.device,
                        dtype=batch.inputs.dtype,
                        generator=generator,
                    )
                    for node_count, generator in zip(node_counts, generators, strict=True)
                ],
                dim=0,
            )
            state = state + dt * drift + torch.sqrt(-dt) * node_diffusion * noise

        if final_denoise:
            endpoint_times = times[-1].expand(batch.num_graphs)
            epsilon_hat = _model_epsilon(model, batch, state, endpoint_times)
            alpha, sigma = self.continuous_marginal_coefficients(endpoint_times)
            node_alpha = alpha[batch.batch_index].unsqueeze(1)
            node_sigma = sigma[batch.batch_index].unsqueeze(1)
            state = (state - node_sigma * epsilon_hat) / node_alpha

        if not torch.isfinite(state).all():
            raise ValueError("linear-beta DDPM reverse-SDE sampler produced NaN or Inf values")
        return state

    def sampler_name(
        self,
        *,
        steps: int,
        eta: float | None = None,
        start_timestep: int | None = None,
        method: str | None = None,
        solver: str | None = None,
        final_denoise: bool | None = None,
    ) -> str:
        """Name the legacy DDPM/DDIM sampler or the continuous reverse SDE."""

        if method is None:
            return super().sampler_name(
                steps=steps,
                eta=0.0 if eta is None else eta,
                start_timestep=start_timestep,
            )
        if method != "reverse_sde" or solver != "euler_maruyama":
            raise ValueError("invalid linear-beta DDPM reverse-SDE method/solver")
        if final_denoise is None:
            final_denoise = True
        base = f"linear_ddpm_reverse_sde_euler_maruyama_steps{int(steps)}"
        return f"{base}_denoise" if final_denoise else base


def _linear_discrete_alpha_bar(
    timesteps: int,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return torch.cat((torch.ones(1, dtype=torch.float64), alpha_bar), dim=0)


def _validate_continuous_times(times: torch.Tensor) -> None:
    if not isinstance(times, torch.Tensor) or times.ndim != 1:
        raise ValueError("continuous DDPM times must be a one-dimensional tensor")
    if not times.is_floating_point():
        raise TypeError("continuous DDPM times must be floating-point")
    if not torch.isfinite(times).all():
        raise ValueError("continuous DDPM times must be finite")
    if bool(torch.any(times < 0.0)) or bool(torch.any(times > 1.0)):
        raise ValueError("continuous DDPM times must lie in [0, 1]")
