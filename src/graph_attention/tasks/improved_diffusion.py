"""Improved-DDPM diffusion task with learned reverse variance.

This module adapts the learned-range variance and hybrid objective from
Nichol & Dhariwal (2021) to the repository's packed, equal-sample CFD task
semantics.  It intentionally keeps the existing cosine forward process and
normalized scalar time conditioning unchanged so the first comparison isolates
only the diffusion formulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from operator import index as operator_index

import torch
from torch import nn

from .diffusion import (
    DiffusionDenoisingTask,
    _randn_by_graph,
    _sample_generator,
    _time_conditioning,
    _validate_state_batch,
)
from .regression import NodeRegressionBatch


@dataclass(frozen=True, slots=True)
class ImprovedDiffusionProblem:
    """One noised state plus the latent variables needed by the hybrid loss."""

    model_batch: NodeRegressionBatch
    clean_state: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor

    @property
    def num_graphs(self) -> int:
        return self.model_batch.num_graphs


@dataclass(frozen=True, slots=True)
class ImprovedDiffusionLoss:
    """Per-sample components of the Improved-DDPM hybrid objective."""

    hybrid_per_sample: torch.Tensor
    simple_per_sample: torch.Tensor
    vlb_per_sample: torch.Tensor

    @property
    def loss_sum(self) -> torch.Tensor:
        return self.hybrid_per_sample.sum()

    @property
    def mean(self) -> torch.Tensor:
        return self.hybrid_per_sample.mean()

    @property
    def sample_count(self) -> int:
        return int(self.hybrid_per_sample.numel())


class LossSecondMomentTimestepSampler:
    """Loss-aware timestep sampler from Improved DDPM.

    Until every timestep has ``history_per_timestep`` observations, sampling is
    uniform.  Afterwards the probability of timestep ``t`` is proportional to
    ``sqrt(E[L_t^2])`` with a small uniform mixture.  Returned importance
    weights are ``1 / (T p_t)``, preserving the uniform-timestep objective in
    expectation.
    """

    def __init__(
        self,
        timesteps: int,
        *,
        history_per_timestep: int = 10,
        uniform_probability: float = 0.001,
    ) -> None:
        self.timesteps = _positive_int(timesteps, "timesteps")
        self.history_per_timestep = _positive_int(
            history_per_timestep,
            "history_per_timestep",
        )
        self.uniform_probability = float(uniform_probability)
        if not 0.0 <= self.uniform_probability < 1.0:
            raise ValueError("uniform_probability must lie in [0, 1)")
        self._history = torch.zeros(
            (self.timesteps, self.history_per_timestep),
            dtype=torch.float64,
        )
        self._counts = torch.zeros(self.timesteps, dtype=torch.long)

    @property
    def warmed_up(self) -> bool:
        return bool(torch.all(self._counts >= self.history_per_timestep))

    def probabilities(self) -> torch.Tensor:
        if not self.warmed_up:
            return torch.full(
                (self.timesteps,),
                1.0 / self.timesteps,
                dtype=torch.float64,
            )
        weights = torch.sqrt(torch.mean(self._history.square(), dim=1))
        total = weights.sum()
        if not torch.isfinite(total) or float(total) <= 0.0:
            raise ValueError("loss-aware timestep weights are not finite and positive")
        probabilities = weights / total
        probabilities = probabilities * (1.0 - self.uniform_probability)
        probabilities = probabilities + self.uniform_probability / self.timesteps
        return probabilities / probabilities.sum()

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = _positive_int(batch_size, "batch_size")
        probabilities = self.probabilities()
        zero_based = torch.multinomial(
            probabilities,
            count,
            replacement=True,
            generator=generator,
        )
        importance = 1.0 / (self.timesteps * probabilities[zero_based])
        return (
            (zero_based + 1).to(device=device, dtype=torch.long),
            importance.to(device=device, dtype=torch.float32),
        )

    def update(self, timesteps: torch.Tensor, losses: torch.Tensor) -> None:
        if timesteps.ndim != 1 or losses.ndim != 1 or timesteps.shape != losses.shape:
            raise ValueError("timesteps and losses must share one-dimensional shape [B]")
        if timesteps.dtype != torch.long:
            raise TypeError("timesteps must use torch.long")
        timestep_values = timesteps.detach().cpu().tolist()
        loss_values = losses.detach().to(torch.float64).cpu()
        if not torch.isfinite(loss_values).all():
            raise ValueError("loss-aware timestep sampler received NaN or Inf losses")
        for timestep, loss in zip(timestep_values, loss_values.tolist(), strict=True):
            index = int(timestep) - 1
            if index < 0 or index >= self.timesteps:
                raise ValueError(f"timestep must lie in [1, {self.timesteps}]")
            observed = int(self._counts[index])
            if observed < self.history_per_timestep:
                self._history[index, observed] = float(loss)
                self._counts[index] += 1
            else:
                self._history[index, :-1] = self._history[index, 1:].clone()
                self._history[index, -1] = float(loss)


class ImprovedDiffusionDenoisingTask(DiffusionDenoisingTask):
    """Improved DDPM with epsilon prediction and learned-range variance.

    The model outputs ``2*C`` channels.  The first ``C`` predict epsilon.  The
    remaining ``C`` predict the learned-range variance variable ``v`` used to
    interpolate in log-variance space between the posterior variance and beta.

    The hybrid objective is

        L = L_simple + vlb_weight * L_vlb

    with the epsilon prediction detached from the VLB branch, matching the
    Improved-DDPM training strategy used by DGN4CFD.
    """

    def __init__(
        self,
        *args: object,
        vlb_weight: float = 0.001,
        importance_history_per_timestep: int = 10,
        importance_uniform_probability: float = 0.001,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.timesteps < 2:
            raise ValueError("improved diffusion requires at least two timesteps")
        self.vlb_weight = float(vlb_weight)
        if self.vlb_weight < 0.0:
            raise ValueError("vlb_weight must be non-negative")
        self.importance_history_per_timestep = _positive_int(
            importance_history_per_timestep,
            "importance_history_per_timestep",
        )
        self.importance_uniform_probability = float(importance_uniform_probability)
        if not 0.0 <= self.importance_uniform_probability < 1.0:
            raise ValueError("importance_uniform_probability must lie in [0, 1)")

        alpha_bar = self._alpha_bar_cpu
        betas = torch.zeros_like(alpha_bar)
        betas[1:] = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        alphas = 1.0 - betas

        posterior_variance = torch.zeros_like(alpha_bar)
        posterior_variance[1:] = (
            betas[1:]
            * (1.0 - alpha_bar[:-1])
            / (1.0 - alpha_bar[1:])
        )
        clipped_variance = posterior_variance.clone()
        clipped_variance[1] = posterior_variance[2]
        clipped_variance[0] = clipped_variance[1]
        posterior_log_variance_clipped = torch.log(clipped_variance.clamp_min(1.0e-30))

        posterior_mean_coef1 = torch.zeros_like(alpha_bar)
        posterior_mean_coef2 = torch.zeros_like(alpha_bar)
        posterior_mean_coef1[1:] = (
            betas[1:]
            * torch.sqrt(alpha_bar[:-1])
            / (1.0 - alpha_bar[1:])
        )
        posterior_mean_coef2[1:] = (
            (1.0 - alpha_bar[:-1])
            * torch.sqrt(alphas[1:])
            / (1.0 - alpha_bar[1:])
        )

        self._improved_schedule_cpu = {
            "betas": betas,
            "alphas": alphas,
            "sqrt_recip_alphas": torch.where(
                alphas > 0.0,
                torch.rsqrt(alphas),
                torch.zeros_like(alphas),
            ),
            "posterior_variance": posterior_variance,
            "posterior_log_variance_clipped": posterior_log_variance_clipped,
            "posterior_mean_coef1": posterior_mean_coef1,
            "posterior_mean_coef2": posterior_mean_coef2,
        }
        self._improved_schedule_cache: dict[
            tuple[str, torch.dtype], dict[str, torch.Tensor]
        ] = {}

    @property
    def prediction_type(self) -> str:
        return "epsilon_and_learned_range_variance"

    @property
    def learned_reverse_variance(self) -> bool:
        return True

    def make_timestep_sampler(self) -> LossSecondMomentTimestepSampler:
        return LossSecondMomentTimestepSampler(
            self.timesteps,
            history_per_timestep=self.importance_history_per_timestep,
            uniform_probability=self.importance_uniform_probability,
        )

    def make_training_problem(
        self,
        batch: NodeRegressionBatch,
        *,
        timesteps: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> ImprovedDiffusionProblem:
        """Construct one noising problem at externally sampled graph timesteps."""

        _validate_state_batch(batch)
        _validate_graph_timesteps(timesteps, batch, self.timesteps)
        noise = torch.randn(
            batch.inputs.shape,
            device=batch.inputs.device,
            dtype=batch.inputs.dtype,
            generator=generator,
        )
        model_batch = self._diffusion_problem(batch, timesteps, noise)
        return ImprovedDiffusionProblem(
            model_batch=model_batch,
            clean_state=batch.inputs,
            noise=noise,
            timesteps=timesteps,
        )

    def make_validation_problem(self, batch: NodeRegressionBatch) -> ImprovedDiffusionProblem:
        """Reuse the baseline deterministic sample-ID validation noising."""

        model_batch = super().make_validation_problem(batch)
        timesteps = torch.round(
            model_batch.conditioning[:, -1] * float(self.timesteps)
        ).to(dtype=torch.long)
        _validate_graph_timesteps(timesteps, batch, self.timesteps)
        return ImprovedDiffusionProblem(
            model_batch=model_batch,
            clean_state=batch.inputs,
            noise=model_batch.targets,
            timesteps=timesteps,
        )

    def make_model_probe(self, problem: ImprovedDiffusionProblem) -> NodeRegressionBatch:
        """Return a shape-only probe whose target width equals the model output width."""

        model_batch = problem.model_batch
        variance_channels = tuple(
            f"variance_range:{name}" for name in model_batch.input_channels
        )
        return replace(
            model_batch,
            targets=torch.cat((problem.noise, torch.zeros_like(problem.noise)), dim=1),
            target_channels=model_batch.target_channels + variance_channels,
        )

    def hybrid_loss(
        self,
        model_output: torch.Tensor,
        problem: ImprovedDiffusionProblem,
    ) -> ImprovedDiffusionLoss:
        """Return equal-sample hybrid, simple-MSE, and VLB losses."""

        batch = problem.model_batch
        channels = problem.noise.shape[1]
        _validate_model_output(model_output, problem.noise, channels)
        epsilon_hat, variance_values = torch.split(model_output, channels, dim=1)

        simple_node = (epsilon_hat - problem.noise).square().mean(dim=1)
        simple_per_sample = _reduce_node_values(simple_node, batch)

        schedule = self._improved_schedule_for(problem.noise)
        node_timesteps = problem.timesteps[batch.batch_index]
        betas = schedule["betas"][node_timesteps].unsqueeze(1)
        alpha_bar = self._alpha_bar_for(problem.noise)[node_timesteps].unsqueeze(1)
        sqrt_recip_alphas = schedule["sqrt_recip_alphas"][node_timesteps].unsqueeze(1)

        true_mean = (
            schedule["posterior_mean_coef1"][node_timesteps].unsqueeze(1)
            * problem.clean_state
            + schedule["posterior_mean_coef2"][node_timesteps].unsqueeze(1)
            * batch.inputs
        )
        true_variance = schedule["posterior_variance"][node_timesteps].unsqueeze(1)

        model_mean = sqrt_recip_alphas * (
            batch.inputs
            - betas
            * epsilon_hat.detach()
            / torch.sqrt((1.0 - alpha_bar).clamp_min(torch.finfo(batch.inputs.dtype).tiny))
        )
        min_log = schedule["posterior_log_variance_clipped"][node_timesteps].unsqueeze(1)
        max_log = torch.log(betas.clamp_min(torch.finfo(batch.inputs.dtype).tiny))
        fraction = (variance_values + 1.0) / 2.0
        model_log_variance = fraction * max_log + (1.0 - fraction) * min_log
        model_variance = torch.exp(model_log_variance)

        safe_true_variance = true_variance.clamp_min(torch.finfo(batch.inputs.dtype).tiny)
        kl = 0.5 * (
            model_log_variance
            - torch.log(safe_true_variance)
            + safe_true_variance / model_variance
            + (true_mean - model_mean).square() / model_variance
            - 1.0
        )
        kl_bits = kl / math.log(2.0)

        decoder_nll = 0.5 * (
            model_log_variance
            + (problem.clean_state - model_mean).square() / model_variance
        )
        first_step = node_timesteps.unsqueeze(1) == 1
        vlb_element = torch.where(first_step, decoder_nll, kl_bits)
        vlb_node = vlb_element.mean(dim=1)
        vlb_per_sample = _reduce_node_values(vlb_node, batch)

        hybrid_per_sample = simple_per_sample + self.vlb_weight * vlb_per_sample
        if not torch.isfinite(hybrid_per_sample).all():
            raise ValueError("improved diffusion hybrid loss contains NaN or Inf")
        return ImprovedDiffusionLoss(
            hybrid_per_sample=hybrid_per_sample,
            simple_per_sample=simple_per_sample,
            vlb_per_sample=vlb_per_sample,
        )

    @torch.no_grad()
    def sample_standardized(
        self,
        model: nn.Module,
        batch: NodeRegressionBatch,
        *,
        steps: int,
        eta: float = 1.0,
        sampling_seed: int = 5678,
        sampling_keys: tuple[str, ...] | list[str] | None = None,
    ) -> torch.Tensor:
        """Sample the full learned-variance ancestral Improved-DDPM chain."""

        _validate_state_batch(batch)
        if _positive_int(steps, "steps") != self.timesteps or float(eta) != 1.0:
            raise ValueError(
                "the first Improved-DDPM implementation supports only the exact full "
                f"learned-variance ancestral chain: steps={self.timesteps}, eta=1"
            )
        seed = _nonnegative_int(sampling_seed, "sampling_seed")
        keys = tuple(batch.source.sample_ids) if sampling_keys is None else tuple(sampling_keys)
        if len(keys) != batch.num_graphs:
            raise ValueError("sampling_keys must contain one key per graph")
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("sampling_keys must contain non-empty strings")

        generators = [
            _sample_generator(batch.inputs.device, seed, "sampling", key)
            for key in keys
        ]
        state = _randn_by_graph(batch, generators)
        schedule = self._improved_schedule_for(state)
        alpha_bar = self._alpha_bar_for(state)

        for timestep in range(self.timesteps, 0, -1):
            graph_timesteps = torch.full(
                (batch.num_graphs,),
                timestep,
                device=state.device,
                dtype=torch.long,
            )
            output = self._model_output(
                model,
                batch,
                state,
                self._normalized_time(graph_timesteps, state.dtype),
            )
            channels = state.shape[1]
            epsilon_hat, variance_values = torch.split(output, channels, dim=1)

            beta_t = schedule["betas"][timestep]
            alpha_t = schedule["alphas"][timestep]
            alpha_bar_t = alpha_bar[timestep]
            model_mean = torch.rsqrt(alpha_t) * (
                state
                - beta_t
                * epsilon_hat
                / torch.sqrt((1.0 - alpha_bar_t).clamp_min(torch.finfo(state.dtype).tiny))
            )

            min_log = schedule["posterior_log_variance_clipped"][timestep]
            max_log = torch.log(beta_t.clamp_min(torch.finfo(state.dtype).tiny))
            fraction = (variance_values + 1.0) / 2.0
            model_log_variance = fraction * max_log + (1.0 - fraction) * min_log

            if timestep == 1:
                state = model_mean
            else:
                state = model_mean + torch.exp(0.5 * model_log_variance) * _randn_by_graph(
                    batch,
                    generators,
                )

        if not torch.isfinite(state).all():
            raise ValueError("improved diffusion sampler produced NaN or Inf values")
        return state

    def sampler_name(self, *, steps: int, eta: float) -> str:
        if _positive_int(steps, "steps") == self.timesteps and float(eta) == 1.0:
            return "improved_ddpm_ancestral_learned_variance"
        return "unsupported_improved_diffusion_sampler"

    def _model_output(
        self,
        model: nn.Module,
        batch: NodeRegressionBatch,
        state: torch.Tensor,
        normalized_times: torch.Tensor,
    ) -> torch.Tensor:
        conditioning = _time_conditioning(batch, normalized_times)
        model_kwargs = {
            "edge_index": batch.edge_index,
            "coords": batch.coords,
            "batch_index": batch.batch_index,
            "conditioning": conditioning,
        }
        if batch.attention_edge_indices:
            model_kwargs["attention_edge_indices"] = batch.attention_edge_indices
        output = model(state, **model_kwargs)
        expected = (state.shape[0], 2 * state.shape[1])
        if output.shape != expected:
            raise ValueError(
                "Improved-DDPM model output must have shape [N, 2*C]: "
                f"got {tuple(output.shape)}, expected {expected}"
            )
        return output

    def _improved_schedule_for(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        key = (str(reference.device), reference.dtype)
        cached = self._improved_schedule_cache.get(key)
        if cached is None:
            cached = {
                name: value.to(device=reference.device, dtype=reference.dtype)
                for name, value in self._improved_schedule_cpu.items()
            }
            self._improved_schedule_cache[key] = cached
        return cached


def _reduce_node_values(values: torch.Tensor, batch: NodeRegressionBatch) -> torch.Tensor:
    if values.ndim != 1 or values.shape[0] != batch.inputs.shape[0]:
        raise ValueError("node loss values must have shape [total_nodes]")
    weights = batch.node_weights
    per_sample: list[torch.Tensor] = []
    for graph_index in range(batch.num_graphs):
        start = int(batch.ptr[graph_index])
        stop = int(batch.ptr[graph_index + 1])
        graph_values = values[start:stop]
        if weights is None:
            per_sample.append(graph_values.mean())
            continue
        graph_weights = weights[start:stop]
        denominator = graph_weights.sum()
        if denominator <= 0:
            raise ValueError(f"node_weights sum to zero for graph {graph_index}")
        per_sample.append((graph_values * graph_weights).sum() / denominator)
    result = torch.stack(per_sample)
    if not torch.isfinite(result).all():
        raise ValueError("per-sample improved diffusion loss contains NaN or Inf")
    return result


def _validate_graph_timesteps(
    timesteps: torch.Tensor,
    batch: NodeRegressionBatch,
    total_timesteps: int,
) -> None:
    if timesteps.shape != (batch.num_graphs,) or timesteps.dtype != torch.long:
        raise ValueError(f"timesteps must have long shape [{batch.num_graphs}]")
    if timesteps.device != batch.inputs.device:
        raise ValueError("timesteps must be on the batch input device")
    if bool(torch.any(timesteps < 1)) or bool(torch.any(timesteps > total_timesteps)):
        raise ValueError(f"timesteps must lie in [1, {total_timesteps}]")


def _validate_model_output(
    model_output: torch.Tensor,
    noise: torch.Tensor,
    channels: int,
) -> None:
    if model_output.ndim != 2 or model_output.shape != (noise.shape[0], 2 * channels):
        raise ValueError(
            "Improved-DDPM model output must have shape [N, 2*C]: "
            f"got {tuple(model_output.shape)}, expected {(noise.shape[0], 2 * channels)}"
        )
    if not model_output.is_floating_point():
        raise TypeError("Improved-DDPM model output must be floating-point")
    if model_output.device != noise.device:
        raise ValueError("Improved-DDPM model output and noise must share a device")


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


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result
