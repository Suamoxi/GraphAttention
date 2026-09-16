"""Continuous variance-preserving SDE task for packed CFD graph fields."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

import torch
from torch import nn

from graph_attention.objectives import SampleLossAggregate, sample_reduced_mse

from .diffusion import (
    _model_epsilon,
    _node_counts,
    _nonnegative_int,
    _positive_int,
    _randn_by_graph,
    _sample_generator,
    _time_conditioning,
    _validate_state_batch,
)
from .regression import NodeRegressionBatch, NodeRegressionTask

_VP_TIME_NAME = "vp_sde_time"


class VPSDEDenoisingTask(NodeRegressionTask):
    """Continuous VP-SDE epsilon-prediction objective and reverse samplers.

    The forward process is

        dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW,

    with ``t in [0, 1]`` and a linear continuous beta schedule. Its analytic
    marginal is

        x_t = alpha(t) * x_0 + sigma(t) * epsilon,

    so training samples the marginal directly rather than numerically simulating
    the forward SDE. The network predicts ``epsilon``; the corresponding score is

        score(x_t, t) = -epsilon_theta(x_t, t) / sigma(t).

    Sampling exposes both the reverse-time SDE and the associated deterministic
    probability-flow ODE while keeping the graph backbone unchanged.
    """

    training_metric_name = "epsilon_mse"
    training_seed_name = "vp_sde_noise_seed"

    def __init__(
        self,
        state_fields: Iterable[str],
        conditioning_parameters: Iterable[str] = (),
        physical_nondimensionalization: bool = False,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        training_eps: float = 1.0e-5,
        validation_seed: int = 1234,
    ) -> None:
        if isinstance(state_fields, str):
            raise TypeError("state_fields must be an iterable of field names, not one string")
        fields = tuple(state_fields)
        super().__init__(
            input_fields=fields,
            target_fields=fields,
            conditioning_parameters=conditioning_parameters,
            physical_nondimensionalization=physical_nondimensionalization,
        )
        self.state_fields = self.input_fields
        self.beta_min = _positive_float(beta_min, "beta_min")
        self.beta_max = _positive_float(beta_max, "beta_max")
        if self.beta_max < self.beta_min:
            raise ValueError("beta_max must be greater than or equal to beta_min")
        self.training_eps = _open_unit_float(training_eps, "training_eps")
        self.validation_seed = _nonnegative_int(validation_seed, "validation_seed")

    def beta(self, times: torch.Tensor) -> torch.Tensor:
        """Return the continuous linear VP noise rate beta(t)."""

        _validate_times_tensor(times)
        return self.beta_min + times * (self.beta_max - self.beta_min)

    def marginal_coefficients(
        self,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``alpha(t)`` and ``sigma(t)`` for the analytic VP marginal."""

        _validate_times_tensor(times)
        log_alpha = (
            -0.25 * (self.beta_max - self.beta_min) * times.square()
            - 0.5 * self.beta_min * times
        )
        alpha = torch.exp(log_alpha)
        sigma_sq = torch.clamp(-torch.expm1(2.0 * log_alpha), min=0.0)
        sigma = torch.sqrt(sigma_sq)
        return alpha, sigma

    def make_training_problem(
        self,
        batch: NodeRegressionBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> NodeRegressionBatch:
        """Sample one continuous VP marginal perturbation per physical graph."""

        _validate_vp_batch(batch)
        times = torch.rand(
            (batch.num_graphs,),
            device=batch.inputs.device,
            dtype=batch.inputs.dtype,
            generator=generator,
        )
        times = self.training_eps + (1.0 - self.training_eps) * times
        noise = torch.randn(
            batch.inputs.shape,
            device=batch.inputs.device,
            dtype=batch.inputs.dtype,
            generator=generator,
        )
        return self._vp_problem(batch, times, noise)

    def make_validation_problem(self, batch: NodeRegressionBatch) -> NodeRegressionBatch:
        """Construct deterministic VP time/noise pairs keyed by sample ID."""

        _validate_vp_batch(batch)
        times: list[torch.Tensor] = []
        noise_parts: list[torch.Tensor] = []
        for node_count, sample_id in zip(
            _node_counts(batch),
            batch.source.sample_ids,
            strict=True,
        ):
            generator = _sample_generator(
                batch.inputs.device,
                self.validation_seed,
                "vp_sde_validation",
                sample_id,
            )
            sampled_time = torch.rand(
                (),
                device=batch.inputs.device,
                dtype=batch.inputs.dtype,
                generator=generator,
            )
            times.append(self.training_eps + (1.0 - self.training_eps) * sampled_time)
            noise_parts.append(
                torch.randn(
                    (node_count, batch.inputs.shape[1]),
                    device=batch.inputs.device,
                    dtype=batch.inputs.dtype,
                    generator=generator,
                )
            )
        return self._vp_problem(
            batch,
            torch.stack(times),
            torch.cat(noise_parts, dim=0),
        )

    def make_model_probe(self, problem: NodeRegressionBatch) -> NodeRegressionBatch:
        """Return the VP perturbation as the model-facing batch."""

        if not isinstance(problem, NodeRegressionBatch):
            raise TypeError("VP-SDE problem must be a NodeRegressionBatch")
        return problem

    def training_loss(
        self,
        model_output: torch.Tensor,
        problem: NodeRegressionBatch,
    ) -> SampleLossAggregate:
        """Return equal-sample epsilon-prediction MSE."""

        if not isinstance(problem, NodeRegressionBatch):
            raise TypeError("VP-SDE problem must be a NodeRegressionBatch")
        return sample_reduced_mse(
            model_output,
            problem.targets,
            problem.ptr,
            node_weights=problem.node_weights,
        )

    def training_summary_metadata(self) -> dict[str, object]:
        """Describe the continuous VP formulation recorded by the generic runner."""

        return {
            "task": "continuous_vp_sde_epsilon_prediction",
            "prediction_type": "epsilon",
            "continuous_time_domain": [self.training_eps, 1.0],
            "beta_schedule": "beta(t)=beta_min+t*(beta_max-beta_min)",
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "forward_sde": "dx=-0.5*beta(t)*x*dt+sqrt(beta(t))*dW",
            "forward_marginal": "x_t=alpha(t)*x0+sigma(t)*epsilon",
            "score_parameterization": "score_theta=-epsilon_theta/sigma(t)",
            "time_conditioning": "continuous_vp_time_t_in_[eps,1]",
            "loss": "equal_sample_MSE(epsilon_theta,epsilon)",
            "reverse_sde": "dx=[f-g^2*score_theta]*dt+g*dW_reverse",
            "probability_flow_ode": "dx/dt=f-0.5*g^2*score_theta",
            "generation": "task.sample_standardized",
        }

    def score_from_epsilon(
        self,
        epsilon_hat: torch.Tensor,
        batch: NodeRegressionBatch,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Convert epsilon prediction to the VP score at graph times."""

        _validate_graph_times(times, batch.num_graphs, lower_bound=0.0)
        if epsilon_hat.shape != batch.inputs.shape:
            raise ValueError("epsilon prediction must match the packed state shape")
        _, sigma = self.marginal_coefficients(times)
        if bool(torch.any(sigma <= 0.0)):
            raise ValueError("VP score is singular at t=0; use strictly positive times")
        node_sigma = sigma[batch.batch_index].unsqueeze(1)
        return -epsilon_hat / node_sigma

    @torch.no_grad()
    def probability_flow_drift(
        self,
        model: nn.Module,
        batch: NodeRegressionBatch,
        state: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Return the VP probability-flow ODE drift at one time per graph."""

        _validate_sampling_state(batch, state)
        _validate_graph_times(times, batch.num_graphs, lower_bound=0.0)
        epsilon_hat = _model_epsilon(model, batch, state, times)
        _, sigma = self.marginal_coefficients(times)
        if bool(torch.any(sigma <= 0.0)):
            raise ValueError("probability-flow drift requires strictly positive times")
        beta = self.beta(times)
        node_beta = beta[batch.batch_index].unsqueeze(1)
        node_sigma = sigma[batch.batch_index].unsqueeze(1)
        return -0.5 * node_beta * state + 0.5 * node_beta * epsilon_hat / node_sigma

    @torch.no_grad()
    def reverse_sde_drift(
        self,
        model: nn.Module,
        batch: NodeRegressionBatch,
        state: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Return the reverse-time VP-SDE drift at one time per graph."""

        _validate_sampling_state(batch, state)
        _validate_graph_times(times, batch.num_graphs, lower_bound=0.0)
        epsilon_hat = _model_epsilon(model, batch, state, times)
        _, sigma = self.marginal_coefficients(times)
        if bool(torch.any(sigma <= 0.0)):
            raise ValueError("reverse-SDE drift requires strictly positive times")
        beta = self.beta(times)
        node_beta = beta[batch.batch_index].unsqueeze(1)
        node_sigma = sigma[batch.batch_index].unsqueeze(1)
        return -0.5 * node_beta * state + node_beta * epsilon_hat / node_sigma

    @torch.no_grad()
    def sample_standardized(
        self,
        model: nn.Module,
        batch: NodeRegressionBatch,
        *,
        steps: int = 1000,
        method: str = "probability_flow_ode",
        solver: str = "heun",
        sampling_eps: float = 1.0e-3,
        final_denoise: bool = True,
        sampling_seed: int = 5678,
        sampling_keys: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Generate standardized states with the VP reverse SDE or PF-ODE.

        ``steps`` is the number of numerical transitions from ``t=1`` to
        ``sampling_eps``. The probability-flow ODE supports Euler or Heun.
        Reverse-SDE integration uses Euler-Maruyama. ``final_denoise`` applies the
        epsilon-based clean-state estimate at the positive endpoint; it is not an
        additional ODE/SDE integration step.
        """

        _validate_vp_batch(batch)
        step_count = _positive_int(steps, "steps")
        endpoint = _open_unit_float(sampling_eps, "sampling_eps")
        if endpoint < self.training_eps:
            raise ValueError("sampling_eps must be greater than or equal to training_eps")
        if method not in {"probability_flow_ode", "reverse_sde"}:
            raise ValueError("method must be 'probability_flow_ode' or 'reverse_sde'")
        if method == "probability_flow_ode":
            if solver not in {"euler", "heun"}:
                raise ValueError("probability-flow ODE solver must be 'euler' or 'heun'")
        elif solver != "euler_maruyama":
            raise ValueError("reverse-SDE solver must be 'euler_maruyama'")
        if not isinstance(final_denoise, bool):
            raise TypeError("final_denoise must be boolean")

        seed = _nonnegative_int(sampling_seed, "sampling_seed")
        keys = (
            tuple(batch.source.sample_ids)
            if sampling_keys is None
            else tuple(sampling_keys)
        )
        if len(keys) != batch.num_graphs:
            raise ValueError("sampling_keys must contain one key per graph")
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("sampling_keys must contain non-empty strings")

        purpose = (
            "vp_pf_ode_sampling"
            if method == "probability_flow_ode"
            else "vp_sde_sampling"
        )
        generators = [
            _sample_generator(batch.inputs.device, seed, purpose, key)
            for key in keys
        ]
        state = _randn_by_graph(batch, generators)
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

            if method == "probability_flow_ode":
                drift = self.probability_flow_drift(model, batch, state, graph_time)
                euler_state = state + dt * drift
                if solver == "heun":
                    next_time = following.expand(batch.num_graphs)
                    next_drift = self.probability_flow_drift(
                        model,
                        batch,
                        euler_state,
                        next_time,
                    )
                    state = state + 0.5 * dt * (drift + next_drift)
                else:
                    state = euler_state
                continue

            drift = self.reverse_sde_drift(model, batch, state, graph_time)
            beta = self.beta(graph_time)
            node_diffusion = torch.sqrt(beta)[batch.batch_index].unsqueeze(1)
            mean_state = state + dt * drift
            stochastic_increment = torch.sqrt(-dt) * node_diffusion * _randn_by_graph(
                batch,
                generators,
            )
            state = mean_state + stochastic_increment

        if final_denoise:
            endpoint_times = times[-1].expand(batch.num_graphs)
            epsilon_hat = _model_epsilon(model, batch, state, endpoint_times)
            alpha, sigma = self.marginal_coefficients(endpoint_times)
            node_alpha = alpha[batch.batch_index].unsqueeze(1)
            node_sigma = sigma[batch.batch_index].unsqueeze(1)
            state = (state - node_sigma * epsilon_hat) / node_alpha

        if not torch.isfinite(state).all():
            raise ValueError("VP-SDE sampler produced NaN or Inf values")
        return state

    def sampler_name(
        self,
        *,
        steps: int,
        method: str,
        solver: str,
        final_denoise: bool = True,
    ) -> str:
        """Return an explicit label for one VP sampling configuration."""

        step_count = _positive_int(steps, "steps")
        if method == "probability_flow_ode" and solver in {"euler", "heun"}:
            base = f"vp_probability_flow_{solver}_steps{step_count}"
        elif method == "reverse_sde" and solver == "euler_maruyama":
            base = f"vp_reverse_sde_euler_maruyama_steps{step_count}"
        else:
            raise ValueError("invalid VP-SDE method/solver combination")
        return f"{base}_denoise" if final_denoise else base

    def _vp_problem(
        self,
        batch: NodeRegressionBatch,
        times: torch.Tensor,
        noise: torch.Tensor,
    ) -> NodeRegressionBatch:
        _validate_graph_times(
            times,
            batch.num_graphs,
            lower_bound=self.training_eps,
        )
        if noise.shape != batch.inputs.shape:
            raise ValueError("VP-SDE noise must have the same shape as the clean state")
        if noise.dtype != batch.inputs.dtype or noise.device != batch.inputs.device:
            raise TypeError("VP-SDE noise must share batch input dtype and device")

        alpha, sigma = self.marginal_coefficients(times)
        node_alpha = alpha[batch.batch_index].unsqueeze(1)
        node_sigma = sigma[batch.batch_index].unsqueeze(1)
        state = node_alpha * batch.inputs + node_sigma * noise
        epsilon_channels = tuple(f"epsilon:{name}" for name in batch.input_channels)
        return replace(
            batch,
            inputs=state,
            targets=noise,
            conditioning=_time_conditioning(batch, times),
            conditioning_names=batch.conditioning_names + (_VP_TIME_NAME,),
            target_channels=epsilon_channels,
        )


def _validate_vp_batch(batch: NodeRegressionBatch) -> None:
    _validate_state_batch(batch)
    if _VP_TIME_NAME in batch.conditioning_names:
        raise ValueError("base conditioning must not already define vp_sde_time")


def _validate_sampling_state(batch: NodeRegressionBatch, state: torch.Tensor) -> None:
    if state.shape != batch.inputs.shape:
        raise ValueError("VP-SDE state must match batch input shape")
    if state.device != batch.inputs.device or state.dtype != batch.inputs.dtype:
        raise TypeError("VP-SDE state must share batch input dtype and device")


def _validate_graph_times(
    times: torch.Tensor,
    num_graphs: int,
    *,
    lower_bound: float,
) -> None:
    _validate_times_tensor(times)
    if times.shape != (num_graphs,):
        raise ValueError(f"VP-SDE times must have shape [{num_graphs}]")
    if bool(torch.any(times < lower_bound)) or bool(torch.any(times > 1.0)):
        raise ValueError(f"VP-SDE times must lie in [{lower_bound}, 1]")


def _validate_times_tensor(times: torch.Tensor) -> None:
    if not isinstance(times, torch.Tensor):
        raise TypeError("times must be a torch.Tensor")
    if not times.is_floating_point():
        raise TypeError("times must use a floating-point dtype")
    if not torch.isfinite(times).all():
        raise ValueError("times must contain only finite values")


def _positive_float(value: object, name: str) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _open_unit_float(value: object, name: str) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return result
