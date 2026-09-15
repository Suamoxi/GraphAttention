"""EDM preconditioned denoising and Karras sigma sampling for packed CFD graphs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import torch
from torch import nn

from .diffusion import (
    _node_counts,
    _randn_by_graph,
    _sample_generator,
    _time_conditioning,
    _validate_state_batch,
)
from .regression import NodeRegressionBatch, NodeRegressionTask

_EDM_NOISE_NAME = "edm_log_sigma_over_4"


@dataclass(frozen=True, slots=True)
class EDMProblem:
    """One EDM perturbation together with the variables needed by its loss."""

    model_batch: NodeRegressionBatch
    clean_state: torch.Tensor
    noisy_state: torch.Tensor
    sigmas: torch.Tensor

    @property
    def num_graphs(self) -> int:
        return self.model_batch.num_graphs


@dataclass(frozen=True, slots=True)
class EDMLoss:
    """Per-sample EDM weighted denoising loss."""

    per_sample: torch.Tensor

    @property
    def mean(self) -> torch.Tensor:
        return self.per_sample.mean()

    @property
    def loss_sum(self) -> torch.Tensor:
        return self.per_sample.sum()

    @property
    def sample_count(self) -> int:
        return int(self.per_sample.numel())


class EDMDenoisingTask(NodeRegressionTask):
    """Continuous-noise EDM objective with Karras/EDM preconditioning.

    Training follows Karras et al. (2022): one log-normal noise level is sampled
    per physical graph, additive Gaussian noise is applied in standardized data
    space, and the network is wrapped by EDM input/output/noise preconditioning.

    The raw network output ``F`` is interpreted as

        D(x, sigma) = c_skip * x + c_out * F(c_in * x, c_noise)

    and optimized with the EDM weighted denoising objective.  Sampling uses the
    Karras power-law sigma discretization and deterministic Euler or Heun ODE
    integration.  Stochastic churn is intentionally excluded from the first
    controlled project baseline.
    """

    def __init__(
        self,
        state_fields: Iterable[str],
        conditioning_parameters: Iterable[str] = (),
        physical_nondimensionalization: bool = False,
        sigma_data: float = 1.0,
        p_mean: float = -0.5068528194400547,
        p_std: float = 1.2,
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
        self.sigma_data = _positive_float(sigma_data, "sigma_data")
        self.p_mean = float(p_mean)
        self.p_std = _positive_float(p_std, "p_std")
        if not torch.isfinite(torch.tensor(self.p_mean)):
            raise ValueError("p_mean must be finite")
        self.validation_seed = _nonnegative_int(validation_seed, "validation_seed")

    @property
    def prediction_type(self) -> str:
        return "edm_preconditioned_denoised_state"

    def make_training_problem(
        self,
        batch: NodeRegressionBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> EDMProblem:
        """Sample one log-normal EDM noise level per physical graph."""

        _validate_state_batch(batch)
        sigmas = torch.exp(
            torch.randn(
                (batch.num_graphs,),
                device=batch.inputs.device,
                dtype=batch.inputs.dtype,
                generator=generator,
            )
            * self.p_std
            + self.p_mean
        )
        noise = torch.randn(
            batch.inputs.shape,
            device=batch.inputs.device,
            dtype=batch.inputs.dtype,
            generator=generator,
        )
        return self._make_problem(batch, sigmas, noise)

    def make_validation_problem(self, batch: NodeRegressionBatch) -> EDMProblem:
        """Construct deterministic EDM sigma/noise pairs keyed by sample ID."""

        _validate_state_batch(batch)
        sigmas: list[torch.Tensor] = []
        noise_parts: list[torch.Tensor] = []
        for node_count, sample_id in zip(
            _node_counts(batch),
            batch.source.sample_ids,
            strict=True,
        ):
            generator = _sample_generator(
                batch.inputs.device,
                self.validation_seed,
                "edm_validation",
                sample_id,
            )
            log_sigma = (
                torch.randn(
                    (),
                    device=batch.inputs.device,
                    dtype=batch.inputs.dtype,
                    generator=generator,
                )
                * self.p_std
                + self.p_mean
            )
            sigmas.append(torch.exp(log_sigma))
            noise_parts.append(
                torch.randn(
                    (node_count, batch.inputs.shape[1]),
                    device=batch.inputs.device,
                    dtype=batch.inputs.dtype,
                    generator=generator,
                )
            )
        return self._make_problem(
            batch,
            torch.stack(sigmas),
            torch.cat(noise_parts, dim=0),
        )

    def make_model_probe(self, problem: EDMProblem) -> NodeRegressionBatch:
        """Return the already preconditioned model-facing batch."""

        return problem.model_batch

    def edm_loss(self, model_output: torch.Tensor, problem: EDMProblem) -> EDMLoss:
        """Return the canonical EDM weighted denoising objective per graph."""

        _validate_model_output(model_output, problem.clean_state)
        batch = problem.model_batch
        sigmas = problem.sigmas.to(
            device=problem.noisy_state.device,
            dtype=problem.noisy_state.dtype,
        )
        node_sigma = sigmas[batch.batch_index].unsqueeze(1)
        c_skip, c_out, _ = self._preconditioning(node_sigma)
        denoised = c_skip * problem.noisy_state + c_out * model_output
        weight = (node_sigma.square() + self.sigma_data**2) / (
            node_sigma * self.sigma_data
        ).square()
        node_loss = (weight * (denoised - problem.clean_state).square()).mean(dim=1)
        per_sample = _reduce_node_values(node_loss, batch)
        if not torch.isfinite(per_sample).all():
            raise ValueError("EDM loss contains NaN or Inf")
        return EDMLoss(per_sample=per_sample)

    @torch.no_grad()
    def denoise(
        self,
        model: nn.Module,
        batch: NodeRegressionBatch,
        state: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """Apply EDM preconditioning and return the denoised-state estimate."""

        if state.ndim != 2 or state.shape != batch.inputs.shape:
            raise ValueError("EDM state must match batch input shape [N, C]")
        _validate_graph_sigmas(sigmas, batch.num_graphs)
        sigma_state = sigmas.to(device=state.device, dtype=state.dtype)
        node_sigma = sigma_state[batch.batch_index].unsqueeze(1)
        c_skip, c_out, c_in = self._preconditioning(node_sigma)

        model_dtype = batch.inputs.dtype
        model_state = (c_in * state).to(dtype=model_dtype)
        c_noise = (torch.log(sigma_state) / 4.0).to(dtype=model_dtype)
        conditioning = _time_conditioning(batch, c_noise)
        model_kwargs = {
            "edge_index": batch.edge_index,
            "coords": batch.coords,
            "batch_index": batch.batch_index,
            "conditioning": conditioning,
        }
        if batch.attention_edge_indices:
            model_kwargs["attention_edge_indices"] = batch.attention_edge_indices
        raw = model(model_state, **model_kwargs)
        _validate_model_output(raw, batch.inputs)
        return c_skip * state + c_out * raw.to(dtype=state.dtype)

    @torch.no_grad()
    def sample_standardized(
        self,
        model: nn.Module,
        batch: NodeRegressionBatch,
        *,
        steps: int = 18,
        sigma_min: float = 0.004,
        sigma_max: float = 160.0,
        rho: float = 7.0,
        solver: str = "heun",
        sampling_seed: int = 5678,
        sampling_keys: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Sample with the deterministic Karras sigma grid and Euler/Heun ODE solver."""

        _validate_state_batch(batch)
        step_count = _positive_int(steps, "steps")
        if step_count < 2:
            raise ValueError("EDM sampling requires at least two nonzero sigma steps")
        sigma_min_value = _positive_float(sigma_min, "sigma_min")
        sigma_max_value = _positive_float(sigma_max, "sigma_max")
        if sigma_max_value <= sigma_min_value:
            raise ValueError("sigma_max must be greater than sigma_min")
        rho_value = _positive_float(rho, "rho")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        seed = _nonnegative_int(sampling_seed, "sampling_seed")

        keys = tuple(batch.source.sample_ids) if sampling_keys is None else tuple(sampling_keys)
        if len(keys) != batch.num_graphs:
            raise ValueError("sampling_keys must contain one key per graph")
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("sampling_keys must contain non-empty strings")

        generators = [
            _sample_generator(batch.inputs.device, seed, "edm_sampling", key)
            for key in keys
        ]
        sigmas = karras_sigma_schedule(
            step_count,
            sigma_min=sigma_min_value,
            sigma_max=sigma_max_value,
            rho=rho_value,
            device=batch.inputs.device,
        )
        state = _randn_by_graph(batch, generators).to(torch.float64) * sigmas[0]

        for index in range(step_count):
            sigma_cur = sigmas[index]
            sigma_next = sigmas[index + 1]
            graph_sigma = torch.full(
                (batch.num_graphs,),
                float(sigma_cur),
                device=batch.inputs.device,
                dtype=torch.float64,
            )
            denoised = self.denoise(model, batch, state, graph_sigma)
            derivative = (state - denoised) / sigma_cur
            step = sigma_next - sigma_cur
            euler_state = state + step * derivative

            if solver == "heun" and float(sigma_next) > 0.0:
                next_graph_sigma = torch.full(
                    (batch.num_graphs,),
                    float(sigma_next),
                    device=batch.inputs.device,
                    dtype=torch.float64,
                )
                next_denoised = self.denoise(
                    model,
                    batch,
                    euler_state,
                    next_graph_sigma,
                )
                next_derivative = (euler_state - next_denoised) / sigma_next
                state = state + step * 0.5 * (derivative + next_derivative)
            else:
                state = euler_state

        result = state.to(dtype=batch.inputs.dtype)
        if not torch.isfinite(result).all():
            raise ValueError("EDM sampler produced NaN or Inf values")
        return result

    def sampler_name(self, *, steps: int, solver: str) -> str:
        step_count = _positive_int(steps, "steps")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        return f"edm_karras_{solver}_steps{step_count}"

    def _make_problem(
        self,
        batch: NodeRegressionBatch,
        sigmas: torch.Tensor,
        noise: torch.Tensor,
    ) -> EDMProblem:
        _validate_graph_sigmas(sigmas, batch.num_graphs)
        if sigmas.device != batch.inputs.device or sigmas.dtype != batch.inputs.dtype:
            raise TypeError("EDM sigmas must share batch input dtype and device")
        if noise.shape != batch.inputs.shape:
            raise ValueError("EDM noise must have the same shape as the clean state")
        if noise.device != batch.inputs.device or noise.dtype != batch.inputs.dtype:
            raise TypeError("EDM noise must share batch input dtype and device")

        node_sigma = sigmas[batch.batch_index].unsqueeze(1)
        noisy_state = batch.inputs + node_sigma * noise
        _, _, c_in = self._preconditioning(node_sigma)
        c_noise = torch.log(sigmas) / 4.0
        model_batch = replace(
            batch,
            inputs=c_in * noisy_state,
            targets=batch.inputs,
            conditioning=_time_conditioning(batch, c_noise),
            conditioning_names=batch.conditioning_names + (_EDM_NOISE_NAME,),
        )
        return EDMProblem(
            model_batch=model_batch,
            clean_state=batch.inputs,
            noisy_state=noisy_state,
            sigmas=sigmas,
        )

    def _preconditioning(
        self,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_data_sq = self.sigma_data**2
        denominator = sigma.square() + sigma_data_sq
        c_skip = sigma_data_sq / denominator
        c_out = sigma * self.sigma_data / torch.sqrt(denominator)
        c_in = torch.rsqrt(denominator)
        return c_skip, c_out, c_in


def karras_sigma_schedule(
    steps: int,
    *,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the EDM/Karras power-law sigma grid followed by the clean endpoint zero."""

    step_count = _positive_int(steps, "steps")
    if step_count < 2:
        raise ValueError("Karras sigma schedule requires at least two nonzero steps")
    sigma_min_value = _positive_float(sigma_min, "sigma_min")
    sigma_max_value = _positive_float(sigma_max, "sigma_max")
    if sigma_max_value <= sigma_min_value:
        raise ValueError("sigma_max must be greater than sigma_min")
    rho_value = _positive_float(rho, "rho")

    ramp = torch.linspace(0.0, 1.0, step_count, dtype=torch.float64, device=device)
    maximum = sigma_max_value ** (1.0 / rho_value)
    minimum = sigma_min_value ** (1.0 / rho_value)
    sigmas = (maximum + ramp * (minimum - maximum)) ** rho_value
    return torch.cat((sigmas, torch.zeros(1, dtype=torch.float64, device=sigmas.device)))


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
    return torch.stack(per_sample)


def _validate_graph_sigmas(sigmas: torch.Tensor, num_graphs: int) -> None:
    if sigmas.ndim != 1 or sigmas.shape != (num_graphs,):
        raise ValueError(f"EDM sigmas must have shape [{num_graphs}]")
    if not sigmas.is_floating_point():
        raise TypeError("EDM sigmas must be floating-point")
    if not torch.isfinite(sigmas).all() or bool(torch.any(sigmas <= 0.0)):
        raise ValueError("EDM sigmas must be finite and strictly positive")


def _validate_model_output(model_output: torch.Tensor, reference: torch.Tensor) -> None:
    if model_output.shape != reference.shape:
        raise ValueError(
            "EDM model output must match state shape: "
            f"got {tuple(model_output.shape)}, expected {tuple(reference.shape)}"
        )
    if not model_output.is_floating_point():
        raise TypeError("EDM model output must be floating-point")
    if model_output.device != reference.device:
        raise ValueError("EDM model output and state must share a device")


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _positive_float(value: float, name: str) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result
