"""Discrete DDPM diffusion task for packed node fields."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import replace
from operator import index as operator_index

import torch
from torch import nn

from .regression import NodeRegressionBatch, NodeRegressionTask

_DIFFUSION_TIME_NAME = "diffusion_time"


class DiffusionDenoisingTask(NodeRegressionTask):
    """Unconditional epsilon-prediction diffusion in standardized state space.

    Physical field selection and optional nondimensionalization reuse the frozen
    ``NodeRegressionTask`` path. Statistical standardization is applied before
    this task constructs the forward diffusion problem.
    """

    def __init__(
        self,
        state_fields: Iterable[str],
        conditioning_parameters: Iterable[str] = (),
        physical_nondimensionalization: bool = False,
        timesteps: int = 1000,
        cosine_s: float = 0.008,
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
        self.timesteps = _positive_int(timesteps, "timesteps")
        self.cosine_s = float(cosine_s)
        if self.cosine_s < 0.0:
            raise ValueError("cosine_s must be non-negative")
        self.validation_seed = _nonnegative_int(validation_seed, "validation_seed")
        self._alpha_bar_cpu = _cosine_discrete_alpha_bar(self.timesteps, self.cosine_s)
        self._alpha_bar_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def make_training_problem(
        self,
        batch: NodeRegressionBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> NodeRegressionBatch:
        """Sample one DDPM noising problem per physical graph."""

        _validate_state_batch(batch)
        timesteps = torch.randint(
            1,
            self.timesteps + 1,
            (batch.num_graphs,),
            device=batch.inputs.device,
            generator=generator,
            dtype=torch.long,
        )
        noise = torch.randn(
            batch.inputs.shape,
            device=batch.inputs.device,
            dtype=batch.inputs.dtype,
            generator=generator,
        )
        return self._diffusion_problem(batch, timesteps, noise)

    def make_validation_problem(self, batch: NodeRegressionBatch) -> NodeRegressionBatch:
        """Construct deterministic timestep/noise pairs keyed by validation seed and sample ID."""

        _validate_state_batch(batch)
        timesteps: list[torch.Tensor] = []
        noise_parts: list[torch.Tensor] = []
        for node_count, sample_id in zip(
            _node_counts(batch),
            batch.source.sample_ids,
            strict=True,
        ):
            generator = _sample_generator(
                batch.inputs.device,
                self.validation_seed,
                "validation",
                sample_id,
            )
            timesteps.append(
                torch.randint(
                    1,
                    self.timesteps + 1,
                    (),
                    device=batch.inputs.device,
                    generator=generator,
                    dtype=torch.long,
                )
            )
            noise_parts.append(
                torch.randn(
                    (node_count, batch.inputs.shape[1]),
                    device=batch.inputs.device,
                    dtype=batch.inputs.dtype,
                    generator=generator,
                )
            )
        return self._diffusion_problem(
            batch,
            torch.stack(timesteps),
            torch.cat(noise_parts, dim=0),
        )

    @torch.no_grad()
    def sample_standardized(
        self,
        model: nn.Module,
        batch: NodeRegressionBatch,
        *,
        steps: int,
        eta: float = 0.0,
        sampling_seed: int = 5678,
        sampling_keys: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Sample with generalized DDIM; eta=1 and steps=T is ancestral DDPM."""

        _validate_state_batch(batch)
        step_count = _positive_int(steps, "steps")
        if step_count > self.timesteps:
            raise ValueError(f"steps must be <= timesteps ({self.timesteps})")
        eta_value = float(eta)
        if not 0.0 <= eta_value <= 1.0:
            raise ValueError("eta must lie in [0, 1]")
        seed = _nonnegative_int(sampling_seed, "sampling_seed")

        keys = tuple(batch.source.sample_ids) if sampling_keys is None else tuple(sampling_keys)
        if len(keys) != batch.num_graphs:
            raise ValueError("sampling_keys must contain one key per graph")
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("sampling_keys must contain non-empty strings")

        generators = [
            _sample_generator(batch.inputs.device, seed, "sampling", key) for key in keys
        ]
        state = _randn_by_graph(batch, generators)
        alpha_bar = self._alpha_bar_for(batch.inputs)
        sampling_times = _sampling_timesteps(
            self.timesteps,
            step_count,
            device=batch.inputs.device,
        )

        for index, t_scalar in enumerate(sampling_times.tolist()):
            previous_scalar = (
                int(sampling_times[index + 1]) if index + 1 < sampling_times.numel() else 0
            )
            timesteps = torch.full(
                (batch.num_graphs,),
                int(t_scalar),
                device=state.device,
                dtype=torch.long,
            )
            epsilon_hat = _model_epsilon(
                model,
                batch,
                state,
                self._normalized_time(timesteps, state.dtype),
            )

            alpha_t = alpha_bar[int(t_scalar)]
            alpha_previous = alpha_bar[previous_scalar]
            x0_hat = (
                state - torch.sqrt(1.0 - alpha_t) * epsilon_hat
            ) / torch.sqrt(alpha_t)

            if previous_scalar == 0:
                state = x0_hat
                continue

            variance_factor = (
                (1.0 - alpha_previous)
                / (1.0 - alpha_t)
                * (1.0 - alpha_t / alpha_previous)
            )
            sigma = eta_value * torch.sqrt(torch.clamp(variance_factor, min=0.0))
            direction_scale = torch.sqrt(
                torch.clamp(1.0 - alpha_previous - sigma**2, min=0.0)
            )
            state = torch.sqrt(alpha_previous) * x0_hat + direction_scale * epsilon_hat
            if eta_value > 0.0:
                state = state + sigma * _randn_by_graph(batch, generators)

        if not torch.isfinite(state).all():
            raise ValueError("diffusion sampler produced NaN or Inf values")
        return state

    def sampler_name(self, *, steps: int, eta: float) -> str:
        """Return an explicit label for the configured reverse process."""

        step_count = _positive_int(steps, "steps")
        eta_value = float(eta)
        if step_count == self.timesteps and eta_value == 1.0:
            return "ddpm_ancestral"
        return "ddim"

    def _diffusion_problem(
        self,
        batch: NodeRegressionBatch,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> NodeRegressionBatch:
        if timesteps.shape != (batch.num_graphs,) or timesteps.dtype != torch.long:
            raise ValueError(f"timesteps must have long shape [{batch.num_graphs}]")
        if timesteps.device != batch.inputs.device:
            raise ValueError("timesteps must be on the batch input device")
        if bool(torch.any(timesteps < 1)) or bool(torch.any(timesteps > self.timesteps)):
            raise ValueError(f"timesteps must lie in [1, {self.timesteps}]")
        if noise.shape != batch.inputs.shape:
            raise ValueError("diffusion noise must have the same shape as the data state")
        if noise.dtype != batch.inputs.dtype or noise.device != batch.inputs.device:
            raise TypeError("diffusion noise must share batch input dtype and device")

        alpha_bar = self._alpha_bar_for(batch.inputs)
        node_timesteps = timesteps[batch.batch_index]
        node_alpha = alpha_bar[node_timesteps].unsqueeze(1)
        state = torch.sqrt(node_alpha) * batch.inputs + torch.sqrt(1.0 - node_alpha) * noise
        conditioning = _time_conditioning(
            batch,
            self._normalized_time(timesteps, batch.inputs.dtype),
        )
        epsilon_channels = tuple(f"epsilon:{name}" for name in batch.input_channels)
        return replace(
            batch,
            inputs=state,
            targets=noise,
            conditioning=conditioning,
            conditioning_names=batch.conditioning_names + (_DIFFUSION_TIME_NAME,),
            target_channels=epsilon_channels,
        )

    def _normalized_time(self, timesteps: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return timesteps.to(dtype=dtype) / float(self.timesteps)

    def _alpha_bar_for(self, reference: torch.Tensor) -> torch.Tensor:
        key = (str(reference.device), reference.dtype)
        cached = self._alpha_bar_cache.get(key)
        if cached is None:
            cached = self._alpha_bar_cpu.to(device=reference.device, dtype=reference.dtype)
            self._alpha_bar_cache[key] = cached
        return cached


def _cosine_discrete_alpha_bar(timesteps: int, cosine_s: float) -> torch.Tensor:
    continuous_time = torch.linspace(0.0, 1.0, timesteps + 1, dtype=torch.float64)
    raw = torch.cos(
        (continuous_time + cosine_s) / (1.0 + cosine_s) * torch.pi / 2.0
    ) ** 2
    raw = raw / raw[0]
    betas = 1.0 - raw[1:] / raw[:-1]
    betas = betas.clamp(min=1.0e-8, max=0.999)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return torch.cat((torch.ones(1, dtype=torch.float64), alpha_bar), dim=0)


def _sampling_timesteps(
    timesteps: int,
    steps: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    values = torch.linspace(timesteps, 1, steps=steps, device=device)
    values = torch.round(values).to(dtype=torch.long)
    values = torch.unique_consecutive(values)
    if values.numel() != steps:
        raise RuntimeError("sampling timestep construction produced duplicate timesteps")
    return values


def _time_conditioning(batch: NodeRegressionBatch, times: torch.Tensor) -> torch.Tensor:
    return torch.cat((batch.conditioning, times.unsqueeze(1)), dim=1)


def _randn_by_graph(
    batch: NodeRegressionBatch,
    generators: Sequence[torch.Generator],
) -> torch.Tensor:
    parts = [
        torch.randn(
            (node_count, batch.inputs.shape[1]),
            device=batch.inputs.device,
            dtype=batch.inputs.dtype,
            generator=generator,
        )
        for node_count, generator in zip(_node_counts(batch), generators, strict=True)
    ]
    return torch.cat(parts, dim=0)


def _node_counts(batch: NodeRegressionBatch) -> list[int]:
    counts = (batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().tolist()
    return [int(value) for value in counts]


def _sample_generator(
    device: torch.device,
    seed: int,
    purpose: str,
    sample_key: str,
) -> torch.Generator:
    payload = f"{seed}\0{purpose}\0{sample_key}".encode()
    hashed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    manual_seed = hashed % (2**63 - 1)
    return torch.Generator(device=device).manual_seed(manual_seed)


def _model_epsilon(
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
    epsilon = model(state, **model_kwargs)
    if epsilon.shape != state.shape:
        raise ValueError(
            "diffusion model output must match the state shape: "
            f"got {tuple(epsilon.shape)}, expected {tuple(state.shape)}"
        )
    return epsilon


def _validate_state_batch(batch: NodeRegressionBatch) -> None:
    if _DIFFUSION_TIME_NAME in batch.conditioning_names:
        raise ValueError("base conditioning must not already define diffusion_time")
    if batch.input_channels != batch.target_channels:
        raise ValueError("diffusion requires identical ordered input and target state channels")
    if batch.inputs.shape != batch.targets.shape:
        raise ValueError("diffusion requires identical input and target state shapes")
    if batch.conditioning.ndim != 2 or batch.conditioning.shape[0] != batch.num_graphs:
        raise ValueError("base conditioning must have shape [B, C]")


def _positive_int(value: int, name: str) -> int:
    count = _integer(value, name)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count


def _nonnegative_int(value: int, name: str) -> int:
    count = _integer(value, name)
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
