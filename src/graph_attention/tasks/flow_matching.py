"""Linear Gaussian flow-matching task for packed node fields."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import replace
from operator import index as operator_index

import torch
from torch import nn

from .regression import NodeRegressionBatch, NodeRegressionTask

_FLOW_TIME_NAME = "flow_time"


class FlowMatchingTask(NodeRegressionTask):
    """Unconditional straight-path flow matching in standardized state space.

    Physical field selection and optional nondimensionalization reuse the frozen
    ``NodeRegressionTask`` data preparation path. Statistical standardization is
    applied by the training layer before this task constructs the Gaussian path.
    """

    def __init__(
        self,
        state_fields: Iterable[str],
        conditioning_parameters: Iterable[str] = (),
        physical_nondimensionalization: bool = False,
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
        self.validation_seed = _nonnegative_int(validation_seed, "validation_seed")

    def make_training_problem(
        self,
        batch: NodeRegressionBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> NodeRegressionBatch:
        """Sample one straight Gaussian-to-data path point per physical graph."""

        _validate_state_batch(batch)
        times = torch.rand(
            (batch.num_graphs,),
            device=batch.inputs.device,
            dtype=batch.inputs.dtype,
            generator=generator,
        )
        source = torch.randn(
            batch.inputs.shape,
            device=batch.inputs.device,
            dtype=batch.inputs.dtype,
            generator=generator,
        )
        return _flow_problem(batch, times, source)

    def make_validation_problem(self, batch: NodeRegressionBatch) -> NodeRegressionBatch:
        """Construct deterministic path points keyed only by validation seed and sample ID."""

        _validate_state_batch(batch)
        times, source = _deterministic_validation_inputs(batch, self.validation_seed)
        return _flow_problem(batch, times, source)

    @torch.no_grad()
    def sample_standardized(
        self,
        model: nn.Module,
        batch: NodeRegressionBatch,
        *,
        steps: int,
        solver: str = "heun",
        sampling_seed: int = 5678,
    ) -> torch.Tensor:
        """Integrate ``dx/dt = v_theta(x,t)`` from deterministic Gaussian sources."""

        _validate_state_batch(batch)
        step_count = _positive_int(steps, "steps")
        solver_name = str(solver)
        if solver_name not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        seed = _nonnegative_int(sampling_seed, "sampling_seed")

        state = _deterministic_sampling_source(batch, seed)
        dt = 1.0 / step_count
        for step in range(step_count):
            t_value = step / step_count
            times = torch.full(
                (batch.num_graphs,),
                t_value,
                dtype=state.dtype,
                device=state.device,
            )
            k1 = _model_velocity(model, batch, state, times)
            if solver_name == "euler":
                state = state + dt * k1
                continue

            predictor = state + dt * k1
            next_times = torch.full(
                (batch.num_graphs,),
                (step + 1) / step_count,
                dtype=state.dtype,
                device=state.device,
            )
            k2 = _model_velocity(model, batch, predictor, next_times)
            state = state + 0.5 * dt * (k1 + k2)

        if not torch.isfinite(state).all():
            raise ValueError("flow-matching sampler produced NaN or Inf values")
        return state


def _flow_problem(
    batch: NodeRegressionBatch,
    times: torch.Tensor,
    source: torch.Tensor,
) -> NodeRegressionBatch:
    if times.shape != (batch.num_graphs,) or not times.is_floating_point():
        raise ValueError(f"times must have floating shape [{batch.num_graphs}]")
    if times.dtype != batch.inputs.dtype or times.device != batch.inputs.device:
        raise TypeError("times must share batch input dtype and device")
    if source.shape != batch.inputs.shape:
        raise ValueError("Gaussian source must have the same shape as the data state")
    if source.dtype != batch.inputs.dtype or source.device != batch.inputs.device:
        raise TypeError("Gaussian source must share batch input dtype and device")

    node_times = times[batch.batch_index].unsqueeze(1)
    state = (1.0 - node_times) * source + node_times * batch.inputs
    velocity = batch.inputs - source
    conditioning = _time_conditioning(batch, times)
    velocity_channels = tuple(f"d_dt:{name}" for name in batch.input_channels)
    return replace(
        batch,
        inputs=state,
        targets=velocity,
        conditioning=conditioning,
        conditioning_names=batch.conditioning_names + (_FLOW_TIME_NAME,),
        target_channels=velocity_channels,
    )


def _time_conditioning(batch: NodeRegressionBatch, times: torch.Tensor) -> torch.Tensor:
    return torch.cat((batch.conditioning, times.unsqueeze(1)), dim=1)


def _deterministic_validation_inputs(
    batch: NodeRegressionBatch,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    times: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    node_counts = _node_counts(batch)
    for node_count, sample_id in zip(node_counts, batch.source.sample_ids, strict=True):
        generator = _sample_generator(batch.inputs.device, seed, "validation", sample_id)
        times.append(
            torch.rand(
                (),
                device=batch.inputs.device,
                dtype=batch.inputs.dtype,
                generator=generator,
            )
        )
        sources.append(
            torch.randn(
                (node_count, batch.inputs.shape[1]),
                device=batch.inputs.device,
                dtype=batch.inputs.dtype,
                generator=generator,
            )
        )
    return torch.stack(times), torch.cat(sources, dim=0)


def _deterministic_sampling_source(batch: NodeRegressionBatch, seed: int) -> torch.Tensor:
    sources: list[torch.Tensor] = []
    node_counts = _node_counts(batch)
    for node_count, sample_id in zip(node_counts, batch.source.sample_ids, strict=True):
        generator = _sample_generator(batch.inputs.device, seed, "sampling", sample_id)
        sources.append(
            torch.randn(
                (node_count, batch.inputs.shape[1]),
                device=batch.inputs.device,
                dtype=batch.inputs.dtype,
                generator=generator,
            )
        )
    return torch.cat(sources, dim=0)


def _node_counts(batch: NodeRegressionBatch) -> list[int]:
    counts = (batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().tolist()
    return [int(value) for value in counts]


def _sample_generator(
    device: torch.device,
    seed: int,
    purpose: str,
    sample_id: str,
) -> torch.Generator:
    if not sample_id:
        raise ValueError("deterministic flow matching requires non-empty sample IDs")
    payload = f"{seed}\0{purpose}\0{sample_id}".encode("utf-8")
    hashed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    manual_seed = hashed % (2**63 - 1)
    return torch.Generator(device=device).manual_seed(manual_seed)


def _model_velocity(
    model: nn.Module,
    batch: NodeRegressionBatch,
    state: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    conditioning = _time_conditioning(batch, times)
    velocity = model(
        state,
        edge_index=batch.edge_index,
        coords=batch.coords,
        batch_index=batch.batch_index,
        conditioning=conditioning,
    )
    if velocity.shape != state.shape:
        raise ValueError(
            "flow-matching model output must match the state shape: "
            f"got {tuple(velocity.shape)}, expected {tuple(state.shape)}"
        )
    return velocity


def _validate_state_batch(batch: NodeRegressionBatch) -> None:
    if _FLOW_TIME_NAME in batch.conditioning_names:
        raise ValueError("base conditioning must not already define the reserved flow_time name")
    if batch.input_channels != batch.target_channels:
        raise ValueError("flow matching requires identical ordered input and target state channels")
    if batch.inputs.shape != batch.targets.shape:
        raise ValueError("flow matching requires identical input and target state shapes")
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
