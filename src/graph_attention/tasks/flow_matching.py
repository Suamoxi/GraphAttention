"""Linear Gaussian flow-matching task for packed node fields."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import replace
from operator import index as operator_index

import torch
from torch import nn

from ..objectives import SampleLossAggregate, sample_reduced_mse
from .regression import NodeRegressionBatch, NodeRegressionTask

_FLOW_TIME_NAME = "flow_time"


class FlowMatchingTask(NodeRegressionTask):
    """Unconditional straight-path flow matching in standardized state space.

    Physical field selection and optional nondimensionalization reuse the frozen
    ``NodeRegressionTask`` data preparation path. Statistical standardization is
    applied by the training layer before this task constructs the Gaussian path.
    """

    training_metric_name = "flow_velocity_mse"
    training_seed_name = "path_seed"

    def __init__(
        self,
        state_fields: Iterable[str],
        conditioning_parameters: Iterable[str] = (),
        physical_nondimensionalization: bool = False,
        validation_seed: int = 1234,
        time_embedding_scale: float = 1.0,
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
        self.time_embedding_scale = float(time_embedding_scale)
        if not torch.isfinite(torch.tensor(self.time_embedding_scale)):
            raise ValueError("time_embedding_scale must be finite")
        if self.time_embedding_scale <= 0.0:
            raise ValueError("time_embedding_scale must be positive")

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
        return _flow_problem(
            batch,
            times,
            source,
            time_embedding_scale=self.time_embedding_scale,
        )

    def make_validation_problem(self, batch: NodeRegressionBatch) -> NodeRegressionBatch:
        """Construct deterministic path points keyed only by validation seed and sample ID."""

        _validate_state_batch(batch)
        times, source = _deterministic_validation_inputs(batch, self.validation_seed)
        return _flow_problem(batch, times, source)

    def make_model_probe(self, problem: NodeRegressionBatch) -> NodeRegressionBatch:
        """Expose the model-facing batch for the common generative runner."""

        if not isinstance(problem, NodeRegressionBatch):
            raise TypeError("flow-matching problem must be a NodeRegressionBatch")
        return problem

    def training_loss(
        self,
        predictions: torch.Tensor,
        problem: NodeRegressionBatch,
    ) -> SampleLossAggregate:
        """Return equal-physical-sample velocity MSE for common training orchestration."""

        if not isinstance(problem, NodeRegressionBatch):
            raise TypeError("flow-matching problem must be a NodeRegressionBatch")
        if predictions.shape != problem.targets.shape:
            raise ValueError(
                "flow-matching predictions must match target velocity shape: "
                f"got {tuple(predictions.shape)}, expected {tuple(problem.targets.shape)}"
            )
        return sample_reduced_mse(
            predictions,
            problem.targets,
            problem.ptr,
            node_weights=problem.node_weights,
        )

    def training_summary_metadata(self) -> dict[str, object]:
        """Describe the unchanged M11 straight-path flow-matching formulation."""

        return {
            "task": "linear_gaussian_flow_matching",
            "prediction_type": "velocity",
            "training_time_distribution": "t ~ Uniform(0,1)",
            "time_conditioning": "scaled_scalar_time",
            "time_embedding_scale": self.time_embedding_scale,
            "path": "x_t=(1-t)*x_source+t*x_data",
            "target_velocity": "x_data-x_source",
            "generation": "task.sample_standardized",
        }

    def sampler_name(
        self,
        *,
        steps: int,
        method: str = "flow_ode",
        solver: str = "heun",
        final_denoise: bool = False,
    ) -> str:
        """Return a stable label for the common generation artifact path."""

        step_count = _positive_int(steps, "steps")
        method_name = str(method)
        solver_name = str(solver)
        if method_name != "flow_ode":
            raise ValueError("flow matching generation requires method='flow_ode'")
        if solver_name not in {"euler", "heun"}:
            raise ValueError("flow matching solver must be 'euler' or 'heun'")
        if final_denoise:
            raise ValueError("flow matching does not use a separate final_denoise step")
        return f"flow_ode_{solver_name}_steps{step_count}"

    @torch.no_grad()
    def sample_standardized(
        self,
        model: nn.Module,
        batch: NodeRegressionBatch,
        *,
        steps: int,
        method: str = "flow_ode",
        solver: str = "heun",
        sampling_eps: float = 0.0,
        final_denoise: bool = False,
        sampling_seed: int = 5678,
        sampling_keys: Iterable[str] | None = None,
    ) -> torch.Tensor:
        """Integrate ``dx/dt = v_theta(x,t)`` from deterministic Gaussian sources."""

        _validate_state_batch(batch)
        step_count = _positive_int(steps, "steps")
        method_name = str(method)
        solver_name = str(solver)
        if method_name != "flow_ode":
            raise ValueError("flow matching generation requires method='flow_ode'")
        if solver_name not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        if float(sampling_eps) != 0.0:
            raise ValueError("flow matching sampling_eps must be exactly 0.0")
        if final_denoise:
            raise ValueError("flow matching does not use a separate final_denoise step")
        seed = _nonnegative_int(sampling_seed, "sampling_seed")
        keys = _sampling_keys(batch, sampling_keys)

        state = _deterministic_sampling_source(batch, seed, keys)
        dt = 1.0 / step_count
        for step in range(step_count):
            t_value = step / step_count
            times = torch.full(
                (batch.num_graphs,),
                t_value,
                dtype=state.dtype,
                device=state.device,
            )
            k1 = _model_velocity(
                model,
                batch,
                state,
                times,
                time_embedding_scale=self.time_embedding_scale,
            )
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
            k2 = _model_velocity(
                model,
                batch,
                predictor,
                next_times,
                time_embedding_scale=self.time_embedding_scale,
            )
            state = state + 0.5 * dt * (k1 + k2)

        if not torch.isfinite(state).all():
            raise ValueError("flow-matching sampler produced NaN or Inf values")
        return state


def _flow_problem(
    batch: NodeRegressionBatch,
    times: torch.Tensor,
    source: torch.Tensor,
    *,
    time_embedding_scale: float,
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
    conditioning = _time_conditioning(
        batch,
        times,
        time_embedding_scale=time_embedding_scale,
    )
    velocity_channels = tuple(f"d_dt:{name}" for name in batch.input_channels)
    return replace(
        batch,
        inputs=state,
        targets=velocity,
        conditioning=conditioning,
        conditioning_names=batch.conditioning_names + (_FLOW_TIME_NAME,),
        target_channels=velocity_channels,
    )


def _time_conditioning(
    batch: NodeRegressionBatch,
    times: torch.Tensor,
    *,
    time_embedding_scale: float,
) -> torch.Tensor:
    scaled_times = times * time_embedding_scale
    return torch.cat((batch.conditioning, scaled_times.unsqueeze(1)), dim=1)


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


def _sampling_keys(
    batch: NodeRegressionBatch,
    sampling_keys: Iterable[str] | None,
) -> tuple[str, ...]:
    if sampling_keys is None:
        keys = tuple(batch.source.sample_ids)
    else:
        keys = tuple(sampling_keys)
    if len(keys) != batch.num_graphs:
        raise ValueError(
            "sampling_keys must contain exactly one key per graph: "
            f"got {len(keys)}, expected {batch.num_graphs}"
        )
    if any(not isinstance(value, str) or not value for value in keys):
        raise ValueError("sampling_keys must contain non-empty strings")
    return keys


def _deterministic_sampling_source(
    batch: NodeRegressionBatch,
    seed: int,
    sampling_keys: tuple[str, ...],
) -> torch.Tensor:
    sources: list[torch.Tensor] = []
    node_counts = _node_counts(batch)
    for node_count, sample_key in zip(node_counts, sampling_keys, strict=True):
        generator = _sample_generator(batch.inputs.device, seed, "sampling", sample_key)
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
    payload = f"{seed}\0{purpose}\0{sample_id}".encode()
    hashed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    manual_seed = hashed % (2**63 - 1)
    return torch.Generator(device=device).manual_seed(manual_seed)


def _model_velocity(
    model: nn.Module,
    batch: NodeRegressionBatch,
    state: torch.Tensor,
    times: torch.Tensor,
    *,
    time_embedding_scale: float,
) -> torch.Tensor:
    conditioning = _time_conditioning(
        batch,
        times,
        time_embedding_scale=time_embedding_scale,
    )
    model_kwargs = {
        "edge_index": batch.edge_index,
        "coords": batch.coords,
        "batch_index": batch.batch_index,
        "conditioning": conditioning,
    }
    if batch.attention_edge_indices:
        model_kwargs["attention_edge_indices"] = batch.attention_edge_indices
    velocity = model(state, **model_kwargs)
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
