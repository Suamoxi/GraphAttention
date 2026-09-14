"""Trace the actual reverse diffusion trajectory of a frozen checkpoint."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from scripts.diagnose_slice_diffusion import (
    _diagnostic_timesteps,
    _existing_directory,
    _write_csv,
)
from scripts.generate_slice_diffusion import _load_standardizers, _manifest_test_ids
from scripts.train_slice_ablation import (
    _attention_topologies_from_geometry,
    _instantiate_model,
    _loader,
    _NodeRegressionCollator,
    _positive_int,
    _task_batch_to_device,
)
from scripts.train_slice_diffusion import _validate_diffusion_standardizers

from graph_attention.data import PrecomputedSlicePTDataset
from graph_attention.evaluation.diffusion_diagnostics import reconstruct_x0_from_epsilon
from graph_attention.geometry import cartesian_4_neighbor_edge_index
from graph_attention.tasks import DiffusionDenoisingTask


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="diagnose_diffusion_trajectory",
)
def main(cfg: DictConfig) -> None:
    summary = run_trajectory_diagnostics(cfg)
    print(json.dumps(summary, indent=2, allow_nan=False))


@torch.no_grad()
def run_trajectory_diagnostics(cfg: DictConfig) -> dict[str, Any]:
    """Run the production sampler while recording its internal state trajectory."""

    run_dir = _existing_directory(cfg.run_dir, "run_dir")
    source_config_path = run_dir / "resolved_config.yaml"
    source_summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "dataset_split_manifest.json"
    standardizers_path = run_dir / "standardizers.pt"
    checkpoint_path = run_dir / str(cfg.checkpoint)
    for path, label in (
        (source_config_path, "resolved_config.yaml"),
        (source_summary_path, "summary.json"),
        (manifest_path, "dataset_split_manifest.json"),
        (standardizers_path, "standardizers.pt"),
        (checkpoint_path, "checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"source diffusion {label} does not exist: {path}")

    source_cfg = OmegaConf.load(source_config_path)
    source_summary = json.loads(source_summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    source_seed = _positive_int(source_cfg.seed, "source seed", allow_zero=True)

    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA trajectory diagnostics requested but CUDA is unavailable")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("trajectory diagnostics are intentionally single-process")

    num_workers = _positive_int(cfg.num_workers, "num_workers", allow_zero=True)
    num_samples = _positive_int(cfg.num_samples, "num_samples")
    sampling_seed = _positive_int(cfg.sampling_seed, "sampling_seed", allow_zero=True)
    eta = float(cfg.eta)
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0, 1]")

    dataset = instantiate(source_cfg.data)
    if not isinstance(dataset, PrecomputedSlicePTDataset):
        raise TypeError("source diffusion run must use data=hit_slice_pt")
    task = instantiate(source_cfg.task)
    if not isinstance(task, DiffusionDenoisingTask):
        raise TypeError("source diffusion run must use task=hit_diffusion")

    standardizers = _load_standardizers(standardizers_path)
    _validate_diffusion_standardizers(standardizers)
    if standardizers.physical_nondimensionalization != task.physical_nondimensionalization:
        raise ValueError("saved standardizers and source task disagree on nondimensionalization")
    device_standardizers = standardizers.to(device=device, dtype=torch.float32)

    test_ids = _manifest_test_ids(manifest)
    if num_samples > len(test_ids):
        raise ValueError(
            f"num_samples={num_samples} exceeds held-out test population size {len(test_ids)}"
        )
    index_by_id = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}
    missing = [sample_id for sample_id in test_ids if sample_id not in index_by_id]
    if missing:
        raise ValueError(
            "source test split contains sample IDs absent from the current dataset: "
            f"{missing[:5]}"
        )
    test_indices = [index_by_id[sample_id] for sample_id in test_ids]

    edge_index = cartesian_4_neighbor_edge_index(dataset.grid_shape_2d)
    attention_edge_indices = _attention_topologies_from_geometry(
        source_cfg.geometry,
        edge_index=edge_index,
        num_nodes=dataset.grid_shape_2d[0] * dataset.grid_shape_2d[1],
    )
    collator = _NodeRegressionCollator(
        task,
        dataset.field_catalog,
        edge_index,
        attention_edge_indices=attention_edge_indices,
    )

    reference_loader = _loader(
        dataset,
        test_indices,
        batch_size=min(128, len(test_indices)),
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=source_seed,
    )
    reference_mean, reference_std, reference_channels = _reference_standardized_moments(
        reference_loader,
        standardizers=device_standardizers,
        device=device,
    )

    trajectory_loader = _loader(
        dataset,
        test_indices[:num_samples],
        batch_size=num_samples,
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=source_seed,
    )
    host_batch = next(iter(trajectory_loader))
    batch = _task_batch_to_device(host_batch, device=device, dtype=torch.float32)
    scaled = device_standardizers.transform(batch)
    if scaled.input_channels != reference_channels:
        raise ValueError("trajectory/reference loaders produced inconsistent channel semantics")

    probe_diffusion = task.make_validation_problem(scaled)
    model, _ = _instantiate_model(source_cfg.model, probe_diffusion, seed=source_seed)
    model = model.to(device=device, dtype=torch.float32)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("diffusion checkpoint does not contain model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    snapshot_timesteps = set(
        _diagnostic_timesteps(cfg.snapshot_timestep_fractions, task.timesteps)
    )
    sampling_keys = tuple(f"trajectory_{index:06d}" for index in range(scaled.num_graphs))
    recorder = _TrajectoryRecorder(
        model,
        task=task,
        eta=eta,
        reference_mean=reference_mean,
        reference_std=reference_std,
        channel_names=scaled.input_channels,
        snapshot_timesteps=snapshot_timesteps,
    ).to(device=device)
    recorder.eval()

    output_root = Path(str(cfg.output_root)).expanduser().resolve()
    output_name = cfg.output_name
    if output_name is None:
        name = f"{run_dir.name}__{checkpoint_path.stem}__eta{eta:g}"
    else:
        if not isinstance(output_name, str) or not output_name.strip():
            raise ValueError("output_name must be null or a non-empty string")
        name = output_name.strip()
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("output_name must be one directory name, not a path")
    output_dir = output_root / name
    if output_dir.exists():
        if not bool(cfg.overwrite):
            raise FileExistsError(
                f"trajectory diagnostic output already exists: {output_dir}; "
                "set overwrite=true explicitly"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output_dir / "diagnostic_config.yaml", resolve=True)

    final_state = task.sample_standardized(
        recorder,
        scaled,
        steps=task.timesteps,
        eta=eta,
        sampling_seed=sampling_seed,
        sampling_keys=sampling_keys,
    )
    recorder.finalize(final_state)

    _write_csv(output_dir / "trajectory_states.csv", recorder.state_rows)
    _write_csv(output_dir / "trajectory_transitions.csv", recorder.transition_rows)
    _write_csv(
        output_dir / "trajectory_mean.csv",
        [row for row in recorder.state_rows if row["channel"] == "__mean__"],
    )

    node_counts = tuple(
        int(value) for value in (scaled.ptr[1:] - scaled.ptr[:-1]).detach().cpu().tolist()
    )
    snapshot_payload = {
        "sampling_keys": sampling_keys,
        "template_test_sample_ids": tuple(scaled.source.sample_ids),
        "node_counts": node_counts,
        "channel_names": scaled.input_channels,
        "snapshot_timesteps": tuple(sorted(recorder.snapshots, reverse=True)),
        "snapshots": recorder.snapshots,
        "final_state_standardized": final_state.detach().cpu(),
        "sampling_seed": sampling_seed,
        "eta": eta,
        "sampling_steps": task.timesteps,
    }
    torch.save(snapshot_payload, output_dir / "trajectory_snapshots.pt")

    mean_rows = [row for row in recorder.state_rows if row["channel"] == "__mean__"]
    mean_transitions = [
        row for row in recorder.transition_rows if row["channel"] == "__mean__"
    ]
    max_state_ratio = max(mean_rows, key=lambda row: float(row["state_std_over_expected_q_std"]))
    max_x0_ratio = max(
        mean_rows,
        key=lambda row: float(row["x0_hat_std_over_reference_std"]),
    )
    max_update = max(mean_transitions, key=lambda row: float(row["update_rms"]))
    first_transition = next(
        row for row in mean_transitions if int(row["from_timestep"]) == task.timesteps
    )
    final_mean, final_std, final_rms, final_max = _pooled_channel_stats(final_state)

    sampler = task.sampler_name(steps=task.timesteps, eta=eta)
    summary = {
        "diagnostic": "diffusion_reverse_trajectory_v1",
        "source_run_name": run_dir.name,
        "source_run_dir": str(run_dir),
        "source_checkpoint": checkpoint_path.name,
        "source_best_epoch": source_summary.get("best_epoch"),
        "model": type(model).__name__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "prediction_type": "epsilon",
        "timesteps": task.timesteps,
        "noise_schedule": "cosine",
        "cosine_s": task.cosine_s,
        "sampler": sampler,
        "sampling_steps": task.timesteps,
        "sampling_eta": eta,
        "sampling_seed": sampling_seed,
        "is_exact_ancestral_ddpm": sampler == "ddpm_ancestral",
        "num_generated_trajectories": scaled.num_graphs,
        "sampling_keys_semantics": "independent deterministic generated sampling keys",
        "template_semantics": (
            "held-out samples provide only graph geometry/base conditioning and shape; "
            "their clean states do not initialize the reverse trajectory"
        ),
        "reference_calibration_semantics": (
            "held-out standardized clean-state moments are used only to compute expected q(x_t) "
            "diagnostic moments and never alter sampler state"
        ),
        "reference_population_size": len(test_ids),
        "channel_names": list(scaled.input_channels),
        "snapshot_timesteps": sorted(recorder.snapshots, reverse=True),
        "max_mean_state_std_over_expected_q_std": {
            "timestep": int(max_state_ratio["timestep"]),
            "value": float(max_state_ratio["state_std_over_expected_q_std"]),
        },
        "max_mean_x0_hat_std_over_reference_std": {
            "timestep": int(max_x0_ratio["timestep"]),
            "value": float(max_x0_ratio["x0_hat_std_over_reference_std"]),
        },
        "largest_mean_transition_update_rms": {
            "from_timestep": int(max_update["from_timestep"]),
            "to_timestep": int(max_update["to_timestep"]),
            "value": float(max_update["update_rms"]),
        },
        "first_reverse_transition": {
            "from_timestep": int(first_transition["from_timestep"]),
            "to_timestep": int(first_transition["to_timestep"]),
            "update_rms": float(first_transition["update_rms"]),
            "state_std_before": float(first_transition["from_state_std"]),
            "state_std_after": float(first_transition["to_state_std"]),
        },
        "final_standardized_state": {
            channel: {
                "mean": float(final_mean[index]),
                "std": float(final_std[index]),
                "rms": float(final_rms[index]),
                "max_abs": float(final_max[index]),
            }
            for index, channel in enumerate(scaled.input_channels)
        },
        "outputs": {
            "trajectory_states": "trajectory_states.csv",
            "trajectory_transitions": "trajectory_transitions.csv",
            "trajectory_mean": "trajectory_mean.csv",
            "trajectory_snapshots": "trajectory_snapshots.pt",
        },
        "device": str(device),
        "dtype": "float32",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


class _TrajectoryRecorder(torch.nn.Module):
    """Model wrapper that records states seen by the unchanged production sampler."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        task: DiffusionDenoisingTask,
        eta: float,
        reference_mean: torch.Tensor,
        reference_std: torch.Tensor,
        channel_names: tuple[str, ...],
        snapshot_timesteps: set[int],
    ) -> None:
        super().__init__()
        self.model = model
        self.task = task
        self.eta = float(eta)
        self.reference_mean = reference_mean.detach().cpu().to(torch.float64)
        self.reference_std = reference_std.detach().cpu().to(torch.float64)
        self.channel_names = channel_names
        self.snapshot_timesteps = snapshot_timesteps
        self.state_rows: list[dict[str, Any]] = []
        self.transition_rows: list[dict[str, Any]] = []
        self.snapshots: dict[int, dict[str, torch.Tensor]] = {}
        self._previous_state: torch.Tensor | None = None
        self._previous_timestep: int | None = None

    def forward(self, state: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        conditioning = kwargs.get("conditioning")
        if not isinstance(conditioning, torch.Tensor) or conditioning.ndim != 2:
            raise ValueError("trajectory recorder requires graph-level conditioning")
        normalized_times = conditioning[:, -1]
        if normalized_times.numel() == 0:
            raise ValueError("trajectory recorder received an empty graph batch")
        if not torch.allclose(
            normalized_times,
            normalized_times[0].expand_as(normalized_times),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError("trajectory recorder requires one common timestep across graphs")
        timestep = int(round(float(normalized_times[0].detach().cpu()) * self.task.timesteps))
        if timestep < 1 or timestep > self.task.timesteps:
            raise ValueError(f"recorded timestep must lie in [1, {self.task.timesteps}]")

        if self._previous_state is not None:
            if self._previous_timestep != timestep + 1:
                raise RuntimeError(
                    "trajectory recorder expects consecutive full reverse steps; "
                    f"got {self._previous_timestep} -> {timestep}"
                )
            self.transition_rows.extend(
                _transition_rows(
                    self._previous_state,
                    state,
                    from_timestep=self._previous_timestep,
                    to_timestep=timestep,
                    total_timesteps=self.task.timesteps,
                    channel_names=self.channel_names,
                )
            )

        epsilon_hat = self.model(state, **kwargs)
        if epsilon_hat.shape != state.shape:
            raise ValueError("wrapped diffusion model output must match sampler state shape")

        alpha_t = float(self.task._alpha_bar_cpu[timestep])
        alpha_previous = float(self.task._alpha_bar_cpu[timestep - 1])
        x0_hat = reconstruct_x0_from_epsilon(state, epsilon_hat, alpha_t)
        sigma, direction_scale = _reverse_schedule_scalars(
            alpha_t,
            alpha_previous,
            eta=self.eta,
            previous_is_clean=timestep == 1,
        )
        self.state_rows.extend(
            _state_rows(
                state,
                epsilon_hat,
                x0_hat,
                timestep=timestep,
                total_timesteps=self.task.timesteps,
                alpha_t=alpha_t,
                alpha_previous=alpha_previous,
                sigma=sigma,
                direction_scale=direction_scale,
                reference_mean=self.reference_mean,
                reference_std=self.reference_std,
                channel_names=self.channel_names,
            )
        )

        if timestep in self.snapshot_timesteps:
            self.snapshots[timestep] = {
                "state_standardized": state.detach().cpu(),
                "epsilon_hat": epsilon_hat.detach().cpu(),
                "x0_hat_standardized": x0_hat.detach().cpu(),
            }

        self._previous_state = state.detach().clone()
        self._previous_timestep = timestep
        return epsilon_hat

    def finalize(self, final_state: torch.Tensor) -> None:
        if self._previous_state is None or self._previous_timestep != 1:
            raise RuntimeError(
                "trajectory recorder did not observe the complete T -> 1 reverse path"
            )
        self.transition_rows.extend(
            _transition_rows(
                self._previous_state,
                final_state,
                from_timestep=1,
                to_timestep=0,
                total_timesteps=self.task.timesteps,
                channel_names=self.channel_names,
            )
        )
        self.snapshots[0] = {"state_standardized": final_state.detach().cpu()}


def _reference_standardized_moments(
    loader: Any,
    *,
    standardizers: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    total_sum: torch.Tensor | None = None
    total_square_sum: torch.Tensor | None = None
    total_nodes = 0
    channel_names: tuple[str, ...] | None = None
    for host_batch in loader:
        batch = _task_batch_to_device(host_batch, device=device, dtype=torch.float32)
        scaled = standardizers.transform(batch)
        if channel_names is None:
            channel_names = scaled.input_channels
        elif channel_names != scaled.input_channels:
            raise ValueError("reference loader produced inconsistent channel semantics")
        values = scaled.inputs.to(torch.float64)
        batch_sum = values.sum(dim=0)
        batch_square_sum = values.square().sum(dim=0)
        total_sum = batch_sum if total_sum is None else total_sum + batch_sum
        total_square_sum = (
            batch_square_sum
            if total_square_sum is None
            else total_square_sum + batch_square_sum
        )
        total_nodes += values.shape[0]

    if channel_names is None or total_sum is None or total_square_sum is None or total_nodes == 0:
        raise ValueError("reference diagnostic loader contains no samples")
    mean = total_sum / total_nodes
    variance = torch.clamp(total_square_sum / total_nodes - mean.square(), min=0.0)
    return mean.cpu(), torch.sqrt(variance).cpu(), channel_names


def _pooled_channel_stats(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if values.ndim != 2:
        raise ValueError("trajectory values must have shape [total_nodes, C]")
    as_float = values.to(torch.float64)
    return (
        as_float.mean(dim=0),
        as_float.std(dim=0, unbiased=False),
        torch.sqrt(as_float.square().mean(dim=0)),
        as_float.abs().amax(dim=0),
    )


def _expected_q_moments(
    reference_mean: torch.Tensor,
    reference_std: torch.Tensor,
    alpha_bar: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = float(alpha_bar)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha_bar must lie in [0, 1]")
    mean = math.sqrt(alpha) * reference_mean
    std = torch.sqrt(alpha * reference_std.square() + (1.0 - alpha))
    return mean, std


def _reverse_schedule_scalars(
    alpha_t: float,
    alpha_previous: float,
    *,
    eta: float,
    previous_is_clean: bool,
) -> tuple[float, float]:
    if previous_is_clean:
        return 0.0, 0.0
    variance_factor = (
        (1.0 - alpha_previous)
        / (1.0 - alpha_t)
        * (1.0 - alpha_t / alpha_previous)
    )
    sigma = float(eta) * math.sqrt(max(variance_factor, 0.0))
    direction_scale = math.sqrt(max(1.0 - alpha_previous - sigma * sigma, 0.0))
    return sigma, direction_scale


def _state_rows(
    state: torch.Tensor,
    epsilon_hat: torch.Tensor,
    x0_hat: torch.Tensor,
    *,
    timestep: int,
    total_timesteps: int,
    alpha_t: float,
    alpha_previous: float,
    sigma: float,
    direction_scale: float,
    reference_mean: torch.Tensor,
    reference_std: torch.Tensor,
    channel_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    state_mean, state_std, state_rms, state_max = _pooled_channel_stats(state)
    epsilon_mean, epsilon_std, epsilon_rms, epsilon_max = _pooled_channel_stats(epsilon_hat)
    x0_mean, x0_std, x0_rms, x0_max = _pooled_channel_stats(x0_hat)
    expected_mean, expected_std = _expected_q_moments(reference_mean, reference_std, alpha_t)
    snr = alpha_t / max(1.0 - alpha_t, torch.finfo(torch.float64).tiny)

    rows: list[dict[str, Any]] = []
    for index, channel in enumerate(channel_names):
        q_std = float(expected_std[index])
        clean_std = float(reference_std[index])
        rows.append(
            {
                "timestep": timestep,
                "previous_timestep": timestep - 1,
                "t_over_T": timestep / total_timesteps,
                "alpha_bar": alpha_t,
                "alpha_bar_previous": alpha_previous,
                "snr": snr,
                "log10_snr": math.log10(snr) if snr > 0.0 else float("-inf"),
                "sigma": sigma,
                "direction_scale": direction_scale,
                "channel": channel,
                "expected_q_mean": float(expected_mean[index]),
                "expected_q_std": q_std,
                "state_mean": float(state_mean[index]),
                "state_std": float(state_std[index]),
                "state_rms": float(state_rms[index]),
                "state_max_abs": float(state_max[index]),
                "state_mean_error_over_expected_q_std": (
                    abs(float(state_mean[index]) - float(expected_mean[index])) / q_std
                    if q_std > 0.0
                    else float("nan")
                ),
                "state_std_over_expected_q_std": (
                    float(state_std[index]) / q_std if q_std > 0.0 else float("nan")
                ),
                "epsilon_hat_mean": float(epsilon_mean[index]),
                "epsilon_hat_std": float(epsilon_std[index]),
                "epsilon_hat_rms": float(epsilon_rms[index]),
                "epsilon_hat_max_abs": float(epsilon_max[index]),
                "x0_hat_mean": float(x0_mean[index]),
                "x0_hat_std": float(x0_std[index]),
                "x0_hat_rms": float(x0_rms[index]),
                "x0_hat_max_abs": float(x0_max[index]),
                "x0_hat_mean_error_over_reference_std": (
                    abs(float(x0_mean[index]) - float(reference_mean[index])) / clean_std
                    if clean_std > 0.0
                    else float("nan")
                ),
                "x0_hat_std_over_reference_std": (
                    float(x0_std[index]) / clean_std if clean_std > 0.0 else float("nan")
                ),
            }
        )
    rows.append(_mean_state_row(rows))
    return rows


def _mean_state_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    fixed = {
        "timestep",
        "previous_timestep",
        "t_over_T",
        "alpha_bar",
        "alpha_bar_previous",
        "snr",
        "log10_snr",
        "sigma",
        "direction_scale",
        "channel",
    }
    result = {key: first[key] for key in fixed if key != "channel"}
    result["channel"] = "__mean__"
    for key in first:
        if key in fixed:
            continue
        values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        result[key] = sum(values) / len(values) if values else float("nan")
    return result


def _transition_rows(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    from_timestep: int,
    to_timestep: int,
    total_timesteps: int,
    channel_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    if before.shape != after.shape:
        raise ValueError("trajectory transition states must have identical shape")
    _, before_std, before_rms, before_max = _pooled_channel_stats(before)
    _, after_std, after_rms, after_max = _pooled_channel_stats(after)
    _, _, update_rms, update_max = _pooled_channel_stats(after - before)
    rows: list[dict[str, Any]] = []
    for index, channel in enumerate(channel_names):
        rows.append(
            {
                "from_timestep": from_timestep,
                "to_timestep": to_timestep,
                "from_t_over_T": from_timestep / total_timesteps,
                "to_t_over_T": to_timestep / total_timesteps,
                "channel": channel,
                "from_state_std": float(before_std[index]),
                "to_state_std": float(after_std[index]),
                "from_state_rms": float(before_rms[index]),
                "to_state_rms": float(after_rms[index]),
                "from_state_max_abs": float(before_max[index]),
                "to_state_max_abs": float(after_max[index]),
                "update_rms": float(update_rms[index]),
                "update_max_abs": float(update_max[index]),
                "to_std_over_from_std": (
                    float(after_std[index] / before_std[index])
                    if float(before_std[index]) > 0.0
                    else float("nan")
                ),
            }
        )
    rows.append(_mean_transition_row(rows))
    return rows


def _mean_transition_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    fixed = {
        "from_timestep",
        "to_timestep",
        "from_t_over_T",
        "to_t_over_T",
        "channel",
    }
    result = {key: first[key] for key in fixed if key != "channel"}
    result["channel"] = "__mean__"
    for key in first:
        if key in fixed:
            continue
        values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        result[key] = sum(values) / len(values) if values else float("nan")
    return result


if __name__ == "__main__":
    main()
