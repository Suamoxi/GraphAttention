"""Diagnose DDPM sensitivity to the reverse-chain starting timestep."""

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
from graph_attention.evaluation.diffusion_diagnostics import (
    graph_channel_moments,
    graph_channel_mse,
)
from graph_attention.geometry import cartesian_4_neighbor_edge_index
from graph_attention.tasks import DiffusionDenoisingTask
from graph_attention.tasks.diffusion import (
    _model_epsilon,
    _randn_by_graph,
    _sample_generator,
)

_EXACT_FORWARD = "exact_forward_marginal"
_GAUSSIAN_RESTART = "gaussian_restart"
_SUPPORTED_MODES = {_EXACT_FORWARD, _GAUSSIAN_RESTART}


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="diagnose_diffusion_restart_sweep",
)
def main(cfg: DictConfig) -> None:
    summary = run_restart_sweep(cfg)
    print(json.dumps(summary, indent=2, allow_nan=False))


@torch.no_grad()
def run_restart_sweep(cfg: DictConfig) -> dict[str, Any]:
    """Run exact-forward and Gaussian reverse restarts over selected timesteps."""

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
        raise RuntimeError("CUDA restart diagnostics requested but CUDA is unavailable")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("restart diagnostics are intentionally single-process")

    num_workers = _positive_int(cfg.num_workers, "num_workers", allow_zero=True)
    num_samples = _positive_int(cfg.num_samples, "num_samples")
    diagnostic_seed = _positive_int(
        cfg.diagnostic_seed,
        "diagnostic_seed",
        allow_zero=True,
    )
    reverse_seed = _positive_int(cfg.reverse_seed, "reverse_seed", allow_zero=True)
    modes = _validated_modes(cfg.modes)

    dataset = instantiate(source_cfg.data)
    if not isinstance(dataset, PrecomputedSlicePTDataset):
        raise TypeError("source diffusion run must use data=hit_slice_pt")
    task = instantiate(source_cfg.task)
    if not isinstance(task, DiffusionDenoisingTask):
        raise TypeError("source diffusion run must use task=hit_diffusion")

    start_timesteps = _diagnostic_timesteps(
        cfg.start_timestep_fractions,
        task.timesteps,
    )

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
    reference_mean, reference_std, channel_names = _reference_equal_sample_moments(
        reference_loader,
        standardizers=device_standardizers,
        device=device,
    )

    diagnostic_loader = _loader(
        dataset,
        test_indices[:num_samples],
        batch_size=num_samples,
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=source_seed,
    )
    host_batch = next(iter(diagnostic_loader))
    batch = _task_batch_to_device(host_batch, device=device, dtype=torch.float32)
    scaled = device_standardizers.transform(batch)
    if scaled.input_channels != channel_names:
        raise ValueError("diagnostic/reference loaders produced inconsistent channel semantics")

    probe_diffusion = task.make_validation_problem(scaled)
    model, _ = _instantiate_model(source_cfg.model, probe_diffusion, seed=source_seed)
    model = model.to(device=device, dtype=torch.float32)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("diffusion checkpoint does not contain model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    sample_keys = tuple(scaled.source.sample_ids)
    forward_noise = _keyed_noise(
        scaled,
        seed=diagnostic_seed,
        purpose="restart_forward_marginal",
        sample_keys=sample_keys,
    )
    gaussian_restart_state = _keyed_noise(
        scaled,
        seed=diagnostic_seed,
        purpose="restart_gaussian_state",
        sample_keys=sample_keys,
    )

    output_root = Path(str(cfg.output_root)).expanduser().resolve()
    output_name = cfg.output_name
    if output_name is None:
        name = f"{run_dir.name}__{checkpoint_path.stem}__restart_sweep"
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
                f"restart diagnostic output already exists: {output_dir}; "
                "set overwrite=true explicitly"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output_dir / "diagnostic_config.yaml", resolve=True)

    rows: list[dict[str, Any]] = []
    snapshots: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for mode in modes:
        snapshots[mode] = {}
        for start_timestep in start_timesteps:
            if mode == _EXACT_FORWARD:
                initial_state = _forward_marginal_state(
                    scaled.inputs,
                    forward_noise,
                    task=task,
                    timestep=start_timestep,
                )
            else:
                initial_state = gaussian_restart_state.clone()

            final_state = _reverse_ancestral_from_state(
                model,
                task=task,
                batch=scaled,
                initial_state=initial_state,
                start_timestep=start_timestep,
                reverse_seed=reverse_seed,
                sample_keys=sample_keys,
            )
            rows.extend(
                _restart_rows(
                    mode=mode,
                    start_timestep=start_timestep,
                    total_timesteps=task.timesteps,
                    initial_state=initial_state,
                    final_state=final_state,
                    clean_state=scaled.inputs,
                    ptr=scaled.ptr,
                    reference_mean=reference_mean,
                    reference_std=reference_std,
                    channel_names=channel_names,
                    alpha_bar=float(task._alpha_bar_cpu[start_timestep]),
                )
            )
            snapshots[mode][start_timestep] = {
                "initial_state_standardized": initial_state.detach().cpu(),
                "final_state_standardized": final_state.detach().cpu(),
            }

    _write_csv(output_dir / "restart_sweep.csv", rows)
    torch.save(
        {
            "source_test_sample_ids": sample_keys,
            "channel_names": channel_names,
            "start_timesteps": tuple(start_timesteps),
            "modes": tuple(modes),
            "clean_state_standardized": scaled.inputs.detach().cpu(),
            "forward_noise": forward_noise.detach().cpu(),
            "gaussian_restart_state": gaussian_restart_state.detach().cpu(),
            "snapshots": snapshots,
            "diagnostic_seed": diagnostic_seed,
            "reverse_seed": reverse_seed,
        },
        output_dir / "restart_samples.pt",
    )

    mean_rows = [row for row in rows if row["channel"] == "__mean__"]
    exact_rows = [row for row in mean_rows if row["mode"] == _EXACT_FORWARD]
    gaussian_rows = [row for row in mean_rows if row["mode"] == _GAUSSIAN_RESTART]
    summary = {
        "diagnostic": "diffusion_restart_sweep_v1",
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
        "start_timesteps": start_timesteps,
        "start_timestep_fractions": [value / task.timesteps for value in start_timesteps],
        "modes": modes,
        "num_diagnostic_samples": scaled.num_graphs,
        "reference_population_size": len(test_ids),
        "channel_names": list(channel_names),
        "diagnostic_seed": diagnostic_seed,
        "reverse_seed": reverse_seed,
        "randomness_semantics": {
            "exact_forward_marginal": (
                "one deterministic Gaussian field per held-out sample is reused for every "
                "start timestep when constructing exact q(x_t|x_0) states"
            ),
            "gaussian_restart": (
                "one deterministic N(0,I) field per held-out sample is reused for every "
                "start timestep; this is diagnostic-only below T"
            ),
            "reverse_transition_noise": (
                "ancestral Gaussian noise is keyed by sample ID and absolute reverse timestep, "
                "so shared transitions receive identical perturbations across restart runs"
            ),
            "production_equivalence": (
                "each reverse transition follows the same ancestral DDPM law as production; "
                "the keyed-per-timestep random stream is chosen for controlled comparison and "
                "is not bitwise identical to the production sequential RNG stream"
            ),
        },
        "interpretation": {
            "exact_forward_marginal": (
                "tests whether the learned reverse dynamics can reconstruct held-out clean "
                "samples when started from a mathematically correct forward marginal"
            ),
            "gaussian_restart": (
                "tests whether omitting the terminal reverse steps stabilizes unconditional "
                "generation; starts below T are not exact samples from q(x_t)"
            ),
            "paired_clean_mse": (
                "available only for exact_forward_marginal and should not be interpreted as "
                "an unconditional generation metric"
            ),
        },
        "best_exact_forward_by_paired_mse": _best_row_summary(
            exact_rows,
            "paired_clean_mse",
        ),
        "best_gaussian_by_std_ratio_error": _best_row_summary(
            gaussian_rows,
            "final_std_ratio_error",
        ),
        "outputs": {
            "restart_sweep": "restart_sweep.csv",
            "restart_samples": "restart_samples.pt",
        },
        "device": str(device),
        "dtype": "float32",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def _validated_modes(values: Any) -> list[str]:
    modes = [str(value) for value in values]
    if not modes:
        raise ValueError("modes must be non-empty")
    if len(set(modes)) != len(modes):
        raise ValueError("modes must not contain duplicates")
    unsupported = sorted(set(modes).difference(_SUPPORTED_MODES))
    if unsupported:
        raise ValueError(f"unsupported restart diagnostic modes: {unsupported}")
    return modes


def _keyed_noise(
    batch: Any,
    *,
    seed: int,
    purpose: str,
    sample_keys: tuple[str, ...],
) -> torch.Tensor:
    generators = [
        _sample_generator(batch.inputs.device, seed, purpose, key)
        for key in sample_keys
    ]
    return _randn_by_graph(batch, generators)


def _forward_marginal_state(
    clean_state: torch.Tensor,
    noise: torch.Tensor,
    *,
    task: DiffusionDenoisingTask,
    timestep: int,
) -> torch.Tensor:
    if clean_state.shape != noise.shape:
        raise ValueError("clean state and forward noise must have identical shape")
    if timestep < 1 or timestep > task.timesteps:
        raise ValueError(f"timestep must lie in [1, {task.timesteps}]")
    alpha = task._alpha_bar_for(clean_state)[timestep]
    return torch.sqrt(alpha) * clean_state + torch.sqrt(1.0 - alpha) * noise


@torch.no_grad()
def _reverse_ancestral_from_state(
    model: torch.nn.Module,
    *,
    task: DiffusionDenoisingTask,
    batch: Any,
    initial_state: torch.Tensor,
    start_timestep: int,
    reverse_seed: int,
    sample_keys: tuple[str, ...],
) -> torch.Tensor:
    if start_timestep < 1 or start_timestep > task.timesteps:
        raise ValueError(f"start_timestep must lie in [1, {task.timesteps}]")
    if initial_state.shape != batch.inputs.shape:
        raise ValueError("initial_state must match packed batch input shape")
    if len(sample_keys) != batch.num_graphs:
        raise ValueError("sample_keys must contain one key per graph")

    state = initial_state.clone()
    alpha_bar = task._alpha_bar_for(state)
    for timestep in range(start_timestep, 0, -1):
        graph_timesteps = torch.full(
            (batch.num_graphs,),
            timestep,
            device=state.device,
            dtype=torch.long,
        )
        epsilon_hat = _model_epsilon(
            model,
            batch,
            state,
            task._normalized_time(graph_timesteps, state.dtype),
        )
        alpha_t = alpha_bar[timestep]
        alpha_previous = alpha_bar[timestep - 1]
        x0_hat = (
            state - torch.sqrt(1.0 - alpha_t) * epsilon_hat
        ) / torch.sqrt(alpha_t)
        if timestep == 1:
            state = x0_hat
            continue

        variance_factor = (
            (1.0 - alpha_previous)
            / (1.0 - alpha_t)
            * (1.0 - alpha_t / alpha_previous)
        )
        sigma = torch.sqrt(torch.clamp(variance_factor, min=0.0))
        direction_scale = torch.sqrt(
            torch.clamp(1.0 - alpha_previous - sigma**2, min=0.0)
        )
        transition_noise = _keyed_noise(
            batch,
            seed=reverse_seed,
            purpose=f"restart_reverse_t{timestep}",
            sample_keys=sample_keys,
        )
        state = (
            torch.sqrt(alpha_previous) * x0_hat
            + direction_scale * epsilon_hat
            + sigma * transition_noise
        )

    if not torch.isfinite(state).all():
        raise ValueError("restart reverse process produced NaN or Inf values")
    return state


def _reference_equal_sample_moments(
    loader: Any,
    *,
    standardizers: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    graph_means: list[torch.Tensor] = []
    graph_second_moments: list[torch.Tensor] = []
    channel_names: tuple[str, ...] | None = None
    for host_batch in loader:
        batch = _task_batch_to_device(host_batch, device=device, dtype=torch.float32)
        scaled = standardizers.transform(batch)
        if channel_names is None:
            channel_names = scaled.input_channels
        elif channel_names != scaled.input_channels:
            raise ValueError("reference loader produced inconsistent channel semantics")
        mean, std, _ = graph_channel_moments(scaled.inputs, scaled.ptr)
        graph_means.append(mean.to(torch.float64))
        graph_second_moments.append((std.square() + mean.square()).to(torch.float64))

    if channel_names is None or not graph_means:
        raise ValueError("reference diagnostic loader contains no samples")
    means = torch.cat(graph_means, dim=0)
    second_moments = torch.cat(graph_second_moments, dim=0)
    mean = means.mean(dim=0)
    variance = torch.clamp(second_moments.mean(dim=0) - mean.square(), min=0.0)
    return mean.cpu(), torch.sqrt(variance).cpu(), channel_names


def _equal_sample_stats(
    values: torch.Tensor,
    ptr: torch.Tensor,
) -> dict[str, torch.Tensor]:
    means, stds, maxima = graph_channel_moments(values, ptr)
    second_moments = stds.square() + means.square()
    population_mean = means.mean(dim=0)
    population_second = second_moments.mean(dim=0)
    return {
        "mean": population_mean,
        "std": torch.sqrt(torch.clamp(population_second - population_mean.square(), min=0.0)),
        "rms": torch.sqrt(torch.clamp(population_second, min=0.0)),
        "mean_sample_max_abs": maxima.mean(dim=0),
        "max_abs": maxima.amax(dim=0),
    }


def _restart_rows(
    *,
    mode: str,
    start_timestep: int,
    total_timesteps: int,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    clean_state: torch.Tensor,
    ptr: torch.Tensor,
    reference_mean: torch.Tensor,
    reference_std: torch.Tensor,
    channel_names: tuple[str, ...],
    alpha_bar: float,
) -> list[dict[str, Any]]:
    initial = _equal_sample_stats(initial_state, ptr)
    final = _equal_sample_stats(final_state, ptr)
    expected_mean = math.sqrt(alpha_bar) * reference_mean
    expected_std = torch.sqrt(
        alpha_bar * reference_std.square() + (1.0 - alpha_bar)
    )
    paired_mse = None
    if mode == _EXACT_FORWARD:
        paired_mse = graph_channel_mse(final_state, clean_state, ptr).mean(dim=0)

    snr = alpha_bar / max(1.0 - alpha_bar, torch.finfo(torch.float64).tiny)
    rows: list[dict[str, Any]] = []
    for index, channel in enumerate(channel_names):
        ref_std = float(reference_std[index])
        q_std = float(expected_std[index])
        row = {
            "mode": mode,
            "start_timestep": start_timestep,
            "start_t_over_T": start_timestep / total_timesteps,
            "alpha_bar": alpha_bar,
            "snr": snr,
            "log10_snr": math.log10(snr) if snr > 0.0 else float("-inf"),
            "channel": channel,
            "expected_q_mean": float(expected_mean[index]),
            "expected_q_std": q_std,
            "initial_mean": float(initial["mean"][index]),
            "initial_std": float(initial["std"][index]),
            "initial_std_over_expected_q_std": (
                float(initial["std"][index]) / q_std if q_std > 0.0 else float("nan")
            ),
            "initial_mean_error_over_expected_q_std": (
                abs(float(initial["mean"][index]) - float(expected_mean[index])) / q_std
                if q_std > 0.0
                else float("nan")
            ),
            "final_mean": float(final["mean"][index]),
            "final_std": float(final["std"][index]),
            "final_rms": float(final["rms"][index]),
            "final_mean_sample_max_abs": float(final["mean_sample_max_abs"][index]),
            "final_max_abs": float(final["max_abs"][index]),
            "final_mean_error_over_reference_std": (
                abs(float(final["mean"][index]) - float(reference_mean[index])) / ref_std
                if ref_std > 0.0
                else float("nan")
            ),
            "final_std_over_reference_std": (
                float(final["std"][index]) / ref_std if ref_std > 0.0 else float("nan")
            ),
            "final_std_ratio_error": (
                abs(float(final["std"][index]) / ref_std - 1.0)
                if ref_std > 0.0
                else float("nan")
            ),
            "paired_clean_mse": (
                float(paired_mse[index]) if paired_mse is not None else float("nan")
            ),
        }
        rows.append(row)
    rows.append(_mean_restart_row(rows))
    return rows


def _mean_restart_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    fixed = {
        "mode",
        "start_timestep",
        "start_t_over_T",
        "alpha_bar",
        "snr",
        "log10_snr",
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


def _best_row_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    finite = [row for row in rows if math.isfinite(float(row[metric]))]
    if not finite:
        return None
    best = min(finite, key=lambda row: float(row[metric]))
    return {
        "start_timestep": int(best["start_timestep"]),
        "start_t_over_T": float(best["start_t_over_T"]),
        "metric": metric,
        "value": float(best[metric]),
    }


if __name__ == "__main__":
    main()
