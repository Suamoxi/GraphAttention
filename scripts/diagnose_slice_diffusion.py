"""Diagnose epsilon-prediction diffusion error as a function of noise timestep."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from graph_attention.data import PrecomputedSlicePTDataset
from graph_attention.evaluation.diffusion_diagnostics import (
    epsilon_to_x0_mse_factor,
    graph_channel_moments,
    graph_channel_mse,
    reconstruct_x0_from_epsilon,
)
from graph_attention.geometry import cartesian_4_neighbor_edge_index
from graph_attention.tasks import DiffusionDenoisingTask
from graph_attention.tasks.diffusion import _model_epsilon, _randn_by_graph, _sample_generator
from scripts.generate_slice_diffusion import _load_standardizers, _manifest_test_ids
from scripts.train_slice_ablation import (
    _NodeRegressionCollator,
    _attention_topologies_from_geometry,
    _instantiate_model,
    _loader,
    _positive_int,
    _task_batch_to_device,
)
from scripts.train_slice_diffusion import _validate_diffusion_standardizers


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="diagnose_slice_diffusion",
)
def main(cfg: DictConfig) -> None:
    summary = run_diffusion_diagnostics(cfg)
    print(json.dumps(summary, indent=2, allow_nan=False))


@torch.no_grad()
def run_diffusion_diagnostics(cfg: DictConfig) -> dict[str, Any]:
    """Evaluate denoiser and reconstructed x0 error at fixed diffusion times."""

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
    seed = _positive_int(source_cfg.seed, "source seed", allow_zero=True)

    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diffusion diagnostics requested but CUDA is unavailable")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("diffusion diagnostics are intentionally single-process")

    batch_size = _positive_int(cfg.batch_size, "batch_size")
    num_workers = _positive_int(cfg.num_workers, "num_workers", allow_zero=True)
    diagnostic_seed = _positive_int(cfg.diagnostic_seed, "diagnostic_seed", allow_zero=True)

    dataset = instantiate(source_cfg.data)
    if not isinstance(dataset, PrecomputedSlicePTDataset):
        raise TypeError("source diffusion run must use data=hit_slice_pt")
    task = instantiate(source_cfg.task)
    if not isinstance(task, DiffusionDenoisingTask):
        raise TypeError("source diffusion run must use task=hit_diffusion")

    timesteps = _diagnostic_timesteps(cfg.timestep_fractions, task.timesteps)

    standardizers = _load_standardizers(standardizers_path)
    _validate_diffusion_standardizers(standardizers)
    if standardizers.physical_nondimensionalization != task.physical_nondimensionalization:
        raise ValueError("saved standardizers and source task disagree on nondimensionalization")
    device_standardizers = standardizers.to(device=device, dtype=torch.float32)

    test_ids = _manifest_test_ids(manifest)
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
    test_loader = _loader(
        dataset,
        test_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=seed,
    )

    probe = next(iter(test_loader))
    probe_scaled = standardizers.transform(probe)
    probe_diffusion = task.make_validation_problem(probe_scaled)
    model, _ = _instantiate_model(source_cfg.model, probe_diffusion, seed=seed)
    model = model.to(device=device, dtype=torch.float32)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("diffusion checkpoint does not contain model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    output_root = Path(str(cfg.output_root)).expanduser().resolve()
    output_name = cfg.output_name
    if output_name is None:
        name = f"{run_dir.name}__{checkpoint_path.stem}"
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
                f"diffusion diagnostic output already exists: {output_dir}; "
                "set overwrite=true explicitly"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output_dir / "diagnostic_config.yaml", resolve=True)

    # Lists contain one [graphs, channels] tensor per loader batch.  Keeping the
    # graph axis explicit preserves equal-sample weighting even if future test
    # batches contain different graph sizes.
    collected: dict[int, dict[str, list[torch.Tensor]]] = {
        timestep: defaultdict(list) for timestep in timesteps
    }
    channel_names: tuple[str, ...] | None = None
    num_graphs = 0

    for host_batch in test_loader:
        batch = _task_batch_to_device(host_batch, device=device, dtype=torch.float32)
        scaled = device_standardizers.transform(batch)
        if channel_names is None:
            channel_names = scaled.input_channels
        elif channel_names != scaled.input_channels:
            raise ValueError("test batches produced inconsistent state-channel semantics")

        generators = [
            _sample_generator(
                device,
                diagnostic_seed,
                "diagnostic_forward",
                sample_id,
            )
            for sample_id in scaled.source.sample_ids
        ]
        noise = _randn_by_graph(scaled, generators)
        num_graphs += scaled.num_graphs

        clean_mean, clean_std, clean_max = graph_channel_moments(scaled.inputs, scaled.ptr)
        noise_mean, noise_std, noise_max = graph_channel_moments(noise, scaled.ptr)

        for timestep in timesteps:
            graph_timesteps = torch.full(
                (scaled.num_graphs,),
                timestep,
                device=device,
                dtype=torch.long,
            )
            problem = task._diffusion_problem(scaled, graph_timesteps, noise)
            epsilon_hat = _model_epsilon(
                model,
                scaled,
                problem.inputs,
                task._normalized_time(graph_timesteps, scaled.inputs.dtype),
            )
            alpha_t = task._alpha_bar_for(scaled.inputs)[timestep]
            x0_hat = reconstruct_x0_from_epsilon(problem.inputs, epsilon_hat, alpha_t)
            x0_oracle = reconstruct_x0_from_epsilon(problem.inputs, noise, alpha_t)

            epsilon_mse = graph_channel_mse(epsilon_hat, noise, scaled.ptr)
            x0_mse = graph_channel_mse(x0_hat, scaled.inputs, scaled.ptr)
            oracle_x0_mse = graph_channel_mse(x0_oracle, scaled.inputs, scaled.ptr)
            noisy_mean, noisy_std, noisy_max = graph_channel_moments(problem.inputs, scaled.ptr)
            epsilon_mean, epsilon_std, epsilon_max = graph_channel_moments(epsilon_hat, scaled.ptr)
            x0_mean, x0_std, x0_max = graph_channel_moments(x0_hat, scaled.ptr)

            target = collected[timestep]
            for key, tensor in (
                ("epsilon_mse", epsilon_mse),
                ("x0_mse", x0_mse),
                ("oracle_x0_mse", oracle_x0_mse),
                ("clean_mean", clean_mean),
                ("clean_std", clean_std),
                ("clean_max", clean_max),
                ("noise_mean", noise_mean),
                ("noise_std", noise_std),
                ("noise_max", noise_max),
                ("noisy_mean", noisy_mean),
                ("noisy_std", noisy_std),
                ("noisy_max", noisy_max),
                ("epsilon_hat_mean", epsilon_mean),
                ("epsilon_hat_std", epsilon_std),
                ("epsilon_hat_max", epsilon_max),
                ("x0_hat_mean", x0_mean),
                ("x0_hat_std", x0_std),
                ("x0_hat_max", x0_max),
            ):
                target[key].append(tensor.detach().cpu())

    if channel_names is None:
        raise ValueError("test diagnostic loader contains no samples")

    alpha_bar = task._alpha_bar_cpu
    rows: list[dict[str, Any]] = []
    for timestep in timesteps:
        metrics = {
            name: torch.cat(parts, dim=0).to(torch.float64)
            for name, parts in collected[timestep].items()
        }
        if metrics["epsilon_mse"].shape[0] != num_graphs:
            raise RuntimeError("diagnostic graph counts are inconsistent")
        alpha = float(alpha_bar[timestep])
        factor = epsilon_to_x0_mse_factor(alpha)
        snr = alpha / max(1.0 - alpha, torch.finfo(torch.float64).tiny)
        amplitude_factor = math.sqrt(factor)

        channel_rows = []
        for channel, channel_name in enumerate(channel_names):
            channel_rows.append(
                _diagnostic_row(
                    timestep=timestep,
                    total_timesteps=task.timesteps,
                    alpha=alpha,
                    snr=snr,
                    mse_factor=factor,
                    amplitude_factor=amplitude_factor,
                    channel=channel_name,
                    metrics=metrics,
                    channel_index=channel,
                )
            )
        rows.extend(channel_rows)
        rows.append(
            _mean_channel_row(
                timestep=timestep,
                total_timesteps=task.timesteps,
                alpha=alpha,
                snr=snr,
                mse_factor=factor,
                amplitude_factor=amplitude_factor,
                channel_rows=channel_rows,
            )
        )

    _write_csv(output_dir / "timestep_diagnostics.csv", rows)

    mean_rows = [row for row in rows if row["channel"] == "__mean__"]
    worst_epsilon = max(mean_rows, key=lambda row: float(row["epsilon_mse"]))
    worst_x0 = max(mean_rows, key=lambda row: float(row["x0_mse"]))
    max_oracle = max(float(row["oracle_x0_mse"]) for row in mean_rows)
    summary = {
        "diagnostic": "diffusion_fixed_timestep_v1",
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
        "diagnostic_timesteps": timesteps,
        "diagnostic_timestep_fractions": [value / task.timesteps for value in timesteps],
        "diagnostic_seed": diagnostic_seed,
        "noise_semantics": (
            "one deterministic Gaussian field per test sample, reused across all diagnostic timesteps"
        ),
        "sample_weighting": "equal physical samples before channel averaging",
        "num_test_samples": num_graphs,
        "channel_names": list(channel_names),
        "worst_mean_epsilon_mse": {
            "timestep": int(worst_epsilon["timestep"]),
            "t_over_T": float(worst_epsilon["t_over_T"]),
            "value": float(worst_epsilon["epsilon_mse"]),
        },
        "worst_mean_x0_mse": {
            "timestep": int(worst_x0["timestep"]),
            "t_over_T": float(worst_x0["t_over_T"]),
            "value": float(worst_x0["x0_mse"]),
        },
        "max_oracle_x0_mse": max_oracle,
        "interpretation": {
            "epsilon_mse": "direct denoiser error in standardized training space",
            "epsilon_to_x0_mse_factor": "exact analytical multiplier (1-alpha_bar)/alpha_bar",
            "x0_mse": "clean-state reconstruction error implied by epsilon prediction",
            "oracle_x0_mse": (
                "reconstruction using the exact injected noise; should remain near numerical precision"
            ),
            "x0_mse_over_expected_from_epsilon": (
                "should be approximately one; verifies the analytical amplification identity"
            ),
        },
        "outputs": {"timestep_diagnostics": "timestep_diagnostics.csv"},
        "device": str(device),
        "dtype": "float32",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def _diagnostic_timesteps(values: Any, total_timesteps: int) -> list[int]:
    fractions = [float(value) for value in values]
    if not fractions:
        raise ValueError("timestep_fractions must be non-empty")
    if any(value <= 0.0 or value > 1.0 for value in fractions):
        raise ValueError("timestep_fractions must lie in (0, 1]")
    timesteps: list[int] = []
    for fraction in fractions:
        timestep = min(total_timesteps, max(1, round(fraction * total_timesteps)))
        if timestep not in timesteps:
            timesteps.append(timestep)
    return timesteps


def _diagnostic_row(
    *,
    timestep: int,
    total_timesteps: int,
    alpha: float,
    snr: float,
    mse_factor: float,
    amplitude_factor: float,
    channel: str,
    metrics: dict[str, torch.Tensor],
    channel_index: int,
) -> dict[str, Any]:
    epsilon_mse = float(metrics["epsilon_mse"][:, channel_index].mean())
    x0_mse = float(metrics["x0_mse"][:, channel_index].mean())
    oracle_x0_mse = float(metrics["oracle_x0_mse"][:, channel_index].mean())
    expected_x0_mse = mse_factor * epsilon_mse
    clean_std = float(metrics["clean_std"][:, channel_index].mean())
    x0_std = float(metrics["x0_hat_std"][:, channel_index].mean())
    return {
        "timestep": timestep,
        "t_over_T": timestep / total_timesteps,
        "alpha_bar": alpha,
        "snr": snr,
        "epsilon_to_x0_amplitude_factor": amplitude_factor,
        "epsilon_to_x0_mse_factor": mse_factor,
        "channel": channel,
        "epsilon_mse": epsilon_mse,
        "epsilon_rmse": math.sqrt(max(epsilon_mse, 0.0)),
        "x0_mse": x0_mse,
        "x0_rmse": math.sqrt(max(x0_mse, 0.0)),
        "expected_x0_mse_from_epsilon": expected_x0_mse,
        "x0_mse_over_expected_from_epsilon": (
            x0_mse / expected_x0_mse if expected_x0_mse > 0.0 else float("nan")
        ),
        "oracle_x0_mse": oracle_x0_mse,
        "clean_mean": float(metrics["clean_mean"][:, channel_index].mean()),
        "clean_std": clean_std,
        "clean_mean_max_abs_per_sample": float(metrics["clean_max"][:, channel_index].mean()),
        "noise_mean": float(metrics["noise_mean"][:, channel_index].mean()),
        "noise_std": float(metrics["noise_std"][:, channel_index].mean()),
        "noise_mean_max_abs_per_sample": float(metrics["noise_max"][:, channel_index].mean()),
        "noisy_state_mean": float(metrics["noisy_mean"][:, channel_index].mean()),
        "noisy_state_std": float(metrics["noisy_std"][:, channel_index].mean()),
        "noisy_state_mean_max_abs_per_sample": float(metrics["noisy_max"][:, channel_index].mean()),
        "epsilon_hat_mean": float(metrics["epsilon_hat_mean"][:, channel_index].mean()),
        "epsilon_hat_std": float(metrics["epsilon_hat_std"][:, channel_index].mean()),
        "epsilon_hat_mean_max_abs_per_sample": float(metrics["epsilon_hat_max"][:, channel_index].mean()),
        "x0_hat_mean": float(metrics["x0_hat_mean"][:, channel_index].mean()),
        "x0_hat_std": x0_std,
        "x0_std_ratio": x0_std / clean_std if clean_std > 0.0 else float("nan"),
        "x0_hat_mean_max_abs_per_sample": float(metrics["x0_hat_max"][:, channel_index].mean()),
    }


def _mean_channel_row(
    *,
    timestep: int,
    total_timesteps: int,
    alpha: float,
    snr: float,
    mse_factor: float,
    amplitude_factor: float,
    channel_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = [
        key
        for key in channel_rows[0]
        if key
        not in {
            "timestep",
            "t_over_T",
            "alpha_bar",
            "snr",
            "epsilon_to_x0_amplitude_factor",
            "epsilon_to_x0_mse_factor",
            "channel",
        }
    ]
    row: dict[str, Any] = {
        "timestep": timestep,
        "t_over_T": timestep / total_timesteps,
        "alpha_bar": alpha,
        "snr": snr,
        "epsilon_to_x0_amplitude_factor": amplitude_factor,
        "epsilon_to_x0_mse_factor": mse_factor,
        "channel": "__mean__",
    }
    for key in keys:
        values = [float(item[key]) for item in channel_rows if math.isfinite(float(item[key]))]
        row[key] = sum(values) / len(values) if values else float("nan")
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _existing_directory(value: object, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a filesystem path")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not an accessible directory: {path}")
    return path


if __name__ == "__main__":
    main()
