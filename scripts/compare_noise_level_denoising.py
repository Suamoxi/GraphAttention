"""Compare fixed-noise denoising and reconstructed spectra across epsilon models.

The diagnostic is intentionally paired at the forward-perturbation level: both
runs use the same held-out sample IDs, the same saved standardization, the same
normalized time fractions, and the same deterministic Gaussian field for each
sample.  It supports the discrete DDPM task family and the continuous VP-SDE
task so M21 and M23 can be compared without changing either trained checkpoint.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from graph_attention.evaluation.diffusion_diagnostics import graph_channel_mse
from graph_attention.evaluation.spectra import (
    CartesianGrid2D,
    infer_cartesian_grid_2d,
    sample_radial_spectra,
    spectral_band_rows,
)
from graph_attention.tasks import DiffusionDenoisingTask, NodeRegressionBatch, VPSDEDenoisingTask
from graph_attention.tasks.diffusion import _model_epsilon, _randn_by_graph, _sample_generator
from graph_attention.training.data_pipeline import (
    GraphTaskCollator,
    dataset_sample_ids,
    make_loader,
    task_batch_to_device,
)
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.generate_generative import (
    _existing_directory,
    _load_standardizers,
    _manifest_test_ids,
    _nonnegative_int,
    _positive_int,
    _validate_generative_standardizers,
)


_SUPPORTED_TASK = DiffusionDenoisingTask | VPSDEDenoisingTask
_NOISE_PURPOSE = "paired_noise_level_diagnostic"


@dataclass(frozen=True)
class _RunArtifacts:
    run_dir: Path
    source_cfg: DictConfig
    source_summary: dict[str, Any]
    task: _SUPPORTED_TASK
    dataset: Any
    standardizers: Any
    test_ids: tuple[str, ...]
    test_indices: tuple[int, ...]
    model: torch.nn.Module
    loader: Any
    grid: CartesianGrid2D
    channel_names: tuple[str, ...]


@dataclass(frozen=True)
class _RunDiagnostics:
    denoising_rows: list[dict[str, Any]]
    spectrum_rows: list[dict[str, Any]]
    band_rows: list[dict[str, Any]]
    schedule_rows: list[dict[str, Any]]
    reference_power: np.ndarray
    k_centers: np.ndarray
    num_samples: int


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="compare_noise_level_denoising",
)
def main(cfg: DictConfig) -> None:
    summary = run_noise_level_comparison(cfg)
    print(json.dumps(summary, indent=2, allow_nan=False))


@torch.no_grad()
def run_noise_level_comparison(cfg: DictConfig) -> dict[str, Any]:
    """Compare two frozen epsilon-prediction checkpoints at fixed noise levels."""

    baseline_dir = _existing_directory(cfg.baseline_run_dir, "baseline_run_dir")
    candidate_dir = _existing_directory(cfg.candidate_run_dir, "candidate_run_dir")
    levels = _diagnostic_levels(cfg.time_fractions)
    diagnostic_seed = _nonnegative_int(cfg.diagnostic_seed, "diagnostic_seed")
    batch_size = _positive_int(cfg.batch_size, "batch_size")
    num_workers = _nonnegative_int(cfg.num_workers, "num_workers")

    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA noise-level diagnostics requested but CUDA is unavailable")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("noise-level diagnostics are intentionally single-process")

    baseline = _load_run(
        baseline_dir,
        checkpoint_name=str(cfg.baseline_checkpoint),
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        spectra_cfg=cfg.spectra,
    )
    candidate = _load_run(
        candidate_dir,
        checkpoint_name=str(cfg.candidate_checkpoint),
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        spectra_cfg=cfg.spectra,
    )
    _validate_pair(baseline, candidate)

    baseline_result = _evaluate_run(
        baseline,
        levels=levels,
        diagnostic_seed=diagnostic_seed,
        spectra_cfg=cfg.spectra,
        device=device,
    )
    candidate_result = _evaluate_run(
        candidate,
        levels=levels,
        diagnostic_seed=diagnostic_seed,
        spectra_cfg=cfg.spectra,
        device=device,
    )
    _validate_reference_spectra(baseline_result, candidate_result)

    output_root = Path(str(cfg.output_root)).expanduser().resolve()
    output_name = _output_name(cfg.output_name, baseline_dir.name, candidate_dir.name)
    output_dir = output_root / output_name
    if output_dir.exists():
        if not bool(cfg.overwrite):
            raise FileExistsError(
                f"noise-level diagnostic output already exists: {output_dir}; "
                "set overwrite=true explicitly"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output_dir / "diagnostic_config.yaml", resolve=True)

    _write_csv(output_dir / "baseline_denoising.csv", baseline_result.denoising_rows)
    _write_csv(output_dir / "candidate_denoising.csv", candidate_result.denoising_rows)
    _write_csv(output_dir / "baseline_spectral_curves.csv", baseline_result.spectrum_rows)
    _write_csv(output_dir / "candidate_spectral_curves.csv", candidate_result.spectrum_rows)
    _write_csv(output_dir / "baseline_spectral_bands.csv", baseline_result.band_rows)
    _write_csv(output_dir / "candidate_spectral_bands.csv", candidate_result.band_rows)
    _write_csv(output_dir / "noise_schedule_comparison.csv", _compare_schedules(baseline_result, candidate_result))
    _write_csv(output_dir / "denoising_comparison.csv", _compare_denoising(baseline_result, candidate_result))
    _write_csv(output_dir / "spectral_band_comparison.csv", _compare_bands(baseline_result, candidate_result))

    summary = {
        "diagnostic": "paired_noise_level_denoising_spectrum_v1",
        "baseline": _run_summary(baseline, str(cfg.baseline_checkpoint)),
        "candidate": _run_summary(candidate, str(cfg.candidate_checkpoint)),
        "num_test_samples": baseline_result.num_samples,
        "channel_names": list(baseline.channel_names),
        "requested_time_fractions": levels,
        "num_noise_levels": len(levels),
        "diagnostic_seed": diagnostic_seed,
        "forward_noise_pairing": (
            "same deterministic Gaussian field per held-out sample ID is reused across all "
            "noise levels and across both checkpoints"
        ),
        "time_pairing": (
            "same normalized t fractions are requested; discrete DDPM maps each fraction to "
            "round(t*T), while continuous VP-SDE uses t directly"
        ),
        "mse_space": "standardized training space with equal physical-sample weighting",
        "spectral_space": "nondimensional physical fields after inverse statistical standardization",
        "spectra": {
            "num_k_bins": int(cfg.spectra.num_k_bins),
            "subtract_mean_per_sample": bool(cfg.spectra.subtract_mean),
            "bands_in_k_over_k_nyquist": {
                name: [float(bounds[0]), float(bounds[1])]
                for name, bounds in _spectral_bands(cfg.spectra).items()
            },
            "k_nyquist_min": baseline.grid.k_nyquist_min,
        },
        "interpretation": {
            "epsilon_mse": "direct epsilon-prediction error at the prescribed forward perturbation",
            "x0_mse": "clean-state reconstruction MSE implied by epsilon prediction",
            "spectral_ratio": "population-mean x0_hat spectral power divided by clean test power",
            "spectral_abs_error": "absolute value of spectral_ratio - 1",
            "candidate_error_reduction": "baseline absolute error - candidate absolute error; positive is smaller candidate error",
        },
        "outputs": {
            "baseline_denoising": "baseline_denoising.csv",
            "candidate_denoising": "candidate_denoising.csv",
            "baseline_spectral_curves": "baseline_spectral_curves.csv",
            "candidate_spectral_curves": "candidate_spectral_curves.csv",
            "baseline_spectral_bands": "baseline_spectral_bands.csv",
            "candidate_spectral_bands": "candidate_spectral_bands.csv",
            "noise_schedule_comparison": "noise_schedule_comparison.csv",
            "denoising_comparison": "denoising_comparison.csv",
            "spectral_band_comparison": "spectral_band_comparison.csv",
        },
        "device": str(device),
        "dtype": "float32",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def _load_run(
    run_dir: Path,
    *,
    checkpoint_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    spectra_cfg: DictConfig,
) -> _RunArtifacts:
    source_config_path = run_dir / "resolved_config.yaml"
    source_summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "dataset_split_manifest.json"
    standardizers_path = run_dir / "standardizers.pt"
    checkpoint_path = run_dir / checkpoint_name
    for path, label in (
        (source_config_path, "resolved_config.yaml"),
        (source_summary_path, "summary.json"),
        (manifest_path, "dataset_split_manifest.json"),
        (standardizers_path, "standardizers.pt"),
        (checkpoint_path, "checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"source run {label} does not exist: {path}")

    source_cfg = OmegaConf.load(source_config_path)
    source_summary = json.loads(source_summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    seed = _nonnegative_int(source_cfg.seed, "source seed")

    dataset = instantiate(source_cfg.data)
    if not hasattr(dataset, "field_catalog"):
        raise TypeError("configured dataset must expose field_catalog")
    task = instantiate(source_cfg.task)
    if not isinstance(task, (DiffusionDenoisingTask, VPSDEDenoisingTask)):
        raise TypeError("noise-level comparison supports discrete DDPM and continuous VP-SDE tasks")

    standardizers = _load_standardizers(standardizers_path)
    _validate_generative_standardizers(standardizers)
    if standardizers.physical_nondimensionalization != task.physical_nondimensionalization:
        raise ValueError("saved standardizers and source task disagree on nondimensionalization")

    test_ids = _manifest_test_ids(manifest)
    sample_ids = dataset_sample_ids(dataset)
    index_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    missing = [sample_id for sample_id in test_ids if sample_id not in index_by_id]
    if missing:
        raise ValueError(f"source test split contains IDs absent from current dataset: {missing[:5]}")
    test_indices = tuple(index_by_id[sample_id] for sample_id in test_ids)

    collator = GraphTaskCollator(task, dataset.field_catalog, source_cfg.geometry)
    loader = make_loader(
        dataset,
        test_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=seed,
    )

    probe_host = next(iter(loader))
    probe_scaled = standardizers.transform(probe_host)
    probe_problem = task.make_validation_problem(probe_scaled)
    model, _ = instantiate_controlled_model(source_cfg.model, probe_problem, seed=seed)
    model = model.to(device=device, dtype=torch.float32)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint does not contain model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    grid = _grid_from_host_batch(probe_host, source_summary, spectra_cfg)
    return _RunArtifacts(
        run_dir=run_dir,
        source_cfg=source_cfg,
        source_summary=source_summary,
        task=task,
        dataset=dataset,
        standardizers=standardizers,
        test_ids=test_ids,
        test_indices=test_indices,
        model=model,
        loader=loader,
        grid=grid,
        channel_names=tuple(probe_scaled.input_channels),
    )


def _evaluate_run(
    run: _RunArtifacts,
    *,
    levels: list[float],
    diagnostic_seed: int,
    spectra_cfg: DictConfig,
    device: torch.device,
) -> _RunDiagnostics:
    standardizers = run.standardizers.to(device=device, dtype=torch.float32)
    channels = len(run.channel_names)
    num_k_bins = _positive_int(spectra_cfg.num_k_bins, "spectra.num_k_bins")
    subtract_mean = bool(spectra_cfg.subtract_mean)
    eps = float(spectra_cfg.eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("spectra.eps must be positive and finite")

    metric_sums = [
        {
            "epsilon_mse": torch.zeros(channels, dtype=torch.float64),
            "x0_mse": torch.zeros(channels, dtype=torch.float64),
            "oracle_x0_mse": torch.zeros(channels, dtype=torch.float64),
        }
        for _ in levels
    ]
    spectral_sums = [np.zeros((channels, num_k_bins), dtype=np.float64) for _ in levels]
    reference_power_sum = np.zeros((channels, num_k_bins), dtype=np.float64)
    schedule: list[dict[str, Any] | None] = [None for _ in levels]
    k_centers: np.ndarray | None = None
    num_samples = 0

    for host_batch in run.loader:
        batch = task_batch_to_device(host_batch, device=device, dtype=torch.float32)
        scaled = standardizers.transform(batch)
        if tuple(scaled.input_channels) != run.channel_names:
            raise ValueError("test batches produced inconsistent channel semantics")

        generators = [
            _sample_generator(device, diagnostic_seed, _NOISE_PURPOSE, sample_id)
            for sample_id in scaled.source.sample_ids
        ]
        noise = _randn_by_graph(scaled, generators)
        clean_nondimensional = standardizers.inputs.inverse(scaled.inputs, scaled.input_channels)
        clean_samples = _fixed_mesh_numpy(clean_nondimensional, scaled.ptr)
        batch_k, clean_power = sample_radial_spectra(
            clean_samples,
            run.grid,
            num_k_bins=num_k_bins,
            subtract_mean=subtract_mean,
        )
        if k_centers is None:
            k_centers = batch_k
        elif not np.allclose(k_centers, batch_k, rtol=0.0, atol=0.0):
            raise RuntimeError("spectral k bins changed across test batches")
        reference_power_sum += np.sum(clean_power, axis=0)
        num_samples += scaled.num_graphs

        for level_index, fraction in enumerate(levels):
            problem, conditioning_time, alpha, sigma, metadata = _problem_at_fraction(
                run.task,
                scaled,
                noise,
                fraction,
            )
            epsilon_hat = _model_epsilon(
                run.model,
                scaled,
                problem.inputs,
                conditioning_time,
            )
            x0_hat = _reconstruct_x0(
                problem.inputs,
                epsilon_hat,
                alpha,
                sigma,
                scaled.batch_index,
            )
            x0_oracle = _reconstruct_x0(
                problem.inputs,
                noise,
                alpha,
                sigma,
                scaled.batch_index,
            )

            metric_sums[level_index]["epsilon_mse"] += graph_channel_mse(
                epsilon_hat, noise, scaled.ptr
            ).detach().cpu().to(torch.float64).sum(dim=0)
            metric_sums[level_index]["x0_mse"] += graph_channel_mse(
                x0_hat, scaled.inputs, scaled.ptr
            ).detach().cpu().to(torch.float64).sum(dim=0)
            metric_sums[level_index]["oracle_x0_mse"] += graph_channel_mse(
                x0_oracle, scaled.inputs, scaled.ptr
            ).detach().cpu().to(torch.float64).sum(dim=0)

            x0_nondimensional = standardizers.inputs.inverse(x0_hat, scaled.input_channels)
            x0_samples = _fixed_mesh_numpy(x0_nondimensional, scaled.ptr)
            _, x0_power = sample_radial_spectra(
                x0_samples,
                run.grid,
                num_k_bins=num_k_bins,
                subtract_mean=subtract_mean,
            )
            spectral_sums[level_index] += np.sum(x0_power, axis=0)

            if schedule[level_index] is None:
                schedule[level_index] = metadata
            else:
                _assert_same_schedule(schedule[level_index], metadata)

    if num_samples == 0 or k_centers is None:
        raise ValueError("diagnostic test loader contains no samples")

    reference_power = reference_power_sum / num_samples
    denoising_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    bands = _spectral_bands(spectra_cfg)

    for index, fraction in enumerate(levels):
        metadata = schedule[index]
        if metadata is None:
            raise RuntimeError("missing noise schedule metadata")
        schedule_rows.append(dict(metadata))
        means = {name: values / num_samples for name, values in metric_sums[index].items()}
        factor = (float(metadata["sigma"]) / float(metadata["alpha"])) ** 2
        for channel_index, channel_name in enumerate(run.channel_names):
            epsilon_mse = float(means["epsilon_mse"][channel_index])
            x0_mse = float(means["x0_mse"][channel_index])
            expected_x0_mse = factor * epsilon_mse
            denoising_rows.append(
                {
                    **metadata,
                    "channel": channel_name,
                    "epsilon_mse": epsilon_mse,
                    "x0_mse": x0_mse,
                    "oracle_x0_mse": float(means["oracle_x0_mse"][channel_index]),
                    "epsilon_to_x0_mse_factor": factor,
                    "expected_x0_mse_from_epsilon": expected_x0_mse,
                    "x0_mse_over_expected": (
                        x0_mse / expected_x0_mse if expected_x0_mse > eps else float("nan")
                    ),
                }
            )
        denoising_rows.append(
            {
                **metadata,
                "channel": "__mean__",
                "epsilon_mse": float(means["epsilon_mse"].mean()),
                "x0_mse": float(means["x0_mse"].mean()),
                "oracle_x0_mse": float(means["oracle_x0_mse"].mean()),
                "epsilon_to_x0_mse_factor": factor,
                "expected_x0_mse_from_epsilon": factor * float(means["epsilon_mse"].mean()),
                "x0_mse_over_expected": _safe_ratio(
                    float(means["x0_mse"].mean()),
                    factor * float(means["epsilon_mse"].mean()),
                    eps,
                ),
            }
        )

        generated_power = spectral_sums[index] / num_samples
        ratio = np.divide(
            generated_power,
            reference_power,
            out=np.full_like(generated_power, np.nan),
            where=reference_power > eps,
        )
        for channel_index, channel_name in enumerate(run.channel_names):
            for k_index, k_value in enumerate(k_centers):
                value = float(ratio[channel_index, k_index])
                spectrum_rows.append(
                    {
                        **metadata,
                        "channel": channel_name,
                        "k": float(k_value),
                        "k_over_k_nyquist": float(k_value / run.grid.k_nyquist_min),
                        "x0_hat_power": float(generated_power[channel_index, k_index]),
                        "reference_power": float(reference_power[channel_index, k_index]),
                        "x0_hat_over_reference": value,
                        "abs_ratio_error": abs(value - 1.0) if math.isfinite(value) else float("nan"),
                        "log_power_ratio": math.log(value) if value > 0.0 and math.isfinite(value) else float("nan"),
                    }
                )
        rows = spectral_band_rows(
            generated_power,
            reference_power,
            k_centers,
            k_nyquist=run.grid.k_nyquist_min,
            channel_names=run.channel_names,
            bands=bands,
            eps=eps,
        )
        for row in rows:
            value = float(row["generated_over_reference"])
            band_rows.append(
                {
                    **metadata,
                    **row,
                    "abs_ratio_error": abs(value - 1.0) if math.isfinite(value) else float("nan"),
                }
            )

    return _RunDiagnostics(
        denoising_rows=denoising_rows,
        spectrum_rows=spectrum_rows,
        band_rows=band_rows,
        schedule_rows=schedule_rows,
        reference_power=reference_power,
        k_centers=k_centers,
        num_samples=num_samples,
    )


def _problem_at_fraction(
    task: _SUPPORTED_TASK,
    batch: NodeRegressionBatch,
    noise: torch.Tensor,
    requested_fraction: float,
) -> tuple[NodeRegressionBatch, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if isinstance(task, DiffusionDenoisingTask):
        timestep = min(task.timesteps, max(1, round(requested_fraction * task.timesteps)))
        graph_timesteps = torch.full(
            (batch.num_graphs,), timestep, device=batch.inputs.device, dtype=torch.long
        )
        problem = task._diffusion_problem(batch, graph_timesteps, noise)
        alpha_bar = task._alpha_bar_for(batch.inputs)[timestep]
        alpha_scalar = torch.sqrt(alpha_bar)
        sigma_scalar = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0))
        alpha = alpha_scalar.expand(batch.num_graphs)
        sigma = sigma_scalar.expand(batch.num_graphs)
        conditioning = task._normalized_time(graph_timesteps, batch.inputs.dtype)
        actual_fraction = timestep / task.timesteps
        kind = "discrete_ddpm"
        discrete_timestep: int | None = timestep
    else:
        actual_fraction = max(task.training_eps, requested_fraction)
        times = torch.full(
            (batch.num_graphs,), actual_fraction, device=batch.inputs.device, dtype=batch.inputs.dtype
        )
        problem = task._vp_problem(batch, times, noise)
        alpha, sigma = task.marginal_coefficients(times)
        conditioning = times
        kind = "continuous_vp_sde"
        discrete_timestep = None

    alpha_value = float(alpha[0].detach().cpu())
    sigma_value = float(sigma[0].detach().cpu())
    snr = (alpha_value / sigma_value) ** 2 if sigma_value > 0.0 else float("inf")
    log_snr = math.log(snr) if snr > 0.0 and math.isfinite(snr) else float("inf")
    metadata = {
        "requested_t_fraction": float(requested_fraction),
        "actual_t_fraction": float(actual_fraction),
        "discrete_timestep": discrete_timestep,
        "process": kind,
        "alpha": alpha_value,
        "sigma": sigma_value,
        "snr": snr,
        "log_snr": log_snr,
    }
    return problem, conditioning, alpha, sigma, metadata


def _reconstruct_x0(
    noisy_state: torch.Tensor,
    epsilon: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    batch_index: torch.Tensor,
) -> torch.Tensor:
    if noisy_state.shape != epsilon.shape:
        raise ValueError("noisy_state and epsilon must have identical shape")
    if alpha.ndim != 1 or sigma.shape != alpha.shape:
        raise ValueError("alpha and sigma must be graph-level one-dimensional tensors")
    node_alpha = alpha[batch_index].unsqueeze(1)
    node_sigma = sigma[batch_index].unsqueeze(1)
    if bool(torch.any(node_alpha <= 0.0)):
        raise ValueError("x0 reconstruction requires positive alpha")
    return (noisy_state - node_sigma * epsilon) / node_alpha


def _grid_from_host_batch(
    batch: NodeRegressionBatch,
    source_summary: dict[str, Any],
    spectra_cfg: DictConfig,
) -> CartesianGrid2D:
    node_count = int(batch.ptr[1] - batch.ptr[0])
    coords = batch.coords[:node_count].detach().cpu().numpy()
    shape_hint = None
    raw_shape = source_summary.get("grid_shape_2d")
    if isinstance(raw_shape, list) and len(raw_shape) == 2:
        shape_hint = (int(raw_shape[0]), int(raw_shape[1]))
    return infer_cartesian_grid_2d(
        coords,
        tolerance=float(spectra_cfg.cartesian_tolerance),
        shape_hint=shape_hint,
    )


def _fixed_mesh_numpy(values: torch.Tensor, ptr: torch.Tensor) -> np.ndarray:
    counts = [int(stop - start) for start, stop in zip(ptr[:-1], ptr[1:], strict=True)]
    if not counts or len(set(counts)) != 1:
        raise ValueError("spectral noise-level diagnostics require one shared fixed-size mesh")
    return values.detach().cpu().to(torch.float64).numpy().reshape(
        len(counts), counts[0], values.shape[1]
    )


def _validate_pair(baseline: _RunArtifacts, candidate: _RunArtifacts) -> None:
    if baseline.test_ids != candidate.test_ids:
        raise ValueError("baseline and candidate must use the identical held-out test sample IDs")
    if baseline.channel_names != candidate.channel_names:
        raise ValueError("baseline and candidate state-channel semantics differ")
    left = baseline.standardizers
    right = candidate.standardizers
    if left.inputs.channel_names != right.inputs.channel_names:
        raise ValueError("baseline and candidate standardizer channel names differ")
    if not torch.equal(left.inputs.mean, right.inputs.mean) or not torch.equal(
        left.inputs.scale, right.inputs.scale
    ):
        raise ValueError("baseline and candidate must use identical input standardization")
    if left.physical_nondimensionalization != right.physical_nondimensionalization:
        raise ValueError("baseline and candidate nondimensionalization settings differ")
    if baseline.grid.shape != candidate.grid.shape:
        raise ValueError("baseline and candidate Cartesian grid shapes differ")
    if not np.allclose(baseline.grid.spacing, candidate.grid.spacing, rtol=0.0, atol=1.0e-12):
        raise ValueError("baseline and candidate Cartesian grid spacing differs")
    if not np.array_equal(baseline.grid.linear_indices, candidate.grid.linear_indices):
        raise ValueError("baseline and candidate Cartesian node ordering differs")


def _validate_reference_spectra(
    baseline: _RunDiagnostics,
    candidate: _RunDiagnostics,
) -> None:
    if baseline.num_samples != candidate.num_samples:
        raise ValueError("baseline and candidate diagnostic sample counts differ")
    if not np.allclose(baseline.k_centers, candidate.k_centers, rtol=0.0, atol=0.0):
        raise ValueError("baseline and candidate spectral k bins differ")
    if not np.allclose(
        baseline.reference_power,
        candidate.reference_power,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("baseline and candidate clean reference spectra differ")


def _compare_schedules(
    baseline: _RunDiagnostics,
    candidate: _RunDiagnostics,
) -> list[dict[str, Any]]:
    rows = []
    for left, right in zip(baseline.schedule_rows, candidate.schedule_rows, strict=True):
        _require_same_requested_fraction(left, right)
        rows.append(
            {
                "requested_t_fraction": left["requested_t_fraction"],
                "baseline_actual_t_fraction": left["actual_t_fraction"],
                "candidate_actual_t_fraction": right["actual_t_fraction"],
                "baseline_discrete_timestep": left["discrete_timestep"],
                "candidate_discrete_timestep": right["discrete_timestep"],
                "baseline_alpha": left["alpha"],
                "candidate_alpha": right["alpha"],
                "alpha_difference": float(right["alpha"]) - float(left["alpha"]),
                "baseline_sigma": left["sigma"],
                "candidate_sigma": right["sigma"],
                "sigma_difference": float(right["sigma"]) - float(left["sigma"]),
                "baseline_log_snr": left["log_snr"],
                "candidate_log_snr": right["log_snr"],
                "log_snr_difference": float(right["log_snr"]) - float(left["log_snr"]),
            }
        )
    return rows


def _compare_denoising(
    baseline: _RunDiagnostics,
    candidate: _RunDiagnostics,
) -> list[dict[str, Any]]:
    right_by_key = {
        (row["requested_t_fraction"], row["channel"]): row
        for row in candidate.denoising_rows
    }
    rows = []
    for left in baseline.denoising_rows:
        key = (left["requested_t_fraction"], left["channel"])
        right = right_by_key.get(key)
        if right is None:
            raise ValueError(f"candidate denoising output is missing key {key}")
        rows.append(
            {
                "requested_t_fraction": left["requested_t_fraction"],
                "channel": left["channel"],
                "baseline_epsilon_mse": left["epsilon_mse"],
                "candidate_epsilon_mse": right["epsilon_mse"],
                "epsilon_mse_reduction": float(left["epsilon_mse"]) - float(right["epsilon_mse"]),
                "baseline_x0_mse": left["x0_mse"],
                "candidate_x0_mse": right["x0_mse"],
                "x0_mse_reduction": float(left["x0_mse"]) - float(right["x0_mse"]),
                "baseline_oracle_x0_mse": left["oracle_x0_mse"],
                "candidate_oracle_x0_mse": right["oracle_x0_mse"],
                "baseline_x0_mse_over_expected": left["x0_mse_over_expected"],
                "candidate_x0_mse_over_expected": right["x0_mse_over_expected"],
            }
        )
    return rows


def _compare_bands(
    baseline: _RunDiagnostics,
    candidate: _RunDiagnostics,
) -> list[dict[str, Any]]:
    right_by_key = {
        (row["requested_t_fraction"], row["channel"], row["band"]): row
        for row in candidate.band_rows
    }
    rows = []
    for left in baseline.band_rows:
        key = (left["requested_t_fraction"], left["channel"], left["band"])
        right = right_by_key.get(key)
        if right is None:
            raise ValueError(f"candidate spectral-band output is missing key {key}")
        left_error = float(left["abs_ratio_error"])
        right_error = float(right["abs_ratio_error"])
        rows.append(
            {
                "requested_t_fraction": left["requested_t_fraction"],
                "channel": left["channel"],
                "band": left["band"],
                "k_fraction_min": left["k_fraction_min"],
                "k_fraction_max": left["k_fraction_max"],
                "baseline_power_ratio": left["generated_over_reference"],
                "candidate_power_ratio": right["generated_over_reference"],
                "baseline_abs_ratio_error": left_error,
                "candidate_abs_ratio_error": right_error,
                "candidate_error_reduction": left_error - right_error,
            }
        )
    return rows


def _run_summary(run: _RunArtifacts, checkpoint_name: str) -> dict[str, Any]:
    return {
        "run_name": run.run_dir.name,
        "run_dir": str(run.run_dir),
        "checkpoint": checkpoint_name,
        "best_epoch": run.source_summary.get("best_epoch"),
        "task_class": type(run.task).__name__,
        "model": type(run.model).__name__,
        "model_parameters": sum(parameter.numel() for parameter in run.model.parameters()),
    }


def _diagnostic_levels(values: Any) -> list[float]:
    levels = [float(value) for value in values]
    if not levels:
        raise ValueError("time_fractions must be non-empty")
    if any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in levels):
        raise ValueError("time_fractions must contain finite values in (0, 1]")
    if any(right <= left for left, right in zip(levels, levels[1:])):
        raise ValueError("time_fractions must be strictly increasing")
    return levels


def _spectral_bands(config: DictConfig) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for name, raw in config.bands.items():
        values = tuple(float(value) for value in raw)
        if len(values) != 2:
            raise ValueError(f"spectral band '{name}' must contain [lower, upper]")
        lower, upper = values
        if lower < 0.0 or upper <= lower or upper > 1.0:
            raise ValueError(f"invalid spectral band '{name}': {values}")
        result[str(name)] = (lower, upper)
    if not result:
        raise ValueError("spectra.bands must be non-empty")
    return result


def _assert_same_schedule(left: dict[str, Any], right: dict[str, Any]) -> None:
    for name in ("requested_t_fraction", "actual_t_fraction", "alpha", "sigma", "snr", "log_snr"):
        if not math.isclose(float(left[name]), float(right[name]), rel_tol=1.0e-6, abs_tol=1.0e-8):
            raise RuntimeError(f"noise schedule metadata changed across batches for '{name}'")
    if left["discrete_timestep"] != right["discrete_timestep"] or left["process"] != right["process"]:
        raise RuntimeError("noise schedule semantics changed across batches")


def _require_same_requested_fraction(left: dict[str, Any], right: dict[str, Any]) -> None:
    if not math.isclose(
        float(left["requested_t_fraction"]),
        float(right["requested_t_fraction"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("baseline and candidate diagnostic levels are misaligned")


def _safe_ratio(numerator: float, denominator: float, eps: float) -> float:
    return numerator / denominator if abs(denominator) > eps else float("nan")


def _output_name(value: object, baseline: str, candidate: str) -> str:
    if value is None:
        return f"{baseline}__vs__{candidate}"
    result = str(value).strip()
    if not result or Path(result).name != result or result in {".", ".."}:
        raise ValueError("output_name must be null or one non-empty directory name")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
