"""Per-run post-processing benchmark for generated CFD slice distributions."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .nearest_reference import (
    build_snapshot_descriptors,
    nearest_neighbor_spatial_metrics,
    nearest_reference_diagnostics,
)
from .plotting import (
    save_marginal_plots,
    save_nearest_reference_field_examples,
    save_spectrum_plots,
)
from .spectra import (
    CartesianGrid2D,
    infer_cartesian_grid_2d,
    sample_radial_spectra,
    spectral_band_rows,
)


def run_generation_benchmark(cfg: DictConfig) -> dict[str, Any]:
    """Benchmark one existing generated-test artifact without rerunning a model."""

    run_dir = _existing_directory(cfg.run_dir, "run_dir")
    run_name = run_dir.name
    output_root = Path(str(cfg.output_root)).expanduser().resolve()
    output_dir = output_root / run_name
    overwrite = bool(cfg.overwrite)
    sample_space = str(cfg.sample_space)
    if sample_space != "nondimensional":
        raise ValueError("benchmark v2 supports sample_space=nondimensional only")

    artifact_path = run_dir / str(cfg.generated_tensor)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"generated tensor artifact does not exist: {artifact_path}")
    source_summary_path = run_dir / "summary.json"
    source_config_path = run_dir / "resolved_config.yaml"
    if not source_summary_path.is_file():
        raise FileNotFoundError(f"source run summary does not exist: {source_summary_path}")
    if not source_config_path.is_file():
        raise FileNotFoundError(f"source resolved config does not exist: {source_config_path}")

    source_summary = json.loads(source_summary_path.read_text())
    source_cfg = OmegaConf.load(source_config_path)
    artifact = _load_generation_artifact(artifact_path)
    generated, reference, sample_ids, channel_names = _fixed_mesh_samples(artifact)
    generation_keys = sample_ids
    reference_ids = sample_ids
    grid, mesh_file = _load_grid(source_cfg, source_summary, cfg)
    if generated.shape[1] != grid.shape[0] * grid.shape[1]:
        raise ValueError(
            "generated sample node count does not match the source Cartesian mesh: "
            f"samples={generated.shape[1]}, grid={grid.shape}"
        )

    nearest_enabled = bool(cfg.nearest_reference.enabled)
    if bool(cfg.plots.enabled) and bool(cfg.plots.fields) and not nearest_enabled:
        raise ValueError("field comparison plots require nearest_reference.enabled=true")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"benchmark output already exists: {output_dir}; set overwrite=true explicitly"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output_dir / "benchmark_config.yaml", resolve=True)

    eps = float(cfg.eps)
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    quantiles = tuple(float(value) for value in cfg.quantiles)
    _validate_quantiles(quantiles)
    bands = _spectral_bands(cfg.spectra.bands)

    channel_rows = _channel_metric_rows(
        generated,
        reference,
        channel_names,
        quantiles=quantiles,
        eps=eps,
    )
    sample_rows = _sample_statistic_rows(
        generated,
        reference,
        generation_keys,
        reference_ids,
        channel_names,
        grid,
    )
    correlation_rows = _correlation_rows(generated, reference, channel_names)

    spectra_enabled = bool(cfg.spectra.enabled)
    need_sample_spectra = spectra_enabled or nearest_enabled
    generated_sample_power: np.ndarray | None = None
    reference_sample_power: np.ndarray | None = None
    generated_power: np.ndarray | None = None
    reference_power: np.ndarray | None = None
    k_centers: np.ndarray | None = None
    if need_sample_spectra:
        k_centers, generated_sample_power = sample_radial_spectra(
            generated,
            grid,
            num_k_bins=int(cfg.spectra.num_k_bins),
            subtract_mean=bool(cfg.spectra.subtract_mean),
        )
        reference_centers, reference_sample_power = sample_radial_spectra(
            reference,
            grid,
            num_k_bins=int(cfg.spectra.num_k_bins),
            subtract_mean=bool(cfg.spectra.subtract_mean),
        )
        np.testing.assert_allclose(k_centers, reference_centers, rtol=0.0, atol=0.0)
        generated_power = np.mean(generated_sample_power, axis=0)
        reference_power = np.mean(reference_sample_power, axis=0)

    spectrum_rows: list[dict[str, float | str]] = []
    band_rows: list[dict[str, float | str]] = []
    if spectra_enabled:
        assert k_centers is not None
        assert generated_power is not None
        assert reference_power is not None
        spectrum_rows = _spectrum_rows(
            k_centers,
            generated_power,
            reference_power,
            channel_names,
            k_nyquist=grid.k_nyquist_min,
            eps=eps,
        )
        band_rows = spectral_band_rows(
            generated_power,
            reference_power,
            k_centers,
            k_nyquist=grid.k_nyquist_min,
            channel_names=channel_names,
            bands=bands,
            eps=eps,
        )

    nearest_rows: list[dict[str, Any]] = []
    nearest_summary: dict[str, Any] | None = None
    nearest_indices: np.ndarray | None = None
    nearest_distances: np.ndarray | None = None
    if nearest_enabled:
        assert k_centers is not None
        assert generated_sample_power is not None
        assert reference_sample_power is not None
        generated_feature_names, generated_features = build_snapshot_descriptors(
            generated,
            grid,
            channel_names,
            generated_sample_power,
            k_centers,
            k_nyquist=grid.k_nyquist_min,
            bands=bands,
            include_channel_correlations=bool(cfg.nearest_reference.include_channel_correlations),
        )
        reference_feature_names, reference_features = build_snapshot_descriptors(
            reference,
            grid,
            channel_names,
            reference_sample_power,
            k_centers,
            k_nyquist=grid.k_nyquist_min,
            bands=bands,
            include_channel_correlations=bool(cfg.nearest_reference.include_channel_correlations),
        )
        if generated_feature_names != reference_feature_names:
            raise RuntimeError("generated/reference snapshot descriptor semantics differ")
        diagnostics = nearest_reference_diagnostics(
            generated_features,
            reference_features,
            generation_keys,
            reference_ids,
            generated_feature_names,
            normalization_eps=float(cfg.nearest_reference.normalization_eps),
        )
        nearest_rows = list(diagnostics.rows)
        nearest_summary = diagnostics.summary
        nearest_indices = diagnostics.generated_to_reference_indices
        nearest_distances = diagnostics.generated_to_reference_distances

    physical_rows: list[dict[str, float | str]] = []
    if bool(cfg.physics.enabled):
        physical_rows = _physical_metric_rows(generated, reference, channel_names, eps=eps)

    _write_csv(output_dir / "channel_metrics.csv", channel_rows)
    _write_csv(output_dir / "sample_statistics.csv", sample_rows)
    _write_csv(output_dir / "correlation_matrix.csv", correlation_rows)
    _write_csv(output_dir / "spectra.csv", spectrum_rows)
    _write_csv(output_dir / "spectral_bands.csv", band_rows)
    _write_csv(output_dir / "nearest_reference.csv", nearest_rows)
    _write_csv(output_dir / "physical_metrics.csv", physical_rows)

    if bool(cfg.plots.enabled):
        plot_root = output_dir / "plots"
        if bool(cfg.plots.marginals):
            save_marginal_plots(
                generated,
                reference,
                channel_names,
                plot_root / "marginals",
                bins=int(cfg.plots.marginal_bins),
                dpi=int(cfg.plots.dpi),
            )
        if bool(cfg.plots.spectra) and spectra_enabled:
            assert k_centers is not None
            assert generated_power is not None
            assert reference_power is not None
            save_spectrum_plots(
                k_centers,
                generated_power,
                reference_power,
                channel_names,
                plot_root / "spectra",
                k_nyquist=grid.k_nyquist_min,
                eps=eps,
                dpi=int(cfg.plots.dpi),
            )
        if bool(cfg.plots.fields):
            assert nearest_indices is not None
            assert nearest_distances is not None
            save_nearest_reference_field_examples(
                generated,
                reference,
                generation_keys,
                reference_ids,
                nearest_indices,
                nearest_distances,
                channel_names,
                grid,
                plot_root / "nearest_reference_fields",
                num_examples=int(cfg.plots.num_field_examples),
                max_channels=int(cfg.plots.max_field_channels),
                dpi=int(cfg.plots.dpi),
            )

    source_manifest_path = run_dir / "dataset_split_manifest.json"
    source_runtime_provenance = None
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text())
        source_runtime_provenance = source_manifest.get("runtime_provenance")

    summary = {
        "benchmark": "generation_distribution_v2",
        "run_name": run_name,
        "source_run_dir": str(run_dir),
        "source_generated_artifact": str(artifact_path),
        "source_mesh_file": str(mesh_file),
        "source_model": source_summary.get("model"),
        "source_model_parameters": source_summary.get("model_parameters"),
        "source_best_epoch": source_summary.get("best_epoch"),
        "source_runtime_provenance": source_runtime_provenance,
        "reference_population": "test",
        "comparison_mode": "unpaired_population",
        "generated_ids_semantics": "deterministic_sampling_rng_keys_only",
        "generated_reference_pairing": False,
        "sample_space": sample_space,
        "marginal_weighting": "node_pooled_equal_node_samples",
        "num_generated_samples": int(generated.shape[0]),
        "num_reference_samples": int(reference.shape[0]),
        "nodes_per_sample": int(generated.shape[1]),
        "channel_names": list(channel_names),
        "grid_shape_2d": list(grid.shape),
        "grid_spacing": list(grid.spacing),
        "k_nyquist_min": grid.k_nyquist_min,
        "spectra": {
            "enabled": spectra_enabled,
            "primary_coordinate": "k",
            "secondary_coordinate": "k_over_k_nyquist",
            "radial_range": "0 < k <= k_nyquist_min",
            "subtract_mean_per_sample": bool(cfg.spectra.subtract_mean),
            "num_k_bins": int(cfg.spectra.num_k_bins),
            "bands_in_k_over_k_nyquist": {name: list(bounds) for name, bounds in bands.items()},
        },
        "nearest_reference": nearest_summary,
        "channel_summary": _channel_summary(channel_rows, band_rows),
        "physical_summary": _physical_summary(physical_rows),
        "outputs": {
            "channel_metrics": "channel_metrics.csv",
            "sample_statistics": "sample_statistics.csv",
            "correlation_matrix": "correlation_matrix.csv",
            "spectra": "spectra.csv",
            "spectral_bands": "spectral_bands.csv",
            "nearest_reference": "nearest_reference.csv",
            "physical_metrics": "physical_metrics.csv",
            "plots": "plots" if bool(cfg.plots.enabled) else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def empirical_wasserstein_1(left: np.ndarray, right: np.ndarray) -> float:
    """Exact empirical W1 for equal-size one-dimensional unweighted samples."""

    lhs = np.asarray(left, dtype=np.float64).reshape(-1)
    rhs = np.asarray(right, dtype=np.float64).reshape(-1)
    if lhs.size != rhs.size:
        raise ValueError("empirical W1 currently requires equal sample counts")
    if lhs.size == 0:
        raise ValueError("empirical W1 requires non-empty samples")
    return float(np.mean(np.abs(np.sort(lhs) - np.sort(rhs))))


def _load_generation_artifact(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping payload in '{path}'")
    required = {
        "sample_ids",
        "node_counts",
        "channel_names",
        "generated_nondimensional",
        "target_nondimensional",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"generation artifact is missing required keys: {missing}")
    return payload


def _fixed_mesh_samples(
    artifact: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    generated = artifact["generated_nondimensional"]
    reference = artifact["target_nondimensional"]
    if not isinstance(generated, torch.Tensor) or not isinstance(reference, torch.Tensor):
        raise TypeError("generated and reference nondimensional fields must be tensors")
    if generated.shape != reference.shape or generated.ndim != 2:
        raise ValueError(
            "generated and reference fields must have identical shape [total_nodes, C]"
        )
    if not torch.isfinite(generated).all() or not torch.isfinite(reference).all():
        raise ValueError("generated and reference fields must be finite")

    sample_ids = tuple(artifact["sample_ids"])
    node_counts = tuple(int(value) for value in artifact["node_counts"])
    channel_names = tuple(artifact["channel_names"])
    if not sample_ids or len(sample_ids) != len(node_counts):
        raise ValueError("sample_ids and node_counts must be non-empty and aligned")
    if len(set(node_counts)) != 1:
        raise ValueError("benchmark v2 requires a shared fixed-size 2-D mesh")
    nodes_per_sample = node_counts[0]
    if nodes_per_sample <= 0 or sum(node_counts) != generated.shape[0]:
        raise ValueError("node_counts do not match the packed generation tensors")
    if len(channel_names) != generated.shape[1]:
        raise ValueError("channel_names do not match the generation tensor columns")
    if any(not isinstance(name, str) or not name for name in channel_names):
        raise ValueError("channel_names must contain non-empty strings")

    shape = (len(sample_ids), nodes_per_sample, generated.shape[1])
    return (
        generated.to(torch.float64).numpy().reshape(shape),
        reference.to(torch.float64).numpy().reshape(shape),
        sample_ids,
        channel_names,
    )


def _load_grid(
    source_cfg: DictConfig,
    source_summary: dict[str, Any],
    benchmark_cfg: DictConfig,
) -> tuple[CartesianGrid2D, Path]:
    mesh_value = OmegaConf.select(source_cfg, "data.mesh_file")
    if not isinstance(mesh_value, str) or not mesh_value:
        raise ValueError("source resolved_config.yaml does not define data.mesh_file")
    mesh_file = Path(mesh_value).expanduser().resolve()
    if not mesh_file.is_file():
        raise FileNotFoundError(f"source mesh file is not accessible: {mesh_file}")
    payload = torch.load(mesh_file, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("coords"), torch.Tensor):
        raise TypeError(f"source mesh '{mesh_file}' requires tensor key 'coords'")
    coords = payload["coords"]
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"source mesh coords must have shape [N, 2], got {tuple(coords.shape)}")

    raw_shape = source_summary.get("grid_shape_2d")
    shape_hint = None
    if isinstance(raw_shape, list) and len(raw_shape) == 2:
        shape_hint = (int(raw_shape[0]), int(raw_shape[1]))
    grid = infer_cartesian_grid_2d(
        coords.detach().cpu().numpy(),
        tolerance=float(benchmark_cfg.spectra.cartesian_tolerance),
        shape_hint=shape_hint,
    )
    return grid, mesh_file


def _channel_metric_rows(
    generated: np.ndarray,
    reference: np.ndarray,
    channel_names: tuple[str, ...],
    *,
    quantiles: tuple[float, ...],
    eps: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for channel, name in enumerate(channel_names):
        generated_values = generated[..., channel].reshape(-1)
        reference_values = reference[..., channel].reshape(-1)
        generated_mean = float(np.mean(generated_values))
        reference_mean = float(np.mean(reference_values))
        mean_bias = generated_mean - reference_mean
        generated_std = float(np.std(generated_values, ddof=0))
        reference_std = float(np.std(reference_values, ddof=0))
        row: dict[str, float | str] = {
            "channel": name,
            "generated_mean": generated_mean,
            "reference_mean": reference_mean,
            "mean_bias": mean_bias,
            "normalized_mean_bias": (
                mean_bias / reference_std if reference_std > eps else float("nan")
            ),
            "generated_std": generated_std,
            "reference_std": reference_std,
            "std_ratio": (generated_std / reference_std if reference_std > eps else float("nan")),
            "wasserstein_1": empirical_wasserstein_1(generated_values, reference_values),
        }
        for quantile in quantiles:
            label = _quantile_label(quantile)
            row[f"generated_{label}"] = float(np.quantile(generated_values, quantile))
            row[f"reference_{label}"] = float(np.quantile(reference_values, quantile))
        rows.append(row)
    return rows


def _sample_statistic_rows(
    generated: np.ndarray,
    reference: np.ndarray,
    generation_keys: tuple[str, ...],
    reference_ids: tuple[str, ...],
    channel_names: tuple[str, ...],
    grid: CartesianGrid2D,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    populations = (
        ("generated", generated, generation_keys, "sampling_key"),
        ("test_reference", reference, reference_ids, "test_sample_id"),
    )
    for population, values, identifiers, identifier_role in populations:
        for sample_index, identifier in enumerate(identifiers):
            for channel, name in enumerate(channel_names):
                field = values[sample_index, :, channel]
                neighbor_correlation, first_difference_rms = nearest_neighbor_spatial_metrics(
                    field,
                    grid,
                )
                rows.append(
                    {
                        "sample_index": sample_index,
                        "sample_id": identifier,
                        "sample_id_role": identifier_role,
                        "population": population,
                        "channel": name,
                        "spatial_mean": float(np.mean(field)),
                        "spatial_std": float(np.std(field, ddof=0)),
                        "nearest_neighbor_correlation": neighbor_correlation,
                        "first_difference_rms": first_difference_rms,
                    }
                )
    return rows


def _correlation_rows(
    generated: np.ndarray,
    reference: np.ndarray,
    channel_names: tuple[str, ...],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for population, values in (("generated", generated), ("test_reference", reference)):
        matrix = np.atleast_2d(np.corrcoef(values.reshape(-1, values.shape[2]), rowvar=False))
        for row_index, row_name in enumerate(channel_names):
            for column_index, column_name in enumerate(channel_names):
                rows.append(
                    {
                        "population": population,
                        "row_channel": row_name,
                        "column_channel": column_name,
                        "correlation": float(matrix[row_index, column_index]),
                    }
                )
    return rows


def _spectrum_rows(
    k_centers: np.ndarray,
    generated_power: np.ndarray,
    reference_power: np.ndarray,
    channel_names: tuple[str, ...],
    *,
    k_nyquist: float,
    eps: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for channel, name in enumerate(channel_names):
        for index, k_value in enumerate(k_centers):
            reference_value = float(reference_power[channel, index])
            generated_value = float(generated_power[channel, index])
            rows.append(
                {
                    "channel": name,
                    "k": float(k_value),
                    "k_over_k_nyquist": float(k_value / k_nyquist),
                    "generated_power": generated_value,
                    "reference_power": reference_value,
                    "generated_over_reference": (
                        generated_value / reference_value if reference_value > eps else float("nan")
                    ),
                }
            )
    return rows


def _physical_metric_rows(
    generated: np.ndarray,
    reference: np.ndarray,
    channel_names: tuple[str, ...],
    *,
    eps: float,
) -> list[dict[str, float | str]]:
    base_to_index = {
        name.split(".", maxsplit=1)[0]: index for index, name in enumerate(channel_names)
    }
    required = ("rho", "rhou", "rhov", "rhow", "rhoE")
    missing = [name for name in required if name not in base_to_index]
    if missing:
        raise ValueError(
            f"physics benchmark requires conservative channels {required}; missing {missing}"
        )

    generated_metrics = _physical_population_metrics(generated, base_to_index)
    reference_metrics = _physical_population_metrics(reference, base_to_index)
    near_zero_means = {
        "u_mean": "u_std",
        "v_mean": "v_std",
        "w_mean": "w_std",
    }
    fraction_metrics = {
        "rho_nonpositive_fraction",
        "rhoE_nonpositive_fraction",
        "specific_internal_energy_nonpositive_fraction",
    }
    rows: list[dict[str, float | str]] = []
    for metric in generated_metrics:
        generated_value = generated_metrics[metric]
        reference_value = reference_metrics[metric]
        difference = generated_value - reference_value
        normalized_difference = float("nan")
        ratio = float("nan")
        if metric in near_zero_means:
            scale = reference_metrics[near_zero_means[metric]]
            normalized_difference = difference / scale if scale > eps else float("nan")
            comparison = "bias_over_reference_std"
        elif metric in fraction_metrics:
            comparison = "absolute_difference"
        else:
            ratio = (
                generated_value / reference_value
                if math.isfinite(reference_value) and abs(reference_value) > eps
                else float("nan")
            )
            comparison = "generated_over_reference"
        rows.append(
            {
                "metric": metric,
                "comparison": comparison,
                "generated": generated_value,
                "reference": reference_value,
                "difference": difference,
                "normalized_difference": normalized_difference,
                "generated_over_reference": ratio,
            }
        )
    return rows


def _physical_population_metrics(
    values: np.ndarray,
    indices: dict[str, int],
) -> dict[str, float]:
    flat = values.reshape(-1, values.shape[2])
    rho = flat[:, indices["rho"]]
    rhou = flat[:, indices["rhou"]]
    rhov = flat[:, indices["rhov"]]
    rhow = flat[:, indices["rhow"]]
    rhoe = flat[:, indices["rhoE"]]
    valid_density = rho > 0.0

    u = np.full_like(rho, np.nan)
    v = np.full_like(rho, np.nan)
    w = np.full_like(rho, np.nan)
    u[valid_density] = rhou[valid_density] / rho[valid_density]
    v[valid_density] = rhov[valid_density] / rho[valid_density]
    w[valid_density] = rhow[valid_density] / rho[valid_density]
    velocity_squared = u**2 + v**2 + w**2
    kinetic_energy_density = 0.5 * rho * velocity_squared
    specific_internal_energy = np.full_like(rho, np.nan)
    specific_internal_energy[valid_density] = (
        rhoe[valid_density] / rho[valid_density] - 0.5 * velocity_squared[valid_density]
    )

    return {
        "rho_nonpositive_fraction": float(np.mean(rho <= 0.0)),
        "rhoE_nonpositive_fraction": float(np.mean(rhoe <= 0.0)),
        "u_mean": _nanmean(u),
        "u_std": _nanstd(u),
        "v_mean": _nanmean(v),
        "v_std": _nanstd(v),
        "w_mean": _nanmean(w),
        "w_std": _nanstd(w),
        "kinetic_energy_density_mean": _nanmean(kinetic_energy_density),
        "kinetic_energy_density_std": _nanstd(kinetic_energy_density),
        "specific_internal_energy_mean": _nanmean(specific_internal_energy),
        "specific_internal_energy_std": _nanstd(specific_internal_energy),
        "specific_internal_energy_nonpositive_fraction": float(
            np.mean((specific_internal_energy <= 0.0) | ~np.isfinite(specific_internal_energy))
        ),
    }


def _spectral_bands(config: DictConfig) -> dict[str, tuple[float, float]]:
    bands: dict[str, tuple[float, float]] = {}
    for name in config.keys():
        raw = config[name]
        if not isinstance(raw, (list, tuple)) and not OmegaConf.is_list(raw):
            raise TypeError(f"spectral band '{name}' must contain [lower, upper]")
        values = tuple(float(value) for value in raw)
        if len(values) != 2:
            raise ValueError(f"spectral band '{name}' must contain exactly two bounds")
        bands[str(name)] = (values[0], values[1])
    if not bands:
        raise ValueError("at least one spectral band is required")
    return bands


def _channel_summary(
    channel_rows: list[dict[str, float | str]],
    band_rows: list[dict[str, float | str]],
) -> dict[str, Any]:
    bands_by_channel: dict[str, dict[str, float | None]] = {}
    for row in band_rows:
        channel = str(row["channel"])
        bands_by_channel.setdefault(channel, {})[str(row["band"])] = _finite_or_none(
            float(row["generated_over_reference"])
        )

    result: dict[str, Any] = {}
    for row in channel_rows:
        channel = str(row["channel"])
        result[channel] = {
            "wasserstein_1": _finite_or_none(float(row["wasserstein_1"])),
            "mean_bias": _finite_or_none(float(row["mean_bias"])),
            "normalized_mean_bias": _finite_or_none(float(row["normalized_mean_bias"])),
            "std_ratio": _finite_or_none(float(row["std_ratio"])),
            "spectral_band_ratios": bands_by_channel.get(channel, {}),
        }
    return result


def _physical_summary(rows: list[dict[str, float | str]]) -> dict[str, Any]:
    return {
        str(row["metric"]): {
            "comparison": str(row["comparison"]),
            "generated": _finite_or_none(float(row["generated"])),
            "reference": _finite_or_none(float(row["reference"])),
            "difference": _finite_or_none(float(row["difference"])),
            "normalized_difference": _finite_or_none(float(row["normalized_difference"])),
            "generated_over_reference": _finite_or_none(float(row["generated_over_reference"])),
        }
        for row in rows
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_quantiles(values: tuple[float, ...]) -> None:
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("quantiles must lie in [0, 1]")
    if len(set(values)) != len(values):
        raise ValueError("quantiles must be unique")


def _quantile_label(value: float) -> str:
    return f"q{value:.3f}".replace(".", "p")


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _nanstd(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.std(finite, ddof=0)) if finite.size else float("nan")


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _existing_directory(value: object, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a filesystem path")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not an accessible directory: {path}")
    return path
