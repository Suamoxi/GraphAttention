"""Distribution benchmark for generated full-volume 3-D Cartesian HIT fields."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .generation_benchmark import (
    _channel_metric_rows,
    _channel_summary,
    _correlation_rows,
    _energy_spectrum_summary,
    _finite_or_none,
    _fixed_mesh_samples,
    _physical_metric_rows,
    _physical_summary,
    _spectral_bands,
    _spectrum_rows,
    _validate_quantiles,
    standardized_wasserstein_rows,
)
from .nearest_reference import nearest_reference_diagnostics
from .plotting import (
    save_energy_spectrum_population_plots,
    save_marginal_plots,
    save_spectrum_plots,
)
from .spectra import (
    energy_spectral_band_rows,
    energy_spectrum_population_rows,
    energy_spectrum_sample_rows,
    spectral_band_rows,
)
from .spectra_3d import (
    CartesianGrid3D,
    infer_cartesian_grid_3d,
    nearest_neighbor_spatial_metrics_3d,
    sample_radial_spectra_3d,
    sample_velocity_energy_spectra_3d,
)


def run_generation_benchmark_3d(cfg: DictConfig) -> dict[str, Any]:
    """Benchmark one existing 3-D generated-test population without model execution."""

    run_dir = _existing_directory(cfg.run_dir, "run_dir")
    output_root = Path(str(cfg.output_root)).expanduser().resolve()
    output_dir = output_root / run_dir.name
    if output_dir.exists():
        if not bool(cfg.overwrite):
            raise FileExistsError(
                f"benchmark output already exists: {output_dir}; set overwrite=true explicitly"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output_dir / "benchmark_config.yaml", resolve=True)

    artifact_path = run_dir / str(cfg.generated_tensor)
    source_summary_path = run_dir / "summary.json"
    source_config_path = run_dir / "resolved_config.yaml"
    for path, label in (
        (artifact_path, "generated tensor"),
        (source_summary_path, "generation summary"),
        (source_config_path, "resolved training config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"3-D benchmark {label} does not exist: {path}")

    source_summary = json.loads(source_summary_path.read_text())
    source_cfg = OmegaConf.load(source_config_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise TypeError("generated_test.pt must contain a mapping")

    generated, reference, generated_ids, reference_ids, channel_names = _fixed_mesh_samples(
        artifact
    )
    if str(cfg.sample_space) != "nondimensional":
        raise ValueError("3-D benchmark currently supports sample_space=nondimensional only")

    grid, mesh_file = _load_grid_3d(source_cfg, cfg)
    expected_nodes = int(np.prod(grid.source_shape))
    if generated.shape[1] != expected_nodes:
        raise ValueError(
            "generated sample node count does not match source 3-D mesh: "
            f"samples={generated.shape[1]}, mesh={grid.source_shape}"
        )

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
    correlation_rows = _correlation_rows(generated, reference, channel_names)
    sample_rows = _sample_statistic_rows_3d(
        generated,
        reference,
        generated_ids,
        reference_ids,
        channel_names,
        grid,
    )

    k_centers, generated_sample_power = sample_radial_spectra_3d(
        generated,
        grid,
        num_k_bins=int(cfg.spectra.num_k_bins),
        subtract_mean=bool(cfg.spectra.subtract_mean),
    )
    reference_centers, reference_sample_power = sample_radial_spectra_3d(
        reference,
        grid,
        num_k_bins=int(cfg.spectra.num_k_bins),
        subtract_mean=bool(cfg.spectra.subtract_mean),
    )
    np.testing.assert_allclose(k_centers, reference_centers, rtol=0.0, atol=0.0)
    generated_power = np.mean(generated_sample_power, axis=0)
    reference_power = np.mean(reference_sample_power, axis=0)
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

    physical_rows: list[dict[str, float | str]] = []
    if bool(cfg.physics.enabled):
        physical_rows = _physical_metric_rows(
            generated,
            reference,
            channel_names,
            eps=eps,
        )

    energy_lower = float(cfg.energy_spectrum.lower_quantile)
    energy_upper = float(cfg.energy_spectrum.upper_quantile)
    energy_centers, generated_energy = sample_velocity_energy_spectra_3d(
        generated,
        grid,
        channel_names,
        num_k_bins=int(cfg.spectra.num_k_bins),
        subtract_mean=True,
    )
    reference_energy_centers, reference_energy = sample_velocity_energy_spectra_3d(
        reference,
        grid,
        channel_names,
        num_k_bins=int(cfg.spectra.num_k_bins),
        subtract_mean=True,
    )
    np.testing.assert_allclose(
        energy_centers,
        reference_energy_centers,
        rtol=0.0,
        atol=0.0,
    )
    energy_sample_rows = energy_spectrum_sample_rows(
        generated_energy,
        reference_energy,
        energy_centers,
        generated_ids,
        reference_ids,
        k_nyquist=grid.k_nyquist_min,
    )
    energy_summary_rows = energy_spectrum_population_rows(
        generated_energy,
        reference_energy,
        energy_centers,
        k_nyquist=grid.k_nyquist_min,
        eps=eps,
        lower_quantile=energy_lower,
        upper_quantile=energy_upper,
    )
    energy_band_rows = energy_spectral_band_rows(
        generated_energy,
        reference_energy,
        energy_centers,
        k_nyquist=grid.k_nyquist_min,
        bands=bands,
        eps=eps,
        lower_quantile=energy_lower,
        upper_quantile=energy_upper,
        bin_width=grid.k_nyquist_min / float(cfg.spectra.num_k_bins),
    )

    standardized_rows = _standardized_wasserstein_from_source(
        source_summary,
        generated,
        reference,
        channel_names,
    )

    nearest_summary: dict[str, Any] | None = None
    nearest_rows: list[dict[str, Any]] = []
    if bool(cfg.nearest_reference.enabled):
        feature_names, generated_features = _snapshot_descriptors_3d(
            generated,
            channel_names,
            grid,
            generated_sample_power,
            k_centers,
            bands=bands,
            include_channel_correlations=bool(
                cfg.nearest_reference.include_channel_correlations
            ),
        )
        reference_feature_names, reference_features = _snapshot_descriptors_3d(
            reference,
            channel_names,
            grid,
            reference_sample_power,
            k_centers,
            bands=bands,
            include_channel_correlations=bool(
                cfg.nearest_reference.include_channel_correlations
            ),
        )
        if feature_names != reference_feature_names:
            raise RuntimeError("generated/reference descriptor semantics differ")
        diagnostics = nearest_reference_diagnostics(
            generated_features,
            reference_features,
            generated_ids,
            reference_ids,
            feature_names,
            normalization_eps=float(cfg.nearest_reference.normalization_eps),
        )
        nearest_summary = diagnostics.summary
        nearest_rows = diagnostics.rows

    _write_csv(output_dir / "channel_metrics.csv", channel_rows)
    _write_csv(output_dir / "standardized_wasserstein.csv", standardized_rows)
    _write_csv(output_dir / "sample_statistics.csv", sample_rows)
    _write_csv(output_dir / "correlation_matrix.csv", correlation_rows)
    _write_csv(output_dir / "spectra.csv", spectrum_rows)
    _write_csv(output_dir / "spectral_bands.csv", band_rows)
    _write_csv(output_dir / "physical_metrics.csv", physical_rows)
    _write_csv(output_dir / "energy_spectrum_per_sample.csv", energy_sample_rows)
    _write_csv(output_dir / "energy_spectrum_summary.csv", energy_summary_rows)
    _write_csv(output_dir / "energy_spectral_bands.csv", energy_band_rows)
    _write_csv(output_dir / "nearest_reference.csv", nearest_rows)

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
        if bool(cfg.plots.spectra):
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
        if bool(cfg.plots.energy_spectrum):
            save_energy_spectrum_population_plots(
                energy_centers,
                generated_energy,
                reference_energy,
                plot_root / "energy_spectrum",
                k_nyquist=grid.k_nyquist_min,
                eps=eps,
                dpi=int(cfg.plots.dpi),
                lower_quantile=energy_lower,
                upper_quantile=energy_upper,
                dimension_label="3-D",
            )
        if bool(OmegaConf.select(cfg, "plots.fields", default=False)):
            raise ValueError(
                "3-D benchmark does not render full-volume field plots; "
                "set plots.fields=false"
            )

    summary = {
        "benchmark": "generation_distribution_3d_v1",
        "run_name": run_dir.name,
        "source_run_dir": source_summary.get("source_run_dir"),
        "source_generated_artifact": str(artifact_path),
        "source_mesh_file": str(mesh_file),
        "source_model": source_summary.get("model"),
        "source_model_parameters": source_summary.get("model_parameters"),
        "source_best_epoch": source_summary.get("source_best_epoch"),
        "reference_population": "test",
        "comparison_mode": "unpaired_population",
        "generated_reference_pairing": False,
        "sample_space": "nondimensional",
        "num_generated_samples": int(generated.shape[0]),
        "num_reference_samples": int(reference.shape[0]),
        "source_nodes_per_sample": expected_nodes,
        "channel_names": list(channel_names),
        "grid_dimension": 3,
        "grid_source_shape": list(grid.source_shape),
        "grid_fft_shape": list(grid.shape),
        "grid_spacing": list(grid.spacing),
        "periodic_endpoint_mode": grid.periodic_endpoint_mode,
        "k_nyquist_min": grid.k_nyquist_min,
        "spectra": {
            "primary_coordinate": "k",
            "secondary_coordinate": "k_over_k_nyquist",
            "radial_range": "0 < k <= k_nyquist_min",
            "subtract_mean_per_sample": bool(cfg.spectra.subtract_mean),
            "num_k_bins": int(cfg.spectra.num_k_bins),
            "periodic_endpoint_convention": (
                "exclude maximum-coordinate plane on x/y/z before FFT"
                if grid.periodic_endpoint_mode == "drop_max"
                else "use all stored grid nodes"
            ),
        },
        "nearest_reference": nearest_summary,
        "channel_summary": _channel_summary(channel_rows, band_rows),
        "physical_summary": _physical_summary(physical_rows),
        "energy_spectrum": _energy_spectrum_summary(
            energy_summary_rows,
            energy_band_rows,
            enabled=True,
            lower_quantile=energy_lower,
            upper_quantile=energy_upper,
        ),
        "standardized_wasserstein": {
            str(row["channel"]): _finite_or_none(
                float(row["wasserstein_1_standardized"])
            )
            for row in standardized_rows
        },
        "outputs": {
            "channel_metrics": "channel_metrics.csv",
            "standardized_wasserstein": "standardized_wasserstein.csv",
            "sample_statistics": "sample_statistics.csv",
            "correlation_matrix": "correlation_matrix.csv",
            "spectra": "spectra.csv",
            "spectral_bands": "spectral_bands.csv",
            "physical_metrics": "physical_metrics.csv",
            "energy_spectrum_per_sample": "energy_spectrum_per_sample.csv",
            "energy_spectrum_summary": "energy_spectrum_summary.csv",
            "energy_spectral_bands": "energy_spectral_bands.csv",
            "nearest_reference": (
                "nearest_reference.csv"
                if bool(cfg.nearest_reference.enabled)
                else None
            ),
            "plots": "plots" if bool(cfg.plots.enabled) else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def _load_grid_3d(
    source_cfg: DictConfig,
    cfg: DictConfig,
) -> tuple[CartesianGrid3D, Path]:
    dataset = instantiate(source_cfg.data)
    if len(dataset) < 1:
        raise ValueError("source dataset is empty")
    sample = dataset[0]
    coords = sample.mesh.coords
    if not isinstance(coords, torch.Tensor) or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("3-D benchmark requires source mesh coordinates with shape [N, 3]")
    mesh_value = OmegaConf.select(source_cfg, "data.mesh_file")
    mesh_file = Path(str(mesh_value)).expanduser().resolve()
    return (
        infer_cartesian_grid_3d(
            coords.detach().cpu().numpy(),
            tolerance=float(cfg.spectra.cartesian_tolerance),
            periodic_endpoint_mode=str(cfg.spectra.periodic_endpoint_mode),
        ),
        mesh_file,
    )


def _standardized_wasserstein_from_source(
    source_summary: dict[str, Any],
    generated: np.ndarray,
    reference: np.ndarray,
    channel_names: tuple[str, ...],
) -> list[dict[str, float | str]]:
    source_run_value = source_summary.get("source_run_dir")
    if not isinstance(source_run_value, str) or not source_run_value:
        raise ValueError("generation summary does not define source_run_dir")
    standardizer_path = Path(source_run_value).expanduser().resolve() / "standardizers.pt"
    if not standardizer_path.is_file():
        raise FileNotFoundError(
            f"source standardizers do not exist: {standardizer_path}"
        )
    payload = torch.load(standardizer_path, map_location="cpu", weights_only=True)
    inputs = payload.get("inputs") if isinstance(payload, dict) else None
    if not isinstance(inputs, dict):
        raise ValueError("source standardizers.pt does not contain input statistics")
    saved_names = tuple(inputs["channel_names"])
    if saved_names != channel_names:
        raise ValueError(
            f"standardizer channels {saved_names} do not match generated channels {channel_names}"
        )
    return standardized_wasserstein_rows(
        generated,
        reference,
        channel_names,
        mean=inputs["mean"].detach().cpu().numpy(),
        scale=inputs["scale"].detach().cpu().numpy(),
    )


def _sample_statistic_rows_3d(
    generated: np.ndarray,
    reference: np.ndarray,
    generated_ids: tuple[str, ...],
    reference_ids: tuple[str, ...],
    channel_names: tuple[str, ...],
    grid: CartesianGrid3D,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for population, values, identifiers, id_role in (
        ("generated", generated, generated_ids, "sampling_key"),
        ("test_reference", reference, reference_ids, "test_sample_id"),
    ):
        for sample_index, identifier in enumerate(identifiers):
            for channel, name in enumerate(channel_names):
                field = values[sample_index, :, channel]
                correlation, difference_rms = nearest_neighbor_spatial_metrics_3d(
                    field,
                    grid,
                )
                rows.append(
                    {
                        "population": population,
                        "sample_index": sample_index,
                        "sample_id": identifier,
                        "sample_id_role": id_role,
                        "channel": name,
                        "spatial_mean": float(np.mean(field)),
                        "spatial_std": float(np.std(field, ddof=0)),
                        "nearest_neighbor_correlation": correlation,
                        "first_difference_rms": difference_rms,
                    }
                )
    return rows


def _snapshot_descriptors_3d(
    samples: np.ndarray,
    channel_names: tuple[str, ...],
    grid: CartesianGrid3D,
    sample_power: np.ndarray,
    k_centers: np.ndarray,
    *,
    bands: dict[str, tuple[float, float]],
    include_channel_correlations: bool,
) -> tuple[tuple[str, ...], np.ndarray]:
    values = np.asarray(samples, dtype=np.float64)
    fraction = np.asarray(k_centers, dtype=np.float64) / grid.k_nyquist_min
    feature_names: list[str] = []
    columns: list[np.ndarray] = []

    for channel, name in enumerate(channel_names):
        channel_values = values[:, :, channel]
        columns.append(np.mean(channel_values, axis=1))
        feature_names.append(f"{name}:spatial_mean")
        columns.append(np.std(channel_values, axis=1, ddof=0))
        feature_names.append(f"{name}:spatial_std")

        neighbor_corr = np.empty(values.shape[0], dtype=np.float64)
        difference_rms = np.empty(values.shape[0], dtype=np.float64)
        for sample_index in range(values.shape[0]):
            neighbor_corr[sample_index], difference_rms[sample_index] = (
                nearest_neighbor_spatial_metrics_3d(
                    channel_values[sample_index],
                    grid,
                )
            )
        columns.append(neighbor_corr)
        feature_names.append(f"{name}:nearest_neighbor_correlation")
        columns.append(difference_rms)
        feature_names.append(f"{name}:first_difference_rms")

        for band_name, (lower, upper) in bands.items():
            mask = (fraction >= lower) & (fraction < upper)
            if upper == 1.0:
                mask = (fraction >= lower) & (fraction <= upper)
            columns.append(np.sum(sample_power[:, channel, :][:, mask], axis=1))
            feature_names.append(f"{name}:spectral_power:{band_name}")

    if include_channel_correlations:
        for first_channel in range(len(channel_names)):
            for second_channel in range(first_channel + 1, len(channel_names)):
                corr = np.empty(values.shape[0], dtype=np.float64)
                for sample_index in range(values.shape[0]):
                    first = values[sample_index, :, first_channel]
                    second = values[sample_index, :, second_channel]
                    first_centered = first - float(np.mean(first))
                    second_centered = second - float(np.mean(second))
                    denominator = float(
                        np.sqrt(
                            np.sum(first_centered**2)
                            * np.sum(second_centered**2)
                        )
                    )
                    corr[sample_index] = (
                        float(np.sum(first_centered * second_centered) / denominator)
                        if denominator > 0.0
                        else float("nan")
                    )
                columns.append(corr)
                feature_names.append(
                    "channel_correlation:"
                    f"{channel_names[first_channel]}|{channel_names[second_channel]}"
                )

    return tuple(feature_names), np.column_stack(columns)


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
