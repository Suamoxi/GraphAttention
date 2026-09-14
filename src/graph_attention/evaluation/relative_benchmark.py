"""Dimensionless comparison of existing generation-distribution benchmarks.

This module deliberately post-processes persisted benchmark outputs instead of
rerunning generation or changing the underlying generation_distribution_v2
metrics.  Every reported comparison quantity is an error magnitude with ideal
value zero so model-to-model relative improvements can always be read together
with the absolute dimensionless error scale.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_REQUIRED_FILES = (
    "summary.json",
    "channel_metrics.csv",
    "sample_statistics.csv",
    "correlation_matrix.csv",
    "spectra.csv",
    "spectral_bands.csv",
    "nearest_reference.csv",
    "physical_metrics.csv",
)


def compare_generation_benchmarks(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    """Compare two persisted generation benchmarks using dimensionless errors.

    ``baseline`` and ``candidate`` retain their ordinary experimental meaning:
    a negative ``absolute_error_change`` means the candidate reduced the error.
    The relative percentage is never presented without the two absolute error
    magnitudes and their absolute difference.
    """

    epsilon = float(eps)
    if epsilon <= 0.0:
        raise ValueError("eps must be positive")

    baseline_path = _benchmark_directory(baseline_dir, "baseline_dir")
    candidate_path = _benchmark_directory(candidate_dir, "candidate_dir")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"relative benchmark output already exists: {destination}; "
                "set overwrite=true explicitly"
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)

    baseline_summary, baseline_rows = relative_error_rows(baseline_path, eps=epsilon)
    candidate_summary, candidate_rows = relative_error_rows(candidate_path, eps=epsilon)

    _validate_compatible_benchmarks(baseline_summary, candidate_summary)

    baseline_by_id = {str(row["metric_id"]): row for row in baseline_rows}
    candidate_by_id = {str(row["metric_id"]): row for row in candidate_rows}
    common_ids = sorted(set(baseline_by_id).intersection(candidate_by_id))
    missing_in_candidate = sorted(set(baseline_by_id).difference(candidate_by_id))
    missing_in_baseline = sorted(set(candidate_by_id).difference(baseline_by_id))
    if not common_ids:
        raise ValueError("benchmarks expose no common relative metrics")

    comparison_rows: list[dict[str, Any]] = []
    for metric_id in common_ids:
        baseline_row = baseline_by_id[metric_id]
        candidate_row = candidate_by_id[metric_id]
        baseline_error = float(baseline_row["value"])
        candidate_error = float(candidate_row["value"])
        change = candidate_error - baseline_error
        reduction = baseline_error - candidate_error
        relative_reduction = (
            reduction / baseline_error if abs(baseline_error) > epsilon else float("nan")
        )
        comparison_rows.append(
            {
                "metric_id": metric_id,
                "category": baseline_row["category"],
                "metric": baseline_row["metric"],
                "channel": baseline_row["channel"],
                "band": baseline_row["band"],
                "baseline_error": baseline_error,
                "candidate_error": candidate_error,
                "absolute_error_change": change,
                "absolute_error_reduction": reduction,
                "relative_error_reduction_fraction": relative_reduction,
                "relative_error_reduction_percent": 100.0 * relative_reduction,
                "candidate_better": candidate_error < baseline_error,
                "notes": baseline_row["notes"],
            }
        )

    _write_csv(destination / "baseline_relative_metrics.csv", baseline_rows)
    _write_csv(destination / "candidate_relative_metrics.csv", candidate_rows)
    _write_csv(destination / "relative_comparison.csv", comparison_rows)

    result = {
        "benchmark": "generation_distribution_relative_comparison_v1",
        "baseline": _source_metadata(baseline_path, baseline_summary),
        "candidate": _source_metadata(candidate_path, candidate_summary),
        "metric_semantics": {
            "all_values_are_dimensionless_error_magnitudes": True,
            "ideal_error": 0.0,
            "absolute_error_change": "candidate_error - baseline_error; negative is better",
            "absolute_error_reduction": "baseline_error - candidate_error; positive is better",
            "relative_error_reduction": (
                "(baseline_error - candidate_error) / baseline_error; interpret only "
                "together with the absolute error magnitudes"
            ),
            "composite_score": None,
            "composite_score_reason": (
                "heterogeneous physical/statistical metrics are not assigned arbitrary weights"
            ),
        },
        "num_common_metrics": len(common_ids),
        "missing_in_candidate": missing_in_candidate,
        "missing_in_baseline": missing_in_baseline,
        "outputs": {
            "baseline_relative_metrics": "baseline_relative_metrics.csv",
            "candidate_relative_metrics": "candidate_relative_metrics.csv",
            "relative_comparison": "relative_comparison.csv",
        },
    }
    (destination / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def relative_error_rows(
    benchmark_dir: str | Path,
    *,
    eps: float = 1.0e-12,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return dimensionless error magnitudes for one persisted benchmark."""

    epsilon = float(eps)
    if epsilon <= 0.0:
        raise ValueError("eps must be positive")
    path = _benchmark_directory(benchmark_dir, "benchmark_dir")
    summary = json.loads((path / "summary.json").read_text())

    channel_rows = _read_csv(path / "channel_metrics.csv")
    sample_rows = _read_csv(path / "sample_statistics.csv")
    correlation_rows = _read_csv(path / "correlation_matrix.csv")
    spectrum_rows = _read_csv(path / "spectra.csv")
    band_rows = _read_csv(path / "spectral_bands.csv")
    physical_rows = _read_csv(path / "physical_metrics.csv")

    rows: list[dict[str, Any]] = []
    rows.extend(_marginal_relative_rows(channel_rows, eps=epsilon))
    rows.extend(_spectral_relative_rows(spectrum_rows, band_rows, eps=epsilon))
    rows.extend(_local_relative_rows(sample_rows, eps=epsilon))
    rows.extend(_correlation_relative_rows(correlation_rows))
    rows.extend(_nearest_relative_rows(summary))
    rows.extend(_physical_relative_rows(physical_rows))
    return summary, rows


def _marginal_relative_rows(
    rows: list[dict[str, str]],
    *,
    eps: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    aggregate: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        channel = row["channel"]
        reference_std = _float(row["reference_std"])
        if not math.isfinite(reference_std) or reference_std <= eps:
            continue
        wasserstein = _float(row["wasserstein_1"])
        mean_bias = _float(row["mean_bias"])
        std_ratio = _float(row["std_ratio"])
        metrics = {
            "wasserstein_1_over_reference_std": abs(wasserstein) / reference_std,
            "abs_mean_bias_over_reference_std": abs(mean_bias) / reference_std,
            "abs_std_ratio_minus_one": abs(std_ratio - 1.0),
        }
        for metric, value in metrics.items():
            if math.isfinite(value):
                result.append(
                    _row(
                        "marginal",
                        metric,
                        value,
                        channel=channel,
                        aggregation="node-pooled channel population",
                    )
                )
                aggregate[metric].append(value)

    for metric, values in aggregate.items():
        result.append(
            _row(
                "marginal",
                metric,
                _mean(values),
                channel="__mean__",
                aggregation="arithmetic mean over channels",
            )
        )
    return result


def _spectral_relative_rows(
    spectrum_rows: list[dict[str, str]],
    band_rows: list[dict[str, str]],
    *,
    eps: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    log_errors: list[float] = []
    for row in spectrum_rows:
        generated = _float(row["generated_power"])
        reference = _float(row["reference_power"])
        if not math.isfinite(generated) or not math.isfinite(reference):
            continue
        log_errors.append(math.log((max(generated, 0.0) + eps) / (max(reference, 0.0) + eps)))
    if log_errors:
        result.append(
            _row(
                "spectrum",
                "rms_log_power_ratio",
                math.sqrt(_mean([value * value for value in log_errors])),
                channel="__all__",
                band="__all__",
                aggregation="RMS over channels and radial k bins",
                notes="symmetric for reciprocal over/under-prediction in log space",
            )
        )

    by_band: dict[str, list[float]] = defaultdict(list)
    for row in band_rows:
        ratio = _float(row["generated_over_reference"])
        if not math.isfinite(ratio):
            continue
        error = abs(ratio - 1.0)
        band = row["band"]
        channel = row["channel"]
        result.append(
            _row(
                "spectrum",
                "abs_band_power_ratio_minus_one",
                error,
                channel=channel,
                band=band,
                aggregation="population-mean radial spectral power within band",
            )
        )
        by_band[band].append(error)
    for band, values in by_band.items():
        result.append(
            _row(
                "spectrum",
                "abs_band_power_ratio_minus_one",
                _mean(values),
                channel="__mean__",
                band=band,
                aggregation="arithmetic mean over channels",
            )
        )
    return result


def _local_relative_rows(
    rows: list[dict[str, str]],
    *,
    eps: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    fields = (
        "spatial_std",
        "nearest_neighbor_correlation",
        "first_difference_rms",
    )
    for row in rows:
        population = row["population"]
        channel = row["channel"]
        for field in fields:
            value = _float(row[field])
            if math.isfinite(value):
                grouped[(population, channel, field)].append(value)

    channels = sorted({channel for population, channel, _ in grouped if population == "generated"})
    aggregates: dict[str, list[float]] = defaultdict(list)
    for channel in channels:
        values: dict[str, tuple[float, float]] = {}
        for field in fields:
            generated_values = grouped.get(("generated", channel, field), [])
            reference_values = grouped.get(("test_reference", channel, field), [])
            if generated_values and reference_values:
                values[field] = (_mean(generated_values), _mean(reference_values))

        if "spatial_std" in values and abs(values["spatial_std"][1]) > eps:
            error = abs(values["spatial_std"][0] / values["spatial_std"][1] - 1.0)
            result.append(
                _row(
                    "local",
                    "spatial_std_ratio_error",
                    error,
                    channel=channel,
                    aggregation="ratio of population-mean per-sample spatial standard deviations",
                )
            )
            aggregates["spatial_std_ratio_error"].append(error)

        if "nearest_neighbor_correlation" in values:
            error = abs(
                values["nearest_neighbor_correlation"][0]
                - values["nearest_neighbor_correlation"][1]
            )
            result.append(
                _row(
                    "local",
                    "nearest_neighbor_correlation_abs_error",
                    error,
                    channel=channel,
                    aggregation="difference of population-mean per-sample correlations",
                )
            )
            aggregates["nearest_neighbor_correlation_abs_error"].append(error)

        if "first_difference_rms" in values and abs(values["first_difference_rms"][1]) > eps:
            error = abs(
                values["first_difference_rms"][0] / values["first_difference_rms"][1] - 1.0
            )
            result.append(
                _row(
                    "local",
                    "first_difference_rms_ratio_error",
                    error,
                    channel=channel,
                    aggregation="ratio of population-mean per-sample first-difference RMS",
                )
            )
            aggregates["first_difference_rms_ratio_error"].append(error)

    for metric, values in aggregates.items():
        result.append(
            _row(
                "local",
                metric,
                _mean(values),
                channel="__mean__",
                aggregation="arithmetic mean over channels",
            )
        )
    return result


def _correlation_relative_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_population: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for row in rows:
        first = row["row_channel"]
        second = row["column_channel"]
        if first == second:
            continue
        pair = tuple(sorted((first, second)))
        by_population[row["population"]][pair] = _float(row["correlation"])

    generated = by_population.get("generated", {})
    reference = by_population.get("test_reference", {})
    errors = [
        generated[pair] - reference[pair]
        for pair in sorted(set(generated).intersection(reference))
        if math.isfinite(generated[pair]) and math.isfinite(reference[pair])
    ]
    if not errors:
        return []
    return [
        _row(
            "cross_channel",
            "offdiagonal_correlation_rmse",
            math.sqrt(_mean([value * value for value in errors])),
            channel="__all_pairs__",
            aggregation="RMSE over unique off-diagonal channel pairs",
        )
    ]


def _nearest_relative_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    nearest = summary.get("nearest_reference")
    if not isinstance(nearest, dict):
        return []
    keys = (
        "generated_to_test_mean_over_test_to_test_mean",
        "test_to_generated_mean_over_test_to_test_mean",
    )
    rows: list[dict[str, Any]] = []
    for key in keys:
        raw = nearest.get(key)
        if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            continue
        rows.append(
            _row(
                "nearest_reference",
                f"{key}_calibration_error",
                abs(float(raw) - 1.0),
                aggregation="mean NN distance calibrated by real-to-real leave-one-out mean",
                notes=(
                    "closeness to one is a calibration diagnostic, not a monotonic standalone "
                    "quality score; inspect the underlying fidelity/coverage distances too"
                ),
            )
        )
    return rows


def _physical_relative_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        metric = row["metric"]
        comparison = row["comparison"]
        if comparison == "bias_over_reference_std":
            value = abs(_float(row["normalized_difference"]))
        elif comparison == "generated_over_reference":
            value = abs(_float(row["generated_over_reference"]) - 1.0)
        elif comparison == "absolute_difference":
            value = abs(_float(row["difference"]))
        else:
            continue
        if math.isfinite(value):
            result.append(
                _row(
                    "physics",
                    f"{metric}_error",
                    value,
                    aggregation=comparison,
                )
            )
    return result


def _validate_compatible_benchmarks(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    checks = (
        "sample_space",
        "reference_population",
        "nodes_per_sample",
        "channel_names",
        "grid_shape_2d",
    )
    mismatches = [name for name in checks if baseline.get(name) != candidate.get(name)]
    if mismatches:
        raise ValueError(
            "generation benchmarks are not directly comparable; mismatched fields: "
            f"{mismatches}"
        )


def _benchmark_directory(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not an accessible directory: {path}")
    missing = [filename for filename in _REQUIRED_FILES if not (path / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"{name} is missing benchmark outputs: {missing}")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.stat().st_size == 0:
        return []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        rows: list[dict[str, str]] = []
        for raw in reader:
            normalized = {
                str(key).strip(): ("" if value is None else str(value).strip())
                for key, value in raw.items()
                if key is not None
            }
            rows.append(normalized)
        return rows


def _source_metadata(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_dir": str(path),
        "benchmark": summary.get("benchmark"),
        "run_name": summary.get("run_name"),
        "source_model": summary.get("source_model"),
        "source_model_parameters": summary.get("source_model_parameters"),
        "source_best_epoch": summary.get("source_best_epoch"),
    }


def _row(
    category: str,
    metric: str,
    value: float,
    *,
    channel: str = "",
    band: str = "",
    aggregation: str,
    notes: str = "",
) -> dict[str, Any]:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"relative metric '{metric}' must be a finite non-negative error")
    channel_label = channel or "__all__"
    band_label = band or "__all__"
    return {
        "metric_id": f"{category}:{metric}:{channel_label}:{band_label}",
        "category": category,
        "metric": metric,
        "channel": channel_label,
        "band": band_label,
        "value": value,
        "ideal_value": 0.0,
        "aggregation": aggregation,
        "notes": notes,
    }


def _float(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
