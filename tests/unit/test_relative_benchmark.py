import csv
import json
from pathlib import Path

import pytest

from graph_attention.evaluation.relative_benchmark import compare_generation_benchmarks


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_benchmark(path: Path, *, wasserstein: float, spectral_ratio: float) -> None:
    path.mkdir()
    summary = {
        "benchmark": "generation_distribution_v2",
        "run_name": path.name,
        "source_model": "ExampleModel",
        "source_model_parameters": 10,
        "source_best_epoch": 5,
        "sample_space": "nondimensional",
        "reference_population": "test",
        "nodes_per_sample": 4,
        "channel_names": ["rho.value"],
        "grid_shape_2d": [2, 2],
        "nearest_reference": {
            "generated_to_test_mean_over_test_to_test_mean": 1.2,
            "test_to_generated_mean_over_test_to_test_mean": 1.1,
        },
    }
    (path / "summary.json").write_text(json.dumps(summary))

    _write_csv(
        path / "channel_metrics.csv",
        [
            "channel",
            "generated_mean",
            "reference_mean",
            "mean_bias",
            "normalized_mean_bias",
            "generated_std",
            "reference_std",
            "std_ratio",
            "wasserstein_1",
        ],
        [
            {
                "channel": "rho.value",
                "generated_mean": 1.01,
                "reference_mean": 1.0,
                "mean_bias": 0.01,
                "normalized_mean_bias": 0.005,
                "generated_std": 2.1,
                "reference_std": 2.0,
                "std_ratio": 1.05,
                "wasserstein_1": wasserstein,
            }
        ],
    )
    _write_csv(
        path / "sample_statistics.csv",
        [
            "sample_index",
            "sample_id",
            "sample_id_role",
            "population",
            "channel",
            "spatial_mean",
            "spatial_std",
            "nearest_neighbor_correlation",
            "first_difference_rms",
        ],
        [
            {
                "sample_index": 0,
                "sample_id": "gen_0",
                "sample_id_role": "sampling_key",
                "population": "generated",
                "channel": "rho.value",
                "spatial_mean": 1.0,
                "spatial_std": 1.1,
                "nearest_neighbor_correlation": 0.8,
                "first_difference_rms": 1.2,
            },
            {
                "sample_index": 0,
                "sample_id": "test_0",
                "sample_id_role": "test_sample_id",
                "population": "test_reference",
                "channel": "rho.value",
                "spatial_mean": 1.0,
                "spatial_std": 1.0,
                "nearest_neighbor_correlation": 0.9,
                "first_difference_rms": 1.0,
            },
        ],
    )
    _write_csv(
        path / "correlation_matrix.csv",
        ["population", "row_channel", "column_channel", "correlation"],
        [
            {
                "population": "generated",
                "row_channel": "rho.value",
                "column_channel": "rho.value",
                "correlation": 1.0,
            },
            {
                "population": "test_reference",
                "row_channel": "rho.value",
                "column_channel": "rho.value",
                "correlation": 1.0,
            },
        ],
    )
    _write_csv(
        path / "spectra.csv",
        [
            "channel",
            "k",
            "k_over_k_nyquist",
            "generated_power",
            "reference_power",
            "generated_over_reference",
        ],
        [
            {
                "channel": "rho.value",
                "k": 1.0,
                "k_over_k_nyquist": 0.5,
                "generated_power": spectral_ratio,
                "reference_power": 1.0,
                "generated_over_reference": spectral_ratio,
            }
        ],
    )
    _write_csv(
        path / "spectral_bands.csv",
        [
            "channel",
            "band",
            "k_fraction_min",
            "k_fraction_max",
            "generated_power",
            "reference_power",
            "generated_over_reference",
        ],
        [
            {
                "channel": "rho.value",
                "band": "mid",
                "k_fraction_min": 0.25,
                "k_fraction_max": 0.5,
                "generated_power": spectral_ratio,
                "reference_power": 1.0,
                "generated_over_reference": spectral_ratio,
            }
        ],
    )
    _write_csv(
        path / "nearest_reference.csv",
        ["direction", "source_id", "nearest_id", "distance"],
        [],
    )
    _write_csv(
        path / "physical_metrics.csv",
        [
            "metric",
            "comparison",
            "generated",
            "reference",
            "difference",
            "normalized_difference",
            "generated_over_reference",
        ],
        [],
    )


def test_relative_comparison_reports_absolute_and_relative_error_change(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    output = tmp_path / "comparison"
    _make_benchmark(baseline, wasserstein=6.0e-7, spectral_ratio=2.0)
    _make_benchmark(candidate, wasserstein=2.0e-7, spectral_ratio=1.5)

    summary = compare_generation_benchmarks(baseline, candidate, output)

    assert summary["num_common_metrics"] > 0
    with (output / "relative_comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    wasserstein = next(
        row
        for row in rows
        if row["metric"] == "wasserstein_1_over_reference_std"
        and row["channel"] == "rho.value"
    )

    # Reference std is 2, so the errors are 3e-7 and 1e-7.  The candidate is
    # 66.7% better relatively, but the absolute reduction remains only 2e-7.
    assert float(wasserstein["baseline_error"]) == pytest.approx(3.0e-7)
    assert float(wasserstein["candidate_error"]) == pytest.approx(1.0e-7)
    assert float(wasserstein["absolute_error_reduction"]) == pytest.approx(2.0e-7)
    assert float(wasserstein["relative_error_reduction_percent"]) == pytest.approx(66.6666667)


def test_relative_comparison_rejects_incompatible_reference_space(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    output = tmp_path / "comparison"
    _make_benchmark(baseline, wasserstein=0.2, spectral_ratio=2.0)
    _make_benchmark(candidate, wasserstein=0.1, spectral_ratio=1.5)

    summary_path = candidate / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["grid_shape_2d"] = [4, 1]
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="not directly comparable"):
        compare_generation_benchmarks(baseline, candidate, output)
