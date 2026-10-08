"""Regressions for historical M33/M39 comparison artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from omegaconf import OmegaConf

from scripts.compare_dinat_self_attention_ablation import (
    _config_differences,
    _load_history,
    _optional_float,
    _optional_ratio,
    _to_plain_config,
)


def test_legacy_history_preserves_losses_without_inventing_epoch_times(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("epoch", "train_flow_velocity_mse", "validation_flow_velocity_mse"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 0,
                "train_flow_velocity_mse": 0.7,
                "validation_flow_velocity_mse": 0.8,
            }
        )

    rows, metric = _load_history(path)
    assert metric == "flow_velocity_mse"
    assert rows[0]["epoch"] == 0
    assert rows[0]["train"] == 0.7
    assert rows[0]["validation"] == 0.8
    assert rows[0]["epoch_compute_seconds"] is None


def test_explicit_epoch_timing_remains_available(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "epoch",
                "train_flow_velocity_mse",
                "validation_flow_velocity_mse",
                "epoch_compute_seconds",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 0,
                "train_flow_velocity_mse": 0.7,
                "validation_flow_velocity_mse": 0.8,
                "epoch_compute_seconds": 2.25,
            }
        )

    rows, metric = _load_history(path)
    assert metric == "flow_velocity_mse"
    assert rows[0]["epoch_compute_seconds"] == 2.25


def test_old_dinat_default_contract_differs_only_by_self_attention(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "m33.yaml"
    candidate = tmp_path / "m39.yaml"
    OmegaConf.save(
        OmegaConf.create({"run_name": "m33", "model": {"hidden_dim": 128}}),
        baseline,
    )
    OmegaConf.save(
        OmegaConf.create(
            {
                "run_name": "m39",
                "model": {
                    "hidden_dim": 128,
                    "sparse_attention_backend": "scatter",
                    "include_self_attention": True,
                },
            }
        ),
        candidate,
    )
    differences = _config_differences(
        _to_plain_config(baseline),
        _to_plain_config(candidate),
    )
    assert set(differences) == {"run_name", "model.include_self_attention"}
    assert differences["model.include_self_attention"] == {
        "baseline": False,
        "candidate": True,
    }


def test_legacy_summary_missing_runtime_metrics_does_not_invent_values() -> None:
    # M33 predates fit-wall-time and CUDA peak-memory fields; generative quality
    # metrics remain comparable even when these efficiency estimates are absent.
    baseline_summary = {
        "run_name": "m33",
        "best_validation_flow_velocity_mse": 0.25,
    }
    candidate_summary = {
        "run_name": "m39",
        "best_validation_flow_velocity_mse": 0.20,
        "fit_wall_seconds": 2500.0,
        "fit_peak_allocated_gib": 5.0,
    }

    assert _optional_float(baseline_summary, "fit_wall_seconds") is None
    assert _optional_float(baseline_summary, "fit_peak_allocated_gib") is None
    assert _optional_float(baseline_summary, "fit_peak_reserved_gib") is None
    assert _optional_ratio(
        _optional_float(candidate_summary, "fit_wall_seconds"),
        _optional_float(baseline_summary, "fit_wall_seconds"),
    ) is None
    assert _optional_ratio(
        _optional_float(candidate_summary, "fit_peak_allocated_gib"),
        _optional_float(baseline_summary, "fit_peak_allocated_gib"),
    ) is None
    assert _optional_ratio(1.0, 0.0) is None
