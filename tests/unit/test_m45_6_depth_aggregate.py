"""M45.6 aggregation qualification: explicit rerun exclusion, no silent dedup."""

from __future__ import annotations

import csv
import json

import pytest

from scripts.m45_6_compare_depth import aggregate


def _write_run(root, job: str, *, depth: int, fm_loss: float) -> None:
    directory = root / job
    diagnostics = directory / "diagnostics"
    diagnostics.mkdir(parents=True)
    modes = ("static", "dynamic")
    summary = {
        "settings": {
            "training_scope": "full_all",
            "fine_pre_layers": depth,
            "fine_post_layers": depth,
            "coarse_static_layers": depth,
            "coarse_dynamic_layers": depth,
            "val_seed": 7002026,
            "seed": 2028,
            "train_samples": 256,
            "steps": 2000,
            "width": 16,
            "refinement": 2,
        },
        "results": [
            {
                "case": "shared_mode",
                "mode": mode,
                "heldout_fm_loss": fm_loss,
                "parameters": 10000 if mode == "static" else 14000,
            }
            for mode in modes
        ],
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    with (diagnostics / "generation_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case", "mode", "variant", "heun_steps",
                "mode_second_moment", "analytic_mode_second_moment",
            ],
        )
        writer.writeheader()
        for mode in modes:
            writer.writerow({
                "case": "shared_mode",
                "mode": mode,
                "variant": "normal",
                "heun_steps": 25,
                "mode_second_moment": 0.0003 if depth == 4 else 0.0001,
                "analytic_mode_second_moment": 0.0015,
            })


def test_aggregation_rejects_duplicate_key_without_explicit_exclusion(tmp_path):
    root = tmp_path / "runs"
    _write_run(root, "job_493366", depth=2, fm_loss=0.55)
    _write_run(root, "job_493370", depth=4, fm_loss=0.4)
    _write_run(root, "job_493421", depth=2, fm_loss=0.65)

    with pytest.raises(ValueError, match="duplicate run"):
        aggregate(root, tmp_path / "strict")


def test_aggregation_excludes_only_explicit_run_and_reports_selection(tmp_path):
    root = tmp_path / "runs"
    _write_run(root, "job_493366", depth=2, fm_loss=0.55)
    _write_run(root, "job_493370", depth=4, fm_loss=0.4)
    _write_run(root, "job_493421", depth=2, fm_loss=0.65)

    output = tmp_path / "canonical"
    report = aggregate(root, output, exclude_jobs=frozenset({"job_493421"}))
    assert report["excluded_jobs"] == ["job_493421"]
    assert report["n_runs"] == 4
    assert report["n_matched"] == 2
    assert report["aggregates"][0]["mean_modal_recovery_depth2_pct"] == pytest.approx(100 / 15)
    with (output / "per_job.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert "job_493421" not in {row["job"] for row in rows}
    assert {row["job"] for row in rows} == {"job_493366", "job_493370"}


def test_excluded_run_must_exist_in_input_root(tmp_path):
    root = tmp_path / "runs"
    _write_run(root, "job_493366", depth=2, fm_loss=0.55)
    with pytest.raises(ValueError, match="excluded jobs not present"):
        aggregate(root, tmp_path / "out", exclude_jobs=frozenset({"job_493421"}))
