"""M45.7.3 qualification: paired seeds, within-field moments and aggregation."""

from __future__ import annotations

import csv
import json

import pytest
import torch

from graph_attention.data.two_scale_synthetic import SyntheticQB, make_synthetic_pair
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from scripts.m45_checkpoint_diagnostics import _fixed_coarse_references
from scripts.m45_7_3_condition_sweep import (
    _check_source, _field_records, _generation_seeds, _summarize_fields,
)
from scripts.m45_7_3_aggregate import _paired_rows, aggregate


def test_generation_seeds_are_paired_and_separated_by_coarse_index():
    seeds0 = _generation_seeds(base=53177, validation_index=0, count=4)
    seeds1 = _generation_seeds(base=53177, validation_index=1, count=4)
    assert seeds0 == [1053177 + draw * 3277 for draw in range(4)]
    assert len(set(seeds0 + seeds1)) == 8
    with pytest.raises(ValueError, match="count"):
        _generation_seeds(base=53177, validation_index=0, count=1)


def test_source_validation_rejects_mismatched_seed_and_policy():
    settings = dict(
        operator="block_average", source_subspace="nullspace",
        cases=["shared_mode"], train_samples=24, val_samples=8,
        steps=10000, coarse_side=4, refinement=2, width=16, heads=4,
        supervision="oracle", seed=2026, val_seed=7002026,
        residual_policy="fixed",
    )
    result = dict(mode="static", supervision="oracle",
                  residual_policy="fixed", training_steps=10000)
    _check_source(settings, result, seed=2026, policy="fixed")
    with pytest.raises(ValueError, match="seed"):
        _check_source(settings, result, seed=2027, policy="fixed")
    with pytest.raises(ValueError, match="residual_policy"):
        _check_source(settings, result, seed=2026, policy="resample")


def test_field_metrics_center_before_variance_and_preserve_pairing():
    mesh = build_hit_two_scale_geometry(coarse_side=4, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    pair = make_synthetic_pair(
        qb, case="shared_mode", sample_id=100000, seed=7002026,
    )
    ref = _fixed_coarse_references(
        qb, pair.coarse, "shared_mode", seed=753177, count=8,
    )
    offset = torch.zeros_like(ref[0])
    offset[:, 1] = 0.04 * pair.mode[:, 0]
    generated = 0.5 * ref + offset[None, :, :]
    row, correlations, amplitudes = _field_records(
        qb, pair, generated, ref, seed=2026, policy="fixed",
        validation_index=0, heun_steps=2,
    )
    assert row["generated_mode_variance"] == pytest.approx(
        0.25 * row["reference_mode_variance"], rel=1e-5,
    )
    assert row["generated_mode_mean"] == pytest.approx(
        0.5 * row["reference_mode_mean"] + 0.04, rel=1e-5,
    )
    assert row["generated_mode_second_moment"] > row["generated_mode_variance"]
    assert row["recovery_vs_analytic"] == pytest.approx(
        row["generated_mode_variance"] / row["analytic_mode_variance"],
    )
    assert row["max_Q_generated"] < 1e-6
    assert len(correlations) == 2 * 3 * (qb.nf // 2 + 1)
    assert len(amplitudes) == 16
    summary = _summarize_fields([row, row])
    assert summary["validation_fields"] == 2
    assert summary["recovery_vs_analytic_std_across_fields"] == 0.0


def _record(seed: int, policy: str, index: int) -> dict:
    recovery = 0.25 + 0.01 * index + (0.1 if policy == "resample" else 0.0)
    return {
        "seed": seed,
        "residual_policy": policy,
        "validation_index": index,
        "sample_id": 100000 + index,
        "heun_steps": 3,
        "generated_samples": 4,
        "analytic_mode_variance": 0.0015,
        "reference_mode_variance": 0.0014 + 0.00001 * index,
        "reference_mode_mean": -0.001,
        "generated_mode_mean": 0.01 if policy == "fixed" else 0.0,
        "recovery_vs_analytic": recovery,
        "recovery_vs_empirical": recovery * 0.0015 / (0.0014 + 0.00001 * index),
        "generated_corr_x_lag1": -0.25,
        "generated_corr_x_lag4": 0.4 + (0.1 if policy == "resample" else 0.0),
        "max_Q_generated": 1e-8,
    }


def test_paired_aggregation_checks_completeness_and_duplicate_rows():
    rows = [
        _record(seed, policy, i)
        for seed in (2026, 2027, 2028)
        for policy in ("fixed", "resample")
        for i in range(2)
    ]
    paired = _paired_rows(rows, expected_fields=2)
    assert len(paired) == 6
    assert all(x["delta_recovery_pp"] == pytest.approx(10.0) for x in paired)
    with pytest.raises(ValueError, match="duplicate"):
        _paired_rows(rows + [rows[0]], expected_fields=2)
    with pytest.raises(ValueError, match="incomplete"):
        _paired_rows(rows[:-1], expected_fields=2)
    inconsistent = [dict(x) for x in rows]
    inconsistent[-1]["reference_mode_variance"] = 8.0
    with pytest.raises(ValueError, match="reference statistics"):
        _paired_rows(inconsistent, expected_fields=2)


def test_full_aggregation_reads_only_complete_six_checkpoint_matrix(tmp_path):
    root = tmp_path / "eval"
    for seed in (2026, 2027, 2028):
        for task, policy in enumerate(("fixed", "resample")):
            job = root / f"task_{seed}_{task}_{policy}"
            job.mkdir(parents=True)
            settings = {
                "seed": seed, "residual_policy": policy,
                "supervision": "oracle", "steps": 10000, "val_seed": 7002026,
            }
            eval_info = {
                "training_seed": seed, "residual_policy": policy,
                "validation_fields": 2, "heun_steps": 3,
                "generated_samples": 4, "eval_seed": 53177,
            }
            (job / "summary.json").write_text(json.dumps({
                "schema": "graph_attention.m45_7_3_condition_sweep.v1",
                "checkpoint_settings": settings, "evaluation": eval_info,
            }))
            records = [_record(seed, policy, i) for i in range(2)]
            with (job / "per_field.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)

    output = tmp_path / "aggregated"
    result = aggregate(
        input_dir=root, output_dir=output, expected_fields=2,
        samples=4, heun_steps=3,
    )
    assert result["results"]["paired_resampling_gain_pp_mean_across_seeds"] == pytest.approx(10.0)
    assert len(list(csv.DictReader((output / "paired_per_field.csv").open()))) == 6
    assert len(list(csv.DictReader((output / "per_seed.csv").open()))) == 3
    assert len(list(csv.DictReader((output / "per_validation_field.csv").open()))) == 2
    with pytest.raises(FileExistsError, match="existing"):
        aggregate(
            input_dir=root, output_dir=output, expected_fields=2,
            samples=4, heun_steps=3,
        )
