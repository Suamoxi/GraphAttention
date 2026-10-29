"""M44 SR-0.5 all-origins diagnostics and per-snapshot second moments."""

import math
from argparse import Namespace

import pytest
import torch

from scripts.diagnose_m44_sr0 import (
    _snapshot_generation_moments, run as diagnose,
)
from scripts.prepare_m44_sr05_continuation import prepare
from scripts.train_m44_sr0 import run as train
from test_m44_sr0_training_runner import _archive


def _entry(sid: str, mean: float, variance: float, target: float = 4.0) -> dict:
    from graph_attention.tasks.sr0_evaluation import STATE_FIELDS

    return {
        "sample_id": sid,
        "draws": 2,
        "residual_spatial_moments": {
            "second_moment_decomposition": {
                "ensemble_mean_square_by_field": {f: mean for f in STATE_FIELDS},
                "stochastic_variance_by_field": {f: variance for f in STATE_FIELDS},
                "generated_second_moment_by_field": {f: mean + variance for f in STATE_FIELDS},
                "dns_reference_second_moment_by_field": {f: target for f in STATE_FIELDS},
            }
        },
    }


def test_pooled_snapshot_statistics_obey_exact_second_moment_identity() -> None:
    entries = [
        _entry("case:t0/sr0_patch_0_0_0", 0.25, 0.75),
        _entry("case:t0/sr0_patch_8_8_8", 0.75, 1.25),
        _entry("case:t1/sr0_patch_0_0_0", 0.16, 0.64),
    ]
    grouped = _snapshot_generation_moments(entries)
    assert grouped["case:t0"]["patches"] == 2
    mom = grouped["case:t0"]["fields"]["rhou"]
    assert mom["ensemble_mean_square"] == pytest.approx(0.5)
    assert mom["stochastic_variance"] == pytest.approx(1.0)
    assert mom["generated_second_moment"] == pytest.approx(1.5)
    assert mom["generated_rms_over_dns"] == pytest.approx(math.sqrt(1.5 / 4.0))
    assert mom["ensemble_mean_rms_over_dns"] == pytest.approx(math.sqrt(0.5 / 4.0))
    assert mom["ensemble_std_rms_over_dns"] == pytest.approx(0.5)
    assert mom["mean_fraction_of_generated_second_moment"] == pytest.approx(1/3)
    assert grouped["case:t1"]["patches"] == 1
    with pytest.raises(ValueError, match="repeated source"):
        _snapshot_generation_moments([entries[0], entries[0]])


def test_read_only_all_origins_diagnostic_on_continued_checkpoint(tmp_path) -> None:
    root, refs = _archive(tmp_path, side=15)
    original = tmp_path / "source"
    args = Namespace(
        root=root, reference_dir=refs, case="Re54/Ma0.3", side=15,
        patch_cells=4, refinement=2, patches_per_snapshot=2,
        boundary_margin=1, max_snapshots=8, train_fraction=0.70,
        validation_fraction=0.15, embargo_snapshots=1,
        epochs=1, resume=False, hidden_dim=16, layers=2, heads=2,
        lr=1e-3, seed=37, time_embedding_scale=1000.0,
        condition_mode="interpolated_only", point_sample_constraint=True,
        absolute_coordinate_embedding=False, sparse_attention_backend="scatter",
        checkpointing=False, sample_steps=2, device="cpu", output=original,
    )
    train(args)
    extended = tmp_path / "continued"
    prepare(
        original, extended, mode="baseline", initial_epochs=1,
        target_epochs=2, production_profile=False,
    )
    train(Namespace(**{**vars(args), "output": extended, "resume": True, "epochs": 2}))
    report = diagnose(Namespace(
        run_dir=extended, output=tmp_path / "diag", device="cpu",
        draws=2, sample_steps=2, max_generation_patches=1,
        path_repeats=2, all_generation_origins=True,
    ))
    assert report["all_generation_origins"] is True
    assert report["completed_training_epochs"] == 2
    assert report["continuation_provenance"]["source_run_dir"] == str(original.resolve())
    for split in ("validation", "test"):
        assert len(report["generation_ensemble"][split]) == 2
        groups = report["generation_ensemble_by_snapshot"][split]
        assert len(groups) == 1
        only = next(iter(groups.values()))
        assert only["patches"] == 2
        assert only["draws_per_patch"] == 2
        assert len(only["sample_ids"]) == 2
        for field in ("rho", "rhou", "rhov", "rhow", "rhoE"):
            mom = only["fields"][field]
            assert mom["ensemble_mean_square"] >= 0
            assert mom["stochastic_variance"] >= 0
            assert math.isclose(
                mom["ensemble_mean_square"] + mom["stochastic_variance"],
                mom["generated_second_moment"], rel_tol=1e-5, abs_tol=1e-7,
            )
    assert max(
        max(entry["raw_point_max_abs_by_field"].values())
        for entry in report["generation_ensemble"]["test"]
    ) == 0.0
