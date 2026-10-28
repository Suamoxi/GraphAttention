"""SR-0.4 comparison must fail closed on mismatched source and evaluation geometry."""

import copy

import pytest

from scripts.compare_m44_sr04_diagnostics import compare
from test_m44_sr03_common_support_comparison import _report


FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def _spatial(constrained_sample: dict, increment: float) -> None:
    constrained_sample["residual_spatial_moments"] = {
        "ensemble_mean_residual_rms_by_field": {f: 0.5 for f in FIELDS},
        "ensemble_pointwise_std_rms_by_field": {f: 0.7 for f in FIELDS},
        "reference_residual_spatial_mean_by_field": {f: 0.1 for f in FIELDS},
        "reference_residual_spatial_std_by_field": {f: 1.0 for f in FIELDS},
        "nearest_neighbor_increments": {
            "separation_over_L_ref": 0.125,
            "graph_edge_count": 1600,
            "reference_rms_by_field": {f: 2.0 for f in FIELDS},
            "generated_rms_by_field": {f: increment for f in FIELDS},
        },
    }


def _pair() -> tuple[dict, dict]:
    baseline = _report(True)
    baseline["source_manifest_sha256"] = "source_sha"
    baseline["train_patch_diversity"] = None
    baseline["frozen_scale_provenance"] = None
    baseline["train_evaluation_patch_scope"] = "all_training_patches"
    diverse = copy.deepcopy(baseline)
    diverse["source_run_dir"] = "new_spatially_diversified_run"
    diverse["train_patch_diversity"] = {
        "policy": "snapshot_seeded_shuffled_cycle_v1",
        "pool_size": 16,
        "epoch_cycle": 8,
        "patches_per_snapshot_per_epoch": 2,
    }
    diverse["frozen_scale_provenance"] = {
        "source_run_dir": baseline["source_run_dir"],
        "source_manifest_sha256": baseline["source_manifest_sha256"],
        "source_scales_sha256": "scales_sha",
    }
    diverse["train_evaluation_patch_scope"] = "original_fixed_train_origins_only"
    for split in ("validation", "test"):
        for e in baseline["generation_ensemble"][split]:
            _spatial(e, increment=0.8)
        for e in diverse["generation_ensemble"][split]:
            _spatial(e, increment=1.0)
    return baseline, diverse


def test_pairing_preserves_sources_and_compares_spatial_increments():
    baseline, diverse = _pair()
    previous = copy.deepcopy(baseline)
    result = compare(baseline, diverse)
    assert result["status"] == "m44_sr04_paired_spatial_diversity_comparison_complete"
    assert result["train_patch_schedule"]["pool_size"] == 16
    assert result["flow_matching"]["test"]["diverse_common_mse"] == pytest.approx(
        result["flow_matching"]["test"]["baseline_common_mse"]
    )
    inc = result["generation"]["test"]["spatial_moments"]["per_field"]["rhou"]
    assert inc["baseline_increment_over_DNS"] == pytest.approx(0.4)
    assert inc["diverse_increment_over_DNS"] == pytest.approx(0.5)
    assert baseline == previous


def test_rejects_incorrect_frozen_source_or_evaluation_scope():
    a, b = _pair()
    b["frozen_scale_provenance"]["source_run_dir"] = "wrong_model"
    with pytest.raises(ValueError, match="frozen scales are not sourced"):
        compare(a, b)
    a, b = _pair()
    b["train_evaluation_patch_scope"] = "all_training_patches"
    with pytest.raises(ValueError, match="fixed origins"):
        compare(a, b)


def test_rejects_missing_spatial_metrics_or_wrong_reference_separation():
    a, b = _pair()
    del a["generation_ensemble"]["test"][0]["residual_spatial_moments"]
    with pytest.raises(ValueError, match="rerun diagnostics"):
        compare(a, b)
    a, b = _pair()
    b["generation_ensemble"]["validation"][0][
        "residual_spatial_moments"
    ]["nearest_neighbor_increments"]["separation_over_L_ref"] = 0.25
    with pytest.raises(ValueError, match="mismatched common-reference"):
        compare(a, b)
