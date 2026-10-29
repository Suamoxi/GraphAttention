"""M44 SR-0.5: strict paired continuation diagnostics on two patch origins."""

import copy

import pytest

from scripts.compare_m44_sr05_diagnostics import compare
from test_m44_sr04_comparison import _pair


def _fixture() -> tuple[dict, dict]:
    original_base, original_diverse = _pair()
    baseline = copy.deepcopy(original_base)
    diverse = copy.deepcopy(original_diverse)
    base_parent = baseline["source_run_dir"]
    diverse_parent = diverse["source_run_dir"]
    baseline["source_run_dir"] = "m44_sr05/baseline_job_1"
    diverse["source_run_dir"] = "m44_sr05/diverse_job_2"
    baseline["all_generation_origins"] = True
    diverse["all_generation_origins"] = True
    baseline["completed_training_epochs"] = 150
    diverse["completed_training_epochs"] = 150
    baseline["best_epoch"] = 101
    diverse["best_epoch"] = 72
    diverse["source_manifest_sha256"] = "diverse_source_sha"
    basis = {
        "protocol": "M44_SR05_IMMUTABLE_CONTINUATION_V1",
        "initial_epochs": 50,
        "target_epochs": 150,
        "optimizer_updates_per_epoch": 30,
        "initial_optimizer_updates": 1500,
        "target_optimizer_updates": 4500,
    }
    baseline["continuation_provenance"] = {
        **basis, "mode": "baseline", "source_run_dir": base_parent,
        "source_manifest_sha256": "source_sha",
        "training_data_policy": "2 fixed origins per snapshot",
        "source_frozen_baseline_run": None,
    }
    diverse["continuation_provenance"] = {
        **basis, "mode": "diverse", "source_run_dir": diverse_parent,
        "source_manifest_sha256": "diverse_source_sha",
        "training_data_policy": "16-origin cyclic pool, 2 per snapshot per epoch",
        "source_frozen_baseline_run": base_parent,
    }
    for role, report in (("baseline", baseline), ("diverse", diverse)):
        for split in ("validation", "test"):
            report["repeated_flow_matching"][split]["num_patches"] = 2
        report["draws_per_generation_patch"] = 16
        report["generation_heun_steps"] = 50
        for split in ("validation", "test"):
            first = report["generation_ensemble"][split][0]
            first["draws"] = 16
            other = copy.deepcopy(first)
            other["sample_id"] = "snapshotA/sr0_patch_8_8_8"
            report["generation_ensemble"][split].append(other)
            if role == "baseline":
                mean_sq, variance = 0.25, 0.75
            else:
                mean_sq, variance = 0.36, 1.08
            ratio = ((mean_sq + variance) / 4.0) ** 0.5
            snap = {
                "ensemble_mean_square": mean_sq,
                "stochastic_variance": variance,
                "generated_second_moment": mean_sq + variance,
                "dns_reference_second_moment": 4.0,
                "ensemble_mean_rms_over_dns": (mean_sq / 4.0) ** 0.5,
                "ensemble_std_rms_over_dns": (variance / 4.0) ** 0.5,
                "generated_rms_over_dns": ratio,
                "mean_fraction_of_generated_second_moment": (
                    mean_sq / (mean_sq + variance)
                ),
                "variance_fraction_of_generated_second_moment": (
                    variance / (mean_sq + variance)
                ),
            }
            report.setdefault("generation_ensemble_by_snapshot", {})[split] = {
                "snapshotA": {
                    "patches": 2,
                    "sample_ids": sorted(e["sample_id"] for e in report["generation_ensemble"][split]),
                    "draws_per_patch": 16,
                    "fields": {f: dict(snap) for f in ("rho", "rhou", "rhov", "rhow", "rhoE")},
                }
            }
    return baseline, diverse


def test_matched_budget_pair_has_provenance_604_support_and_snapshot_moments():
    a, b = _fixture()
    unchanged = copy.deepcopy(a)
    result = compare(a, b)
    assert result["status"] == "m44_sr05_matched_budget_pair_complete"
    assert result["total_optimizer_updates"] == 4500
    assert result["total_epochs"] == 150
    assert result["flow_matching"]["test"]["free_nodes_per_patch"] == 604
    assert result["flow_matching"]["test"]["repeated_paths_per_patch"] == 2
    assert result["generation"]["test"]["snapshot_moments"]["source_snapshots"] == 1
    rms = result["generation"]["test"]["snapshot_moments"][
        "per_snapshot"
    ]["snapshotA"]["fields"]["rho"]
    assert rms["baseline"]["generated_rms_over_dns"] == pytest.approx(0.5)
    assert rms["diverse"]["generated_rms_over_dns"] == pytest.approx(0.6)
    assert rms["diverse_minus_baseline_rms_over_dns"] == pytest.approx(0.1)
    assert a == unchanged


def test_rejects_wrong_optimization_budget_or_source_lineage():
    a, b = _fixture()
    b["continuation_provenance"]["target_optimizer_updates"] = 4800
    with pytest.raises(ValueError, match="inconsistent total update budget"):
        compare(a, b)
    a, b = _fixture()
    b["continuation_provenance"]["target_epochs"] = 200
    b["continuation_provenance"]["target_optimizer_updates"] = 6000
    b["completed_training_epochs"] = 200
    with pytest.raises(ValueError, match="not matched optimizer budget"):
        compare(a, b)
    a, b = _fixture()
    b["continuation_provenance"]["source_frozen_baseline_run"] = "wrong"
    with pytest.raises(ValueError, match="source lineage"):
        compare(a, b)


def test_rejects_incomplete_origin_enumeration_and_changed_dns_statistics():
    a, b = _fixture()
    b["generation_ensemble_by_snapshot"]["validation"]["snapshotA"]["patches"] = 1
    with pytest.raises(ValueError, match="both fixed origins"):
        compare(a, b)
    a, b = _fixture()
    b["all_generation_origins"] = False
    with pytest.raises(ValueError, match="both fixed generation origins"):
        compare(a, b)
    a, b = _fixture()
    b["generation_ensemble_by_snapshot"]["test"]["snapshotA"][
        "fields"
    ]["rho"]["dns_reference_second_moment"] = 4.5
    with pytest.raises(ValueError, match="mismatched common-reference"):
        compare(a, b)
    a, b = _fixture()
    b["generation_ensemble_by_snapshot"]["test"]["snapshotA"][
        "fields"
    ]["rho"]["stochastic_variance"] = 2.0
    with pytest.raises(ValueError, match="mismatched common-reference"):
        compare(a, b)
