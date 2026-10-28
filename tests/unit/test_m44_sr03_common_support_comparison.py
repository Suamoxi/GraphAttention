"""Synthetic gates: refuse misleading comparisons of incompatible diagnostics."""

import copy

import pytest

from scripts.compare_m44_sr03_diagnostics import compare


FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def _report(constrained: bool) -> dict:
    def source_mse():
        return {
            "matched_flow_mse": 1.0 if not constrained else 0.8,
            "wrong_minus_matched_mse": 0.2,
            "zero_velocity_flow_mse": 2.0,
        }
    def part():
        m = source_mse()
        return {
            "matched_flow_mse": m["matched_flow_mse"],
            "zero_velocity_flow_mse": m["zero_velocity_flow_mse"],
            "wrong_coarse_flow_mse": 1.2 if not constrained else 1.0,
            "wrong_minus_matched_mse": m["wrong_minus_matched_mse"],
            "matched_by_field": {f: m["matched_flow_mse"] for f in FIELDS},
            "fine_nodes_per_patch": 729,
            "coarse_nodes_per_patch": 125,
            "unobserved_nodes_per_patch": 604,
            "per_snapshot": {"snapshotA": dict(m)},
        }
    repeats = {
        split: {
            "paths_per_patch": 2,
            "num_patches": 1,
            "num_snapshots": 1,
            "common_unobserved_node_support": part(),
        } for split in ("train", "validation", "test")
    }
    ensemble = {
        "sample_id": "snapshotA/sr0_patch_0_0_0",
        "draws": 2,
        "coarse_node_count": 125,
        "reference_residual_rms_by_field": {f: 2.0 for f in FIELDS},
        "raw_residual_rms_by_field": {f: 1.0 if not constrained else 1.5 for f in FIELDS},
        "projected_residual_rms_by_field": {f: 0.75 if not constrained else 1.5 for f in FIELDS},
        "sample_to_sample_std_rms_by_field": {f: 0.5 for f in FIELDS},
        "raw_point_max_abs_by_field": {f: 0.2 if not constrained else 0.0 for f in FIELDS},
        "projected_point_max_abs_by_field": {f: 0.0 for f in FIELDS},
        "nonpositive_density_fraction_raw": 0.0,
    }
    return {
        "status": "sr0_checkpoint_diagnostics_complete",
        "best_validation_loss_reproduced": True,
        "source_run_dir": f"run_{'constrained' if constrained else 'unconstrained'}",
        "coarse_observation": "point_sampled_DNS_not_filtered_LES",
        "point_sample_constraint": constrained,
        "model_configuration": {"hidden_dim": 64},
        "snapshot_splits": {"train": 1, "validation": 1, "test": 1},
        "repeated_flow_matching": repeats,
        "generation_ensemble": {"validation": [copy.deepcopy(ensemble)], "test": [copy.deepcopy(ensemble)]},
        "draws_per_generation_patch": 2,
        "generation_heun_steps": 50,
    }


def test_common_support_comparator_is_paired_and_read_only():
    old = _report(False)
    new = _report(True)
    original = copy.deepcopy(old)
    result = compare(old, new)
    assert result["status"] == "m44_sr03_common_support_comparison_complete"
    assert result["flow_matching"]["test"]["constrained_minus_unconstrained_mse"] == pytest.approx(-0.2)
    assert result["flow_matching"]["test"]["constrained_better_snapshots"] == 1
    assert result["generation"]["test"]["unconstrained"]["raw_generated_to_reference_rms"]["rhou"]["mean_patch_ratio"] == 0.5
    assert result["generation"]["test"]["constrained"]["raw_generated_to_reference_rms"]["rhou"]["mean_patch_ratio"] == 0.75
    assert old == original


def test_rejects_missing_common_support_in_old_native_report():
    old, new = _report(False), _report(True)
    del old["repeated_flow_matching"]["validation"]["common_unobserved_node_support"]
    with pytest.raises(ValueError, match="missing common_unobserved_node_support"):
        compare(old, new)


def test_rejects_different_reference_noise_or_snapshot():
    old, new = _report(False), _report(True)
    new["repeated_flow_matching"]["test"]["common_unobserved_node_support"]["zero_velocity_flow_mse"] = 2.3
    with pytest.raises(ValueError, match="mismatched common-reference"):
        compare(old, new)
    new = _report(True)
    new["repeated_flow_matching"]["validation"]["common_unobserved_node_support"]["per_snapshot"] = {"other": {}}
    with pytest.raises(ValueError, match="different source snapshot IDs"):
        compare(old, new)


def test_rejects_incompatible_generation():
    old, new = _report(False), _report(True)
    new["draws_per_generation_patch"] = 16
    with pytest.raises(ValueError, match="draw counts differ"):
        compare(old, new)
    new = _report(True)
    new["generation_ensemble"]["test"][0]["sample_id"] = "other"
    with pytest.raises(ValueError, match="sample IDs differ"):
        compare(old, new)
    new = _report(True)
    new["generation_ensemble"]["test"][0]["reference_residual_rms_by_field"]["rho"] = 3.0
    with pytest.raises(ValueError, match="mismatched common-reference"):
        compare(old, new)


def test_rejects_incompatible_architecture_or_point_proxy():
    old, new = _report(False), _report(True)
    new["model_configuration"]["hidden_dim"] = 128
    with pytest.raises(ValueError, match="differ in model_configuration"):
        compare(old, new)
    new = _report(True)
    new["coarse_observation"] = "filtered_LES"
    with pytest.raises(ValueError, match="point-sampled DNS"):
        compare(old, new)
