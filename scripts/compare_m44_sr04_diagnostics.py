"""M44 SR-0.4: strict paired comparison against the constrained SR-0.3 baseline.

Both models preserve point-sampled coarse observations intrinsically.
No Python from this script should run on login nodes: use the grace Slurm
Singularity launcher (m44_sr04_compare.slurm).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.compare_m44_sr03_diagnostics import (
    FIELDS, SPLITS, GENERATED_SPLITS,
    _assemble_generation, _assemble_split, _checked_report, _close, _require,
)


def _compare_spatial_ensemble(
    baseline: dict[str, Any],
    diverse: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    original = {entry["sample_id"]: entry for entry in baseline["generation_ensemble"][split]}
    augmented = {entry["sample_id"]: entry for entry in diverse["generation_ensemble"][split]}
    _require(original and set(original) == set(augmented),
             f"{split}: spatial diagnostic sample IDs differ")
    paired = []
    for sample_id in sorted(original):
        a = original[sample_id].get("residual_spatial_moments")
        b = augmented[sample_id].get("residual_spatial_moments")
        _require(isinstance(a, dict) and isinstance(b, dict),
                 f"{sample_id}: missing new residual_spatial_moments; rerun diagnostics")
        ai = a["nearest_neighbor_increments"]
        bi = b["nearest_neighbor_increments"]
        _require(ai["graph_edge_count"] == bi["graph_edge_count"],
                 f"{sample_id}: different spatial graph edge count")
        _close(ai["separation_over_L_ref"], bi["separation_over_L_ref"],
               f"{sample_id}: spatial increment separation")
        for field in FIELDS:
            _close(ai["reference_rms_by_field"][field],
                   bi["reference_rms_by_field"][field],
                   f"{sample_id}/{field}/reference increment")
            _close(a["reference_residual_spatial_mean_by_field"][field],
                   b["reference_residual_spatial_mean_by_field"][field],
                   f"{sample_id}/{field}/reference residual mean")
            _close(a["reference_residual_spatial_std_by_field"][field],
                   b["reference_residual_spatial_std_by_field"][field],
                   f"{sample_id}/{field}/reference residual std")
        paired.append((a, b))

    result: dict[str, Any] = {
        "patches": len(paired),
        "samples": sorted(original),
        "per_field": {},
    }
    for field in FIELDS:
        ref_inc = [a["nearest_neighbor_increments"]["reference_rms_by_field"][field]
                   for a, _ in paired]
        _require(all(float(x) > 0 for x in ref_inc),
                 f"{split}/{field}: zero reference increment RMS")
        original_inc = [
            a["nearest_neighbor_increments"]["generated_rms_by_field"][field]
            for a, _ in paired
        ]
        new_inc = [
            b["nearest_neighbor_increments"]["generated_rms_by_field"][field]
            for _, b in paired
        ]
        n = len(paired)
        result["per_field"][field] = {
            "baseline_increment_over_DNS": sum(
                float(x) / float(ref) for x, ref in zip(original_inc, ref_inc)
            ) / n,
            "diverse_increment_over_DNS": sum(
                float(x) / float(ref) for x, ref in zip(new_inc, ref_inc)
            ) / n,
            "baseline_ensemble_pointwise_std_rms": sum(
                float(a["ensemble_pointwise_std_rms_by_field"][field])
                for a, _ in paired
            ) / n,
            "diverse_ensemble_pointwise_std_rms": sum(
                float(b["ensemble_pointwise_std_rms_by_field"][field])
                for _, b in paired
            ) / n,
            "baseline_ensemble_mean_residual_rms": sum(
                float(a["ensemble_mean_residual_rms_by_field"][field])
                for a, _ in paired
            ) / n,
            "diverse_ensemble_mean_residual_rms": sum(
                float(b["ensemble_mean_residual_rms_by_field"][field])
                for _, b in paired
            ) / n,
        }
    return result


def compare(baseline: dict[str, Any], diverse: dict[str, Any]) -> dict[str, Any]:
    _checked_report(baseline, constrained=True)
    _checked_report(diverse, constrained=True)
    _require(baseline["source_run_dir"] != diverse["source_run_dir"],
             "baseline and spatially diversified runs must be distinct")
    _require(baseline.get("train_patch_diversity") is None,
             "SR-0.3 baseline unexpectedly uses patch diversity")
    schedule = diverse.get("train_patch_diversity")
    _require(isinstance(schedule, dict), "SR-0.4 report must contain patch schedule")
    provenance = diverse.get("frozen_scale_provenance")
    _require(isinstance(provenance, dict), "SR-0.4 frozen scale provenance missing")
    _require(provenance["source_run_dir"] == baseline["source_run_dir"],
             "SR-0.4 frozen scales are not sourced from the supplied baseline")
    _require(provenance["source_manifest_sha256"] == baseline["source_manifest_sha256"],
             "source manifest hash differs from SR-0.3 baseline diagnostic")
    _require(diverse.get("train_evaluation_patch_scope") == "original_fixed_train_origins_only",
             "SR-0.4 training diagnostic does not use comparable fixed origins")
    for key in ("model_configuration", "snapshot_splits", "coarse_observation"):
        _require(baseline[key] == diverse[key], f"different {key}")
    fm = {}
    for split in SPLITS:
        data = _assemble_split(baseline, diverse, split)
        fm[split] = {
            "source_snapshots": data["source_snapshots"],
            "patches": data["patches"],
            "paths_per_patch": data["deterministic_paths_per_patch"],
            "free_nodes_per_patch": data["common_unobserved_nodes_per_patch"],
            "reference_zero_velocity_mse": data["zero_velocity_reference_mse"],
            "baseline_common_mse": data["unconstrained_common_mse"],
            "diverse_common_mse": data["constrained_common_mse"],
            "diverse_minus_baseline_mse": data["constrained_minus_unconstrained_mse"],
            "relative_mse_change_percent": data["relative_mse_change_percent"],
            "diverse_better_snapshots": data["constrained_better_snapshots"],
            "baseline_condition_gap": data["unconstrained_condition_gap"],
            "diverse_condition_gap": data["constrained_condition_gap"],
            "per_field_mse_change": data["constrained_minus_unconstrained_mse_by_field"],
            "per_snapshot": {
                sid: {
                    "baseline_common_mse": v["unconstrained_common_mse"],
                    "diverse_common_mse": v["constrained_common_mse"],
                    "diverse_minus_baseline_mse": v["constrained_minus_unconstrained_mse"],
                    "baseline_condition_gap": v["unconstrained_condition_gap"],
                    "diverse_condition_gap": v["constrained_condition_gap"],
                }
                for sid, v in data["per_snapshot"].items()
            },
        }
    ensembles = {}
    for split in GENERATED_SPLITS:
        traditional = _assemble_generation(baseline, diverse, split)
        ensembles[split] = {
            "sample_ids": traditional["sample_ids"],
            "baseline": traditional["unconstrained"],
            "diverse": traditional["constrained"],
            "diverse_minus_baseline_raw_rms_ratio": (
                traditional["constrained_minus_unconstrained_mean_patch_raw_rms_ratio"]
            ),
            "spatial_moments": _compare_spatial_ensemble(baseline, diverse, split),
        }
    return {
        "status": "m44_sr04_paired_spatial_diversity_comparison_complete",
        "baseline_source": baseline["source_run_dir"],
        "diverse_source": diverse["source_run_dir"],
        "frozen_scale_provenance": provenance,
        "train_patch_schedule": schedule,
        "flow_matching": fm,
        "generation": ensembles,
        "interpretation": [
            "Both models share the same constrained FM objective and frozen train-only scales.",
            "Both are scored on identical original snapshot-held-out patch identities.",
            "Spatial pool diversity changes source training examples, not optimizer updates per epoch.",
            "A lower FM velocity MSE does not prove better conditional PDF or physical LES super-resolution.",
            "Snapshot count is unchanged; 16 Gaussian paths are not 16 independent DNS realizations.",
            "Use validation for model decisions and test for final reporting only.",
        ],
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M44 SR-0.4 versus constrained SR-0.3",
        "",
        "**Point-sampled DNS proxy. Spatial diversity only; no LES filtering.**",
        "",
        "| Split | SR-0.3 FM MSE | SR-0.4 FM MSE | Change | Better snapshots |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        metrics = result["flow_matching"][split]
        lines.append(
            f"| {split} | {metrics['baseline_common_mse']:.5f} | "
            f"{metrics['diverse_common_mse']:.5f} | "
            f"{metrics['relative_mse_change_percent']:+.1f}% | "
            f"{metrics['diverse_better_snapshots']}/{metrics['source_snapshots']} |"
        )
    for split in GENERATED_SPLITS:
        lines.extend(["", f"## {split.capitalize()} generated residual RMS", "",
                      "| Channel | SR-0.3 RMS / DNS | SR-0.4 RMS / DNS | "
                      "SR-0.3 increment / DNS | SR-0.4 increment / DNS |",
                      "|---|---:|---:|---:|---:|"])
        for field in FIELDS:
            a = result["generation"][split]
            old = a["baseline"]["raw_generated_to_reference_rms"][field]["mean_patch_ratio"]
            new = a["diverse"]["raw_generated_to_reference_rms"][field]["mean_patch_ratio"]
            inc = a["spatial_moments"]["per_field"][field]
            lines.append(
                f"| {field} | {old:.3f} | {new:.3f} | "
                f"{inc['baseline_increment_over_DNS']:.3f} | "
                f"{inc['diverse_increment_over_DNS']:.3f} |"
            )
    lines += [
        "",
        "Increment ratios use a single nearest-neighbor graph-edge separation.",
        "No spectrum, conditional PDF, volume-conservative LES restriction or SGS validation is implied.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--diverse", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.baseline.open(encoding="utf-8") as file:
        baseline = json.load(file)
    with args.diverse.open(encoding="utf-8") as file:
        diverse = json.load(file)
    result = compare(baseline, diverse)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "comparison_summary.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    print(_markdown(result))


if __name__ == "__main__":
    main()
