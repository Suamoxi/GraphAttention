"""M44 SR-0.5: paired continued-training comparison with snapshot-level moments.

Both 150-epoch checkpoints are selected on validation. Every free-node
FM reference path, generation sample ID, coarse observation and DNS
moment must match. No independent LES filtering/SGS statistics are tested.
Execute only through Singularity Slurm jobs on the grace partition.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.compare_m44_sr03_diagnostics import (
    FIELDS, GENERATED_SPLITS, SPLITS,
    _assemble_generation, _assemble_split, _checked_report, _close, _require,
)
from scripts.compare_m44_sr04_diagnostics import _compare_spatial_ensemble

_PROTOCOL = "M44_SR05_IMMUTABLE_CONTINUATION_V1"


def _continuation(report: dict[str, Any], role: str) -> dict[str, Any]:
    _checked_report(report, constrained=True)
    c = report.get("continuation_provenance")
    _require(isinstance(c, dict) and c.get("protocol") == _PROTOCOL,
             f"{role}: missing SR-0.5 immutable continuation provenance")
    _require(c.get("mode") == role, f"{role}: incorrect continuation mode")
    _require(report.get("completed_training_epochs") == c.get("target_epochs"),
             f"{role}: training did not complete the target budget")
    _require(c["initial_epochs"] >= 1 and c["target_epochs"] > c["initial_epochs"],
             f"{role}: incorrect initial/final epoch provenance")
    _require(c["optimizer_updates_per_epoch"] > 0,
             f"{role}: invalid optimizer update budget")
    _require(c["initial_optimizer_updates"] == (
        c["initial_epochs"] * c["optimizer_updates_per_epoch"]
    ), f"{role}: inconsistent initial update budget")
    _require(c["target_optimizer_updates"] == (
        c["target_epochs"] * c["optimizer_updates_per_epoch"]
    ), f"{role}: inconsistent total update budget")
    _require(report["source_manifest_sha256"] == c["source_manifest_sha256"],
             f"{role}: continued manifest not identical to source")
    _require(report.get("all_generation_origins") is True,
             f"{role}: require both fixed generation origins")
    _require(report.get("best_epoch", -1) < c["target_epochs"],
             f"{role}: best checkpoint epoch exceeds target budget")
    _require(report.get("generation_heun_steps") == 50,
             f"{role}: expected 50-step Heun generations")
    _require(report.get("draws_per_generation_patch") == 16,
             f"{role}: expected 16 generation draws")
    for split in GENERATED_SPLITS:
        by_snapshot = report.get("generation_ensemble_by_snapshot", {}).get(split)
        _require(isinstance(by_snapshot, dict) and by_snapshot,
                 f"{role}/{split}: missing per-snapshot ensemble moments")
        _require(len(by_snapshot) == report["snapshot_splits"][split],
                 f"{role}/{split}: not all held-out source snapshots evaluated")
        observed_ids: dict[str, set[str]] = {}
        for sample in report["generation_ensemble"][split]:
            sample_id = sample["sample_id"]
            _require(sample_id.count("/sr0_patch_") == 1,
                     f"{role}/{split}: invalid patch sample ID")
            source = sample_id.split("/sr0_patch_", 1)[0]
            observed_ids.setdefault(source, set()).add(sample_id)
        _require(set(observed_ids) == set(by_snapshot),
                 f"{role}/{split}: snapshot generation summaries differ from raw sample IDs")
        for sid, entry in by_snapshot.items():
            _require(entry.get("patches") == 2, f"{role}/{sid}: expected both fixed origins")
            _require(entry.get("draws_per_patch") == 16,
                     f"{role}/{sid}: expected 16 draws per origin")
            _require(len(entry.get("sample_ids", [])) == 2,
                     f"{role}/{sid}: missing original patch identities")
            _require(set(entry["sample_ids"]) == observed_ids[sid],
                     f"{role}/{sid}: summary sample IDs do not match generated patches")
    return c


def _snapshot_moments(
    baseline: dict[str, Any],
    diverse: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    a = baseline["generation_ensemble_by_snapshot"][split]
    b = diverse["generation_ensemble_by_snapshot"][split]
    _require(a and set(a) == set(b), f"{split}: source snapshot sets differ")
    output: dict[str, Any] = {"source_snapshots": len(a), "per_snapshot": {}}
    for sid in sorted(a):
        _require(a[sid]["sample_ids"] == b[sid]["sample_ids"],
                 f"{split}/{sid}: patch origins differ")
        fields = {}
        for field in FIELDS:
            u = a[sid]["fields"][field]
            v = b[sid]["fields"][field]
            _close(u["dns_reference_second_moment"], v["dns_reference_second_moment"],
                   f"{split}/{sid}/{field}/DNS second moment")
            for label, item in (("baseline", u), ("diverse", v)):
                mean = float(item["ensemble_mean_square"])
                variance = float(item["stochastic_variance"])
                generated = float(item["generated_second_moment"])
                reference = float(item["dns_reference_second_moment"])
                _require(all(math.isfinite(x) and x >= 0
                             for x in (mean, variance, generated, reference))
                         and reference > 0,
                         f"{split}/{sid}/{field}/{label}: nonfinite/negative energy")
                _close(mean + variance, generated,
                       f"{split}/{sid}/{field}/{label}/variance identity")
            fields[field] = {
                "dns_reference_second_moment": u["dns_reference_second_moment"],
                "baseline": {
                    key: u[key] for key in (
                        "ensemble_mean_square", "stochastic_variance",
                        "generated_second_moment", "generated_rms_over_dns",
                        "ensemble_mean_rms_over_dns", "ensemble_std_rms_over_dns",
                        "mean_fraction_of_generated_second_moment",
                        "variance_fraction_of_generated_second_moment",
                    )
                },
                "diverse": {
                    key: v[key] for key in (
                        "ensemble_mean_square", "stochastic_variance",
                        "generated_second_moment", "generated_rms_over_dns",
                        "ensemble_mean_rms_over_dns", "ensemble_std_rms_over_dns",
                        "mean_fraction_of_generated_second_moment",
                        "variance_fraction_of_generated_second_moment",
                    )
                },
                "diverse_minus_baseline_rms_over_dns": (
                    float(v["generated_rms_over_dns"])
                    - float(u["generated_rms_over_dns"])
                ),
            }
        output["per_snapshot"][sid] = {
            "patches": 2,
            "sample_ids": a[sid]["sample_ids"],
            "fields": fields,
        }
    output["equal_weight_snapshot_means"] = {
        field: {
            role: {
                key: sum(
                    float(output["per_snapshot"][sid]["fields"][field][role][key])
                    for sid in a
                ) / len(a)
                for key in (
                    "generated_rms_over_dns",
                    "ensemble_mean_rms_over_dns",
                    "ensemble_std_rms_over_dns",
                )
            }
            for role in ("baseline", "diverse")
        }
        for field in FIELDS
    }
    return output


def compare(baseline: dict[str, Any], diverse: dict[str, Any]) -> dict[str, Any]:
    a = _continuation(baseline, "baseline")
    b = _continuation(diverse, "diverse")
    _require(baseline["source_run_dir"] != diverse["source_run_dir"],
             "cannot compare a training checkpoint with itself")
    _require(a["source_run_dir"] != b["source_run_dir"],
             "continuation parents must be distinct")
    for key in ("initial_epochs", "target_epochs",
                "optimizer_updates_per_epoch", "target_optimizer_updates"):
        _require(a[key] == b[key], f"not matched optimizer budget: {key}")
    _require(baseline.get("train_patch_diversity") is None,
             "baseline mistakenly uses spatial patch diversity")
    schedule = diverse.get("train_patch_diversity")
    _require(isinstance(schedule, dict) and schedule.get("pool_size") == 16
             and schedule.get("patches_per_snapshot_per_epoch") == 2,
             "diverse continuation must use the original 16-origin schedule")
    frozen = diverse.get("frozen_scale_provenance")
    _require(isinstance(frozen, dict) and frozen.get("source_run_dir") == a["source_run_dir"],
             "diversified training not normalized from original SR-0.3 baseline")
    _require(frozen["source_manifest_sha256"] == a["source_manifest_sha256"],
             "SR-0.4 frozen scaling references different original baseline manifest")
    _require(b.get("source_frozen_baseline_run") == a["source_run_dir"],
             "diverse continuation source lineage does not match baseline")
    _require(diverse.get("train_evaluation_patch_scope") ==
             "original_fixed_train_origins_only",
             "SR-0.4 trained-patch diagnostics must use original fixed origins")
    for key in ("model_configuration", "snapshot_splits", "coarse_observation"):
        _require(baseline[key] == diverse[key], f"different {key}")
    flow = {}
    for split in SPLITS:
        metrics = _assemble_split(baseline, diverse, split)
        if split in GENERATED_SPLITS:
            _require(metrics["patches"] == 2 * metrics["source_snapshots"],
                     f"{split}: Flow Matching must use both fixed origins per snapshot")
        flow[split] = {
            "snapshots": metrics["source_snapshots"],
            "patches": metrics["patches"],
            "free_nodes_per_patch": metrics["common_unobserved_nodes_per_patch"],
            "repeated_paths_per_patch": metrics["deterministic_paths_per_patch"],
            "zero_velocity_reference_mse": metrics["zero_velocity_reference_mse"],
            "baseline_mse": metrics["unconstrained_common_mse"],
            "diverse_mse": metrics["constrained_common_mse"],
            "diverse_minus_baseline_mse": metrics["constrained_minus_unconstrained_mse"],
            "relative_mse_change_percent": metrics["relative_mse_change_percent"],
            "diverse_better_source_snapshots": metrics["constrained_better_snapshots"],
            "baseline_condition_gap": metrics["unconstrained_condition_gap"],
            "diverse_condition_gap": metrics["constrained_condition_gap"],
            "mse_change_by_field": metrics["constrained_minus_unconstrained_mse_by_field"],
            "per_snapshot": metrics["per_snapshot"],
        }
    generation = {}
    for split in GENERATED_SPLITS:
        ensemble = _assemble_generation(baseline, diverse, split)
        generation[split] = {
            "sample_ids": ensemble["sample_ids"],
            "raw_rms": {
                "baseline": ensemble["unconstrained"]["raw_generated_to_reference_rms"],
                "diverse": ensemble["constrained"]["raw_generated_to_reference_rms"],
            },
            "spatial_increments": _compare_spatial_ensemble(baseline, diverse, split),
            "snapshot_moments": _snapshot_moments(baseline, diverse, split),
            "coarse_observation_max_error": {
                "baseline": ensemble["unconstrained"]["maximum_raw_coarse_observation_abs_error"],
                "diverse": ensemble["constrained"]["maximum_raw_coarse_observation_abs_error"],
            },
        }
        _require(all(value == 0.0 for value in generation[split][
            "coarse_observation_max_error"
        ].values()), "intrinsic point-observation constraint violated")
    return {
        "status": "m44_sr05_matched_budget_pair_complete",
        "baseline_run": baseline["source_run_dir"],
        "diverse_run": diverse["source_run_dir"],
        "baseline_initial_source": a["source_run_dir"],
        "diverse_initial_source": b["source_run_dir"],
        "initial_epochs": a["initial_epochs"],
        "total_epochs": a["target_epochs"],
        "optimizer_updates_per_epoch": a["optimizer_updates_per_epoch"],
        "total_optimizer_updates": a["target_optimizer_updates"],
        "baseline_best_epoch": baseline["best_epoch"],
        "diverse_best_epoch": diverse["best_epoch"],
        "data_policy": {
            "baseline": a["training_data_policy"],
            "diverse": b["training_data_policy"],
        },
        "flow_matching": flow,
        "generation": generation,
        "interpretation": [
            "Same total 150-epoch optimizer budget and original 50-epoch full checkpoint lineage.",
            "Validation/test use the exact two fixed patch origins per source snapshot.",
            "Per-snapshot mean-square decomposition does not measure a conditional PDF.",
            "Spatial patches and Gaussian draws are not independent DNS snapshots.",
            "Test diagnostics are for final reporting, never checkpoint/architecture selection.",
            "Point-sampled DNS proxy, not a volume-filtered LES consistency operator.",
        ],
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M44 SR-0.5 — matched optimization budget",
        "",
        f"Total epochs: {result['total_epochs']}; total updates/model: "
        f"{result['total_optimizer_updates']}; both full-state continuations.",
        "",
        "| Split | Baseline FM MSE | Diverse FM MSE | Delta (%) | Diverse better snapshots |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        row = result["flow_matching"][split]
        lines.append(
            f"| {split} | {row['baseline_mse']:.5f} | {row['diverse_mse']:.5f} | "
            f"{row['relative_mse_change_percent']:+.2f}% | "
            f"{row['diverse_better_source_snapshots']}/{row['snapshots']} |"
        )
    for split in GENERATED_SPLITS:
        lines += [
            "", f"## {split} — two origins per source snapshot",
            "", "| Field | Baseline RMS/DNS | Diverse RMS/DNS | "
            "Baseline std/DNS | Diverse std/DNS |",
            "|---|---:|---:|---:|---:|",
        ]
        for field in FIELDS:
            row = result["generation"][split]["snapshot_moments"][
                "equal_weight_snapshot_means"
            ][field]
            lines.append(
                f"| {field} | {row['baseline']['generated_rms_over_dns']:.3f} | "
                f"{row['diverse']['generated_rms_over_dns']:.3f} | "
                f"{row['baseline']['ensemble_std_rms_over_dns']:.3f} | "
                f"{row['diverse']['ensemble_std_rms_over_dns']:.3f} |"
            )
    lines.extend([
        "",
        "Residual second moments are decomposed exactly into squared ensemble mean + "
        "sample-to-sample variance. Ratios average *snapshots*, never treat "
        "two patch origins or 16 noise draws as independent flow realizations.",
        "Point-sampled DNS only; these are not conservative LES or SGS conclusions.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--diverse", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.baseline.open(encoding="utf-8") as stream:
        baseline = json.load(stream)
    with args.diverse.open(encoding="utf-8") as stream:
        diverse = json.load(stream)
    result = compare(baseline, diverse)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "comparison_summary.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    print(_markdown(result), flush=True)


if __name__ == "__main__":
    main()
