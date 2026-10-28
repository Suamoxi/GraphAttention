"""Compare SR-0.1 and SR-0.3 checkpoint diagnostics on *common* free-node support.

Read-only, JSON-only analysis. The source runs are untouched. This deliberately
fails closed if either report lacks the post-M44 SR-0.3b common-support metrics,
or if different snapshots, FM paths, geometry, or sample ensembles were evaluated.

Point-sampled DNS proxy only; this is not LES coarse-filter consistency or SGS
validation. Flow Matching velocity MSE does not measure generative PDF fidelity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")
SPLITS = ("train", "validation", "test")
GENERATED_SPLITS = ("validation", "test")


def _close(a: float, b: float, label: str) -> None:
    if not (math.isfinite(float(a)) and math.isfinite(float(b))):
        raise ValueError(f"nonfinite statistic: {label}")
    if not math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=2e-5):
        raise ValueError(f"mismatched common-reference statistic ({label}): {a} != {b}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _checked_report(data: dict[str, Any], constrained: bool) -> None:
    _require(data.get("status") == "sr0_checkpoint_diagnostics_complete", "invalid diagnostic status")
    _require(data.get("best_validation_loss_reproduced") is True, "checkpoint validation parity failed")
    _require(
        data.get("coarse_observation") == "point_sampled_DNS_not_filtered_LES",
        "only point-sampled DNS proxy diagnostics are comparable",
    )
    _require(bool(data.get("point_sample_constraint", False)) == constrained,
             "constrained/unconstrained report roles are reversed")
    _require(isinstance(data.get("repeated_flow_matching"), dict),
             "missing repeated Flow Matching diagnostics")


def _common(report: dict[str, Any], split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    repeated = report["repeated_flow_matching"][split]
    _require(int(repeated["paths_per_patch"]) >= 2, "need at least two repeat paths")
    _require("common_unobserved_node_support" in repeated,
             f"{split}: missing common_unobserved_node_support; rerun diagnostic at revision 71a9653 or later")
    common = repeated["common_unobserved_node_support"]
    for key in ("matched_flow_mse", "zero_velocity_flow_mse", "wrong_coarse_flow_mse"):
        _require(math.isfinite(float(common[key])), f"{split}: missing or nonfinite {key}")
    _require(tuple(common["matched_by_field"]) == FIELDS,
             f"{split}: unexpected conservative-field order")
    return repeated, common


def _assemble_split(old: dict[str, Any], new: dict[str, Any], split: str) -> dict[str, Any]:
    a, oa = _common(old, split)
    b, ob = _common(new, split)
    for key in ("paths_per_patch", "num_patches", "num_snapshots"):
        _require(a[key] == b[key], f"{split}: different {key}")
    for key in ("fine_nodes_per_patch", "coarse_nodes_per_patch", "unobserved_nodes_per_patch"):
        _require(oa[key] == ob[key], f"{split}: different common support {key}")
    _require(oa["unobserved_nodes_per_patch"] + oa["coarse_nodes_per_patch"]
             == oa["fine_nodes_per_patch"], f"{split}: support sizes are inconsistent")
    _close(oa["zero_velocity_flow_mse"], ob["zero_velocity_flow_mse"], f"{split}/zero/reference")
    old_snapshots, new_snapshots = oa["per_snapshot"], ob["per_snapshot"]
    _require(set(old_snapshots) == set(new_snapshots), f"{split}: different source snapshot IDs")
    for sid in old_snapshots:
        _close(old_snapshots[sid]["zero_velocity_flow_mse"],
               new_snapshots[sid]["zero_velocity_flow_mse"], f"{split}/{sid}/zero/reference")
    changes = {
        name: float(ob["matched_by_field"][name]) - float(oa["matched_by_field"][name])
        for name in FIELDS
    }
    per_snapshot = {
        sid: {
            "unconstrained_common_mse": float(old_snapshots[sid]["matched_flow_mse"]),
            "constrained_common_mse": float(new_snapshots[sid]["matched_flow_mse"]),
            "constrained_minus_unconstrained_mse": (
                float(new_snapshots[sid]["matched_flow_mse"])
                - float(old_snapshots[sid]["matched_flow_mse"])
            ),
            "unconstrained_condition_gap": float(old_snapshots[sid]["wrong_minus_matched_mse"]),
            "constrained_condition_gap": float(new_snapshots[sid]["wrong_minus_matched_mse"]),
        }
        for sid in sorted(old_snapshots)
    }
    delta = float(ob["matched_flow_mse"]) - float(oa["matched_flow_mse"])
    return {
        "patches": a["num_patches"],
        "source_snapshots": a["num_snapshots"],
        "deterministic_paths_per_patch": a["paths_per_patch"],
        "common_unobserved_nodes_per_patch": oa["unobserved_nodes_per_patch"],
        "zero_velocity_reference_mse": float(oa["zero_velocity_flow_mse"]),
        "unconstrained_common_mse": float(oa["matched_flow_mse"]),
        "constrained_common_mse": float(ob["matched_flow_mse"]),
        "constrained_minus_unconstrained_mse": delta,
        "relative_mse_change_percent": 100 * delta / float(oa["matched_flow_mse"]),
        "constrained_better_snapshots": sum(
            record["constrained_minus_unconstrained_mse"] < 0 for record in per_snapshot.values()
        ),
        "unconstrained_condition_gap": float(oa["wrong_minus_matched_mse"]),
        "constrained_condition_gap": float(ob["wrong_minus_matched_mse"]),
        "constrained_minus_unconstrained_mse_by_field": changes,
        "per_snapshot": per_snapshot,
    }


def _ensemble_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    def _summary(group: str) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for field in FIELDS:
            values = [float(e[group][field]) for e in entries]
            target = [float(e["reference_residual_rms_by_field"][field]) for e in entries]
            _require(all(math.isfinite(v) and v > 0 for v in target),
                     f"zero/nonfinite DNS reference RMS for {field}")
            _require(all(math.isfinite(v) and v >= 0 for v in values),
                     f"nonfinite generated RMS for {field}")
            result[field] = {
                "mean_patch_ratio": sum(v/t for v, t in zip(values, target)) / len(entries),
                "pooled_rms_ratio": math.sqrt(sum(v*v for v in values) / sum(t*t for t in target)),
            }
        return result

    return {
        "draws": entries[0]["draws"],
        "patches": len(entries),
        "raw_generated_to_reference_rms": _summary("raw_residual_rms_by_field"),
        "projected_generated_to_reference_rms": _summary("projected_residual_rms_by_field"),
        "ensemble_spread_to_reference_rms": _summary("sample_to_sample_std_rms_by_field"),
        "maximum_raw_coarse_observation_abs_error": max(
            float(v) for e in entries for v in e["raw_point_max_abs_by_field"].values()
        ),
        "maximum_projected_coarse_observation_abs_error": max(
            float(v) for e in entries for v in e["projected_point_max_abs_by_field"].values()
        ),
        "maximum_nonpositive_density_fraction": max(
            float(e["nonpositive_density_fraction_raw"]) for e in entries
        ),
    }


def _assemble_generation(old: dict[str, Any], new: dict[str, Any], split: str) -> dict[str, Any]:
    a = {e["sample_id"]: e for e in old["generation_ensemble"][split]}
    b = {e["sample_id"]: e for e in new["generation_ensemble"][split]}
    _require(a and set(a) == set(b), f"{split}: generation ensemble sample IDs differ or are empty")
    _require(old["draws_per_generation_patch"] == new["draws_per_generation_patch"],
             "generation ensemble draw counts differ")
    _require(old["generation_heun_steps"] == new["generation_heun_steps"],
             "generation Heun steps differ")
    for sid in a:
        _require(a[sid]["draws"] == b[sid]["draws"], f"{sid}: generation draws differ")
        _require(a[sid]["coarse_node_count"] == b[sid]["coarse_node_count"],
                 f"{sid}: coarse node count differs")
        for field in FIELDS:
            _close(a[sid]["reference_residual_rms_by_field"][field],
                   b[sid]["reference_residual_rms_by_field"][field], f"{sid}/{field}/DNS RMS")
    old_stats = _ensemble_stats([a[sid] for sid in sorted(a)])
    new_stats = _ensemble_stats([b[sid] for sid in sorted(b)])
    return {
        "sample_ids": sorted(a),
        "unconstrained": old_stats,
        "constrained": new_stats,
        "constrained_minus_unconstrained_mean_patch_raw_rms_ratio": {
            field: (
                new_stats["raw_generated_to_reference_rms"][field]["mean_patch_ratio"]
                - old_stats["raw_generated_to_reference_rms"][field]["mean_patch_ratio"]
            ) for field in FIELDS
        },
        "caveat": (
            "One DNS target per coarse condition; an RMS ratio near 1 is not "
            "conditional-PDF fidelity or accurate turbulence spectra."
        ),
    }


def compare(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Compare matching checkpoint diagnostic reports without touching checkpoints."""
    _checked_report(old, constrained=False)
    _checked_report(new, constrained=True)
    _require(old["source_run_dir"] != new["source_run_dir"], "reports reference the same run")
    for key in ("model_configuration", "snapshot_splits", "coarse_observation"):
        _require(old[key] == new[key], f"reports differ in {key}")
    split_metrics = {split: _assemble_split(old, new, split) for split in SPLITS}
    ensemble = {split: _assemble_generation(old, new, split) for split in GENERATED_SPLITS}
    return {
        "status": "m44_sr03_common_support_comparison_complete",
        "unconstrained_source": old["source_run_dir"],
        "constrained_source": new["source_run_dir"],
        "common_observation": "point-sampled DNS proxy, not physically filtered LES",
        "flow_matching": split_metrics,
        "generation": ensemble,
        "interpretation": [
            "Both checkpoints retain their native FM paths and distinct observed-node inputs.",
            "Matched/zero MSE uses the same 604 free nodes and fixed random paths.",
            "The independent DNS source snapshots, not noise draws, determine evidence breadth.",
            "Do not select an architecture using test data; use test for final reporting only.",
            "Neither FM MSE nor residual RMS alone establishes conditional PDFs or SGS accuracy.",
        ],
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = ["# M44 SR-0.3b common-support checkpoint comparison", "",
             "**Point-sampled DNS proxy only; not conservative LES filtering.**", "",
             "| Split | Unconstrained MSE | Constrained MSE | Change (%) | Constrained better snapshots |",
             "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        a = result["flow_matching"][split]
        lines.append(
            f"| {split} | {a['unconstrained_common_mse']:.5f} | "
            f"{a['constrained_common_mse']:.5f} | {a['relative_mse_change_percent']:+.1f}% | "
            f"{a['constrained_better_snapshots']}/{a['source_snapshots']} |"
        )
    for split in GENERATED_SPLITS:
        a = result["generation"][split]
        lines.extend(["", f"## {split.capitalize()} generation ensemble", "",
                      "| Field | Unconstrained generated / DNS RMS | Constrained generated / DNS RMS |",
                      "|---|---:|---:|"])
        for f in FIELDS:
            x = a["unconstrained"]["raw_generated_to_reference_rms"][f]["mean_patch_ratio"]
            y = a["constrained"]["raw_generated_to_reference_rms"][f]["mean_patch_ratio"]
            lines.append(f"| {f} | {x:.3f} | {y:.3f} |")
    lines.extend(["", "The FM metric uses a common scoring support but native model inputs differ at observed nodes.",
                  "A residual RMS ratio near one is not proof of matching spatial structure or conditional distribution.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unconstrained", type=Path, required=True)
    parser.add_argument("--constrained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.unconstrained.open(encoding="utf-8") as handle:
        old = json.load(handle)
    with args.constrained.open(encoding="utf-8") as handle:
        new = json.load(handle)
    result = compare(old, new)
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
