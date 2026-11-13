"""Compare completed M47 physical-conditioning arms after matching scientific inputs.

This is a comparison of deterministic FM velocity MSE, not a generative
ensemble assessment. Test metrics are descriptive, never used for selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MODES = {
    "none": [],
    "mach": ["Ma_condition"],
    "mach_re": ["Ma_condition", "log_Re_condition"],
}
CASES = {
    f"{re}/Ma{mach}"
    for re in ("Re54", "Re94")
    for mach in ("0.05", "0.2", "0.3")
}


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing M47 comparison input: {path}")
    return json.loads(path.read_text())


def _normal_signature(signature: dict, mode: str) -> dict:
    content = json.loads(json.dumps(signature["content"]))
    physics = content.get("m47_physical_conditions")
    if not isinstance(physics, dict):
        raise ValueError(f"{mode}: expected qualified M47 physics in signature")
    if physics.get("condition_mode") != mode or physics.get("conditioning_parameters") != MODES[mode]:
        raise ValueError(f"{mode}: condition labels and signature disagree")
    if content.get("mode") != "train" or content.get("seed") != 42:
        raise ValueError(f"{mode}: comparison is restricted to matched seed-42 training")
    if content.get("physical_nondimensionalization") is not True:
        raise ValueError(f"{mode}: physical nondimensionalization is required")
    if content.get("temporal_embargo") != 2 or content.get("accumulation") != 6:
        raise ValueError(f"{mode}: unexpected split/optimizer settings")
    physics.pop("condition_mode")
    physics.pop("conditioning_parameters")
    return content


def _finite_mse(value: object, label: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: missing numerical MSE") from exc
    if not math.isfinite(scalar) or scalar < 0:
        raise ValueError(f"{label}: MSE must be finite and nonnegative")
    return scalar


def _check_split(split: dict) -> None:
    expected = {"train_ids": 240, "validation_ids": 30, "test_ids": 42, "excluded_ids": 48}
    sets = {name: split[name] for name in expected}
    if any(len(sets[name]) != size for name, size in expected.items()):
        raise ValueError("M47 comparison expects the reviewed 360-snapshot 41..100 temporal split")
    all_ids = [sample_id for part in sets.values() for sample_id in part]
    if len(set(all_ids)) != 360:
        raise ValueError("Overlapping or repeated M47 sample IDs")
    if {sample_id.split(":")[0] for sample_id in all_ids} != CASES:
        raise ValueError("M47 must contain exactly six physical cases")
    if any(int(sample_id.rsplit("_", 1)[1]) < 41 for sample_id in all_ids):
        raise ValueError("M47 comparison found transient-contaminated snapshots")


def compare(root: Path, basename: str = "train_s42") -> dict:
    baseline_signature = None
    baseline_split = None
    expected_epochs = None
    expected_lr = None
    rows = []
    for mode, conditions in MODES.items():
        run_dir = root / mode / basename
        summary = _read(run_dir / "summary.json")
        signature = _read(run_dir / "run_signature.json")
        split = _read(run_dir / "split.json")
        if not (run_dir / "best.pt").is_file():
            raise FileNotFoundError(f"{mode}: selected checkpoint missing")
        _check_split(split)
        normalized = _normal_signature(signature, mode)
        if baseline_signature is None:
            baseline_signature, baseline_split = normalized, split
        elif normalized != baseline_signature or split != baseline_split:
            raise ValueError(f"{mode}: scientific input signature or temporal split differs")
        if (summary.get("status") != "train_complete"
            or summary.get("mode") != "train"
            or summary.get("conditioning_mode") != mode
            or summary.get("conditioning_parameters") != conditions
            or summary.get("qualified_stationarity_manifest") is not True):
            raise ValueError(f"{mode}: incomplete or incorrectly labeled M47 training")
        if any(summary.get(name) != expected for name, expected in (
            ("num_physical_samples", 360), ("train_samples", 240),
            ("validation_samples", 30), ("test_samples", 42),
            ("excluded_samples", 48),
        )):
            raise ValueError(f"{mode}: unexpected physical sample counts")
        hyper = summary.get("m47_training_hyperparameters")
        if not isinstance(hyper, dict) or hyper.get("seed") != 42:
            raise ValueError(f"{mode}: missing or mismatched M47 hyperparameter record")
        lr = hyper.get("learning_rate")
        if not isinstance(lr, (float, int)) or lr <= 0 or not math.isfinite(float(lr)):
            raise ValueError(f"{mode}: missing or invalid learning rate")
        if expected_lr is None:
            expected_lr = lr
        elif lr != expected_lr:
            raise ValueError(f"{mode}: unequal learning rates")
        epochs = len(summary.get("history", []))
        if not epochs:
            raise ValueError(f"{mode}: no completed training epochs")
        if expected_epochs is None:
            expected_epochs = epochs
        elif epochs != expected_epochs:
            raise ValueError(f"{mode}: unequal training epoch counts")
        per_case = summary["test_at_best_validation"]["by_case"]
        per_val_case = summary["per_case_validation_at_best"]
        if set(per_case) != CASES or set(per_val_case) != CASES:
            raise ValueError(f"{mode}: missing per-case evaluation")
        rows.append({
            "mode": mode,
            "conditions": conditions,
            "epochs": epochs,
            "best_epoch": summary["best_epoch"],
            "validation_velocity_mse": _finite_mse(summary["best_validation_velocity_mse"], f"{mode}.validation"),
            "test_velocity_mse": _finite_mse(summary["test_at_best_validation"]["mean"], f"{mode}.test"),
            "per_case_test_velocity_mse": {name: _finite_mse(per_case[name]["mse"], f"{mode}.{name}") for name in sorted(CASES)},
            "per_case_validation_velocity_mse": {name: _finite_mse(per_val_case[name]["mse"], f"{mode}.{name}") for name in sorted(CASES)},
        })
    baseline = rows[0]["validation_velocity_mse"]
    for row in rows:
        row["validation_mse_delta_vs_none"] = row["validation_velocity_mse"] - baseline
    return {
        "status": "matched_three_arm_m47_comparison",
        "same_signature_except_condition": True,
        "same_split": True,
        "same_number_of_epochs": expected_epochs,
        "shared_learning_rate": expected_lr,
        "selection_metric": "held-out validation FM velocity MSE (lower is better)",
        "test_usage": "descriptive only; never select architecture using test data",
        "limitations": [
            "Velocity MSE does not establish generated PDFs, spectra, or spatial covariance.",
            "One seed; differences need replication and uncertainty estimates.",
            "Reference Reynolds number and resolution remain confounded across these six cases.",
            "Stationarity 41..100 is a reviewed exploratory eligibility window, not strict stationarity proof.",
        ],
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m47_conditioned"))
    parser.add_argument("--basename", default="train_s42")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.root, basename=args.basename)
    out = args.output or args.root / "comparison_s42.json"
    if out.exists():
        raise FileExistsError(f"Refusing overwrite of M47 comparison: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in result["results"]:
        print(f"{row['mode']:8} validation={row['validation_velocity_mse']:.6g} "
              f"test={row['test_velocity_mse']:.6g} "
              f"delta_vs_none={row['validation_mse_delta_vs_none']:+.6g}", flush=True)
    print(f"M47 matched velocity-MSE comparison: {out}", flush=True)


if __name__ == "__main__":
    main()