"""M47.4 validation-only six-run comparison (3 conditions x epochs 5/10).

This script never reads M47.1/M47.4 test-at-best metrics. It requires the
frozen original optimizer continuation, matching validation populations and
matching independently generated latent keys.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from scripts.m47_4_stage import ORIGINAL_SIGNATURES

MODES = ("none", "mach", "mach_re")
STAGES = (5, 10)
CASES = tuple(f"{r}/Ma{m}" for r in ("Re54", "Re94")
              for m in ("0.05", "0.2", "0.3"))
CHANNELS = ("rho", "rhou", "rhov", "rhow", "rhoE")
TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)


def read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def aggregate_mse(per_case: dict[str, Any], t: float) -> float:
    values = []
    for case in CASES:
        row = per_case[case][str(t)]
        if row["samples"] != 5 or set(row["channels"]) != set(CHANNELS):
            raise ValueError(f"invalid per-case/channel validation count: {case}, {t}")
        for channel in CHANNELS:
            mse = float(row["channels"][channel]["mse"])
            if not math.isfinite(mse) or mse < 0:
                raise ValueError("invalid validation MSE")
            values.append(mse)
    return sum(values) / len(values)


def validate_six_summaries(results: dict[tuple[str, int], dict[str, Any]]) -> None:
    if set(results) != {(mode, epochs) for mode in MODES for epochs in STAGES}:
        raise ValueError("M47.4 comparison needs exactly three arms x two checkpoints")
    common = None
    physical_keys = None
    physical_population = None
    for (mode, epochs), s in results.items():
        if (s.get("status") != "complete"
            or s.get("mode") != mode
            or s.get("epochs_completed") != epochs
            or s.get("training_signature_sha256") != ORIGINAL_SIGNATURES[mode]
            or s.get("validation_per_case") != 5
            or s.get("test_snapshot_state_used") is not False
            or s.get("flow_times") != list(TIMES)):
            raise ValueError(f"{mode} epoch {epochs}: inconsistent scientific contract")
        setup = (s["shared_sampling_seed"], s["physical_generation_steps"],
                 s["physical_generation_samples_per_case"],
                 tuple(s["physical_generation_cases"]))
        if common is None:
            common = setup
        elif common != setup:
            raise ValueError("M47.4 sampling settings differ across modes/stages")
        current_keys = {}
        current_population = {}
        for case, report in s["physical_generation"].items():
            if (case not in CASES or report["generated_count"] != setup[2]
                or report["reference_count"] != 5):
                raise ValueError("invalid physical generation case or population")
            current_keys[case] = tuple(report["generated_sampling_keys"])
            current_population[case] = (
                tuple(report["reference_sample_ids"]),
                tuple(report["unique_grid_shape"]),
                tuple(report["source_grid_shape"]),
            )
        if physical_keys is None:
            physical_keys = current_keys
            physical_population = current_population
        elif current_keys != physical_keys or current_population != physical_population:
            raise ValueError("M47.4 latent keys, DNS references or grids differ")
        for t in TIMES:
            aggregate_mse(s["validation_velocity_mse_at_fixed_times"], t)


def compare(root: Path, *, steps: int = 50, samples: int = 4,
            seed: int = 20261116) -> dict[str, Any]:
    results = {}
    for mode in MODES:
        for stage in STAGES:
            path = (root / mode /
                    f"eval_epoch{stage:02d}_h{steps}_n{samples}_seed{seed}" /
                    "summary.json")
            results[(mode, stage)] = read(path)
    validate_six_summaries(results)

    # Confirm that epoch-10 history is a strict extension of the frozen
    # epoch-5 run (rather than an independent seed-42 restart).
    for mode in MODES:
        five = read(root / mode / "epoch05" / "summary.json")["history"]
        ten = read(root / mode / "epoch10" / "summary.json")["history"]
        if len(five) != 5 or len(ten) != 10 or ten[:5] != five:
            raise ValueError(f"{mode}: epoch-10 training history does not extend epoch 5")

    rows = []
    for mode in MODES:
        for stage in STAGES:
            summary = results[(mode, stage)]
            row = {
                "mode": mode,
                "epochs": stage,
                "checkpoint": summary["evaluated_checkpoint"],
                "best_validation_epoch_in_training": summary["training_validation_best_epoch"],
                "best_validation_mse_in_training": summary["training_validation_best_mse"],
                "mse_by_flow_time": {
                    str(t): aggregate_mse(
                        summary["validation_velocity_mse_at_fixed_times"], t)
                    for t in TIMES
                },
                "physical": {
                    case: {
                        "validation_reference_count": report["reference_count"],
                        "generated_count": report["generated_count"],
                        "density_valid_count": report["generated_density_valid_count"],
                        "thermodynamically_valid_count":
                            report["generated_thermodynamically_valid_count"],
                        "mean_bad_internal_energy_fraction":
                            report["generated_mean_bad_thermo_fraction"],
                        "mean_dilatational_energy_fraction":
                            report["generated_mean_dilatational_energy_fraction"],
                        "relative_velocity_spectrum_l1":
                            report["relative_velocity_spectrum_l1"],
                        "velocity_correlation_mae":
                            report["vector_correlation_mean_absolute_error"],
                        "periodic_seam_jump_over_std":
                            report["generated_seam_jump_over_std"],
                        "normalized_wasserstein":
                            {ch: report["one_point_wasserstein"][ch][
                                "wasserstein_over_ref_std"] for ch in CHANNELS},
                    } for case, report in summary["physical_generation"].items()
                },
            }
            rows.append(row)
    for mode in MODES:
        a = next(r for r in rows if r["mode"] == mode and r["epochs"] == 5)
        b = next(r for r in rows if r["mode"] == mode and r["epochs"] == 10)
        b["mse_change_from_epoch5"] = {
            str(t): b["mse_by_flow_time"][str(t)] - a["mse_by_flow_time"][str(t)]
            for t in TIMES
        }
    return {
        "status": "complete_matched_m47_4_validation_comparison",
        "cohort": "30 original validation snapshots, not held-out test",
        "modes": list(MODES),
        "epochs": list(STAGES),
        "shared_sampling_seed": seed,
        "generation_steps": steps,
        "samples_per_case": samples,
        "test_metrics_influencing_selection": False,
        "caveats": [
            "Reynolds reference and mesh resolution are confounded.",
            "Consecutive validation snapshots are not independent replicates.",
            "Unavailable velocity spectra remain null for invalid-density fields.",
            "The pinned legacy trainer evaluates held-out test after each stage; "
            "do not use those metrics to select M47.4 checkpoints.",
        ],
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m47_4_longer_training"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    result = compare(args.root, steps=args.steps, samples=args.samples, seed=args.seed)
    output = args.output or args.root / "comparison_epoch05_epoch10.json"
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    csv_path = output.with_suffix(".csv")
    if csv_path.exists():
        raise FileExistsError(f"refusing overwrite: {csv_path}")
    with csv_path.open("w", newline="") as f:
        columns = ["mode", "epochs", *(f"mse_t{t}" for t in TIMES),
                   "best_validation_epoch_in_training", "best_validation_mse_in_training"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in result["results"]:
            writer.writerow({
                "mode": row["mode"], "epochs": row["epochs"],
                **{f"mse_t{t}": row["mse_by_flow_time"][str(t)] for t in TIMES},
                "best_validation_epoch_in_training": row["best_validation_epoch_in_training"],
                "best_validation_mse_in_training": row["best_validation_mse_in_training"],
            })
    print(f"[M47.4] complete six-run comparison: {output}", flush=True)


if __name__ == "__main__":
    main()
