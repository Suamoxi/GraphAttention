"""M45.7.3: aggregate matched multi-coarse evaluations across training seeds.

Read-only CPU analysis. Never pool modal amplitudes across conditioning fields
before centering: variance/recovery is computed within each U_c first.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from scripts.m45_checkpoint_diagnostics import _write_csv

SEEDS = (2026, 2027, 2028)
POLICIES = ("fixed", "resample")


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _paired_rows(rows: list[dict], *, expected_fields: int) -> list[dict]:
    if not 1 <= expected_fields <= 8:
        raise ValueError("expected_fields must be in [1,8]")
    by_key: dict[tuple[int, str, int], dict] = {}
    references: dict[int, tuple[float, float, float]] = {}
    for row in rows:
        seed = int(row["seed"])
        policy = row["residual_policy"]
        index = int(row["validation_index"])
        if seed not in SEEDS or policy not in POLICIES or not 0 <= index < expected_fields:
            raise ValueError(f"unexpected seed/policy/validation_index: {seed}/{policy}/{index}")
        if int(row["sample_id"]) != 100000 + index:
            raise ValueError(f"incorrect held-out sample_id for index {index}")
        key = (seed, policy, index)
        if key in by_key:
            raise ValueError(f"duplicate configuration/field: {key}")
        by_key[key] = row
        ref = (
            float(row["analytic_mode_variance"]),
            float(row["reference_mode_variance"]),
            float(row["reference_mode_mean"]),
        )
        if index in references and any(
            abs(a - b) > 1e-8 for a, b in zip(ref, references[index])
        ):
            raise ValueError(f"reference statistics disagree for validation index {index}")
        references[index] = ref

    expected = {
        (seed, policy, index)
        for seed in SEEDS for policy in POLICIES
        for index in range(expected_fields)
    }
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise ValueError(f"incomplete seed/policy/field matrix: missing={missing} extra={extra}")

    result = []
    for seed in SEEDS:
        for index in range(expected_fields):
            a = by_key[(seed, "fixed", index)]
            b = by_key[(seed, "resample", index)]
            af = float(a["recovery_vs_analytic"])
            br = float(b["recovery_vs_analytic"])
            result.append({
                "seed": seed,
                "validation_index": index,
                "sample_id": 100000 + index,
                "analytic_mode_variance": float(a["analytic_mode_variance"]),
                "reference_mode_variance": float(a["reference_mode_variance"]),
                "fixed_recovery_analytic_pct": 100 * af,
                "resampled_recovery_analytic_pct": 100 * br,
                "delta_recovery_pp": 100 * (br - af),
                "fixed_recovery_empirical_pct": 100 * float(a["recovery_vs_empirical"]),
                "resampled_recovery_empirical_pct": 100 * float(b["recovery_vs_empirical"]),
                "fixed_mode_mean": float(a["generated_mode_mean"]),
                "resampled_mode_mean": float(b["generated_mode_mean"]),
                "fixed_corr_x_lag1": float(a["generated_corr_x_lag1"]),
                "resampled_corr_x_lag1": float(b["generated_corr_x_lag1"]),
                "fixed_corr_x_lag4": float(a["generated_corr_x_lag4"]),
                "resampled_corr_x_lag4": float(b["generated_corr_x_lag4"]),
                "delta_corr_x_lag4": (
                    float(b["generated_corr_x_lag4"])
                    - float(a["generated_corr_x_lag4"])
                ),
                "fixed_max_Q": float(a["max_Q_generated"]),
                "resampled_max_Q": float(b["max_Q_generated"]),
            })
    return result


def _seed_summary(paired: list[dict]) -> list[dict]:
    result = []
    for seed in SEEDS:
        samples = [row for row in paired if row["seed"] == seed]
        if not samples:
            raise ValueError(f"missing seed {seed}")
        out = {"seed": seed, "validation_fields": len(samples)}
        for key in ("fixed_recovery_analytic_pct",
                    "resampled_recovery_analytic_pct",
                    "delta_recovery_pp",
                    "fixed_corr_x_lag4",
                    "resampled_corr_x_lag4",
                    "delta_corr_x_lag4"):
            values = [row[key] for row in samples]
            out[key + "_mean"] = mean(values)
            out[key + "_std_across_fields"] = (
                stdev(values) if len(values) > 1 else 0.0
            )
        result.append(out)
    return result


def _field_summary(paired: list[dict], *, expected_fields: int) -> list[dict]:
    result = []
    for index in range(expected_fields):
        samples = [row for row in paired if row["validation_index"] == index]
        if len(samples) != len(SEEDS):
            raise ValueError(f"missing training seed for validation field {index}")
        out = {"validation_index": index, "training_seeds": len(samples)}
        for key in ("fixed_recovery_analytic_pct",
                    "resampled_recovery_analytic_pct", "delta_recovery_pp",
                    "fixed_corr_x_lag4", "resampled_corr_x_lag4"):
            vals = [row[key] for row in samples]
            out[key + "_mean_across_seeds"] = mean(vals)
            out[key + "_std_across_seeds"] = stdev(vals)
        result.append(out)
    return result


def aggregate(*, input_dir: Path, output_dir: Path,
              expected_fields: int = 8, heun_steps: int = 200,
              samples: int = 128, eval_seed: int = 53177) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing existing aggregate directory: {output_dir}")
    jobs = sorted(input_dir.glob("task_*/summary.json"))
    if len(jobs) != len(SEEDS) * len(POLICIES):
        raise ValueError(f"expected six completed jobs, got {len(jobs)}")

    all_rows, config_to_path = [], {}
    for path in jobs:
        report = json.loads(path.read_text())
        if report.get("schema") != "graph_attention.m45_7_3_condition_sweep.v1":
            raise ValueError(f"unexpected evaluation schema in {path}")
        config = report["evaluation"]
        settings = report["checkpoint_settings"]
        key = (int(config["training_seed"]), config["residual_policy"])
        if key in config_to_path:
            raise ValueError(f"duplicate checkpoint configuration: {key}")
        if (int(config["validation_fields"]) != expected_fields
                or int(config["heun_steps"]) != heun_steps
                or int(config["generated_samples"]) != samples
                or int(config["eval_seed"]) != eval_seed):
            raise ValueError(f"incompatible sampling protocol: {path}")
        if (settings["supervision"] != "oracle"
                or settings["steps"] != 10000
                or settings["val_seed"] != 7002026
                or settings["seed"] != key[0]
                or settings["residual_policy"] != key[1]):
            raise ValueError(f"incompatible checkpoint settings: {path}")
        rows = _read_rows(path.parent / "per_field.csv")
        if len(rows) != expected_fields:
            raise ValueError(f"wrong number of validation fields: {path}")
        for row in rows:
            if (int(row["seed"]), row["residual_policy"]) != key:
                raise ValueError(f"CSV contents disagree with metadata: {path}")
            if int(row["generated_samples"]) != samples or int(row["heun_steps"]) != heun_steps:
                raise ValueError(f"CSV sampling protocol disagrees: {path}")
        config_to_path[key] = str(path.parent)
        all_rows.extend(rows)

    paired = _paired_rows(all_rows, expected_fields=expected_fields)
    seed_rows = _seed_summary(paired)
    field_rows = _field_summary(paired, expected_fields=expected_fields)
    deltas = [r["delta_recovery_pp_mean"] for r in seed_rows]
    report = {
        "schema": "graph_attention.m45_7_3_aggregate.v1",
        "scope": "synthetic shared_mode block-average Q/B, NOT physical DNS/LES",
        "settings": {
            "training_seeds": list(SEEDS), "policies": list(POLICIES),
            "validation_fields_per_checkpoint": expected_fields,
            "generated_samples_per_field": samples,
            "heun_steps": heun_steps, "eval_seed": eval_seed,
            "primary_metric": "within-field centered modal variance / analytic conditional variance",
            "uncertainty_unit": "training seed, not individual generated sample",
        },
        "checkpoint_directories": [
            {"seed": seed, "policy": policy, "path": config_to_path[(seed, policy)]}
            for seed in SEEDS for policy in POLICIES
        ],
        "results": {
            "fixed_recovery_pct_mean_across_seeds": mean(
                x["fixed_recovery_analytic_pct_mean"] for x in seed_rows
            ),
            "resampled_recovery_pct_mean_across_seeds": mean(
                x["resampled_recovery_analytic_pct_mean"] for x in seed_rows
            ),
            "paired_resampling_gain_pp_mean_across_seeds": mean(deltas),
            "paired_resampling_gain_pp_sd_across_seeds": stdev(deltas),
            "paired_seed_differences_pp": deltas,
            "max_coarse_defect": max(float(r["max_Q_generated"]) for r in all_rows),
        },
    }

    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "paired_per_field.csv", paired, list(paired[0]))
    _write_csv(output_dir / "per_seed.csv", seed_rows, list(seed_rows[0]))
    _write_csv(output_dir / "per_validation_field.csv", field_rows, list(field_rows[0]))
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"M45.7.3 aggregated {len(jobs)} checkpoints, {len(paired)} seed/field pairs", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-fields", type=int, default=8)
    parser.add_argument("--heun-steps", type=int, default=200)
    parser.add_argument("--generated-samples", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=53177)
    args = parser.parse_args()
    aggregate(
        input_dir=args.input_dir, output_dir=args.output_dir,
        expected_fields=args.expected_fields, heun_steps=args.heun_steps,
        samples=args.generated_samples, eval_seed=args.eval_seed,
    )


if __name__ == "__main__":
    main()
