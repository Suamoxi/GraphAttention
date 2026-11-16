"""Audit and aggregate matched, independent M47.2 generated 3-D HIT ensembles.

The comparison is descriptive; training used validation data for checkpoint
selection and this test partition must not be recycled for architecture tuning.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from scripts.m47_generation_metrics import CHANNELS, finite_mean

MODES = ("none", "mach", "mach_re")
CASES = tuple(
    f"{re}/Ma{ma}" for re in ("Re54", "Re94") for ma in ("0.05", "0.2", "0.3")
)


def check_contract(results: dict[str, dict[str, Any]]) -> None:
    """Reject mismatched Monte Carlo sampling, references, and FFT settings."""
    if set(results) != set(MODES):
        raise ValueError("M47.2 requires all three conditioning arms")
    shared_keys = (
        "version", "generator_code_revision", "held_out_reference_pairing",
        "gaussian_sampling_seed", "sampling_steps",
        "heun_model_evaluations_per_generated_field", "samples_per_case",
        "node_subsample_per_field", "energy_spectral_bins",
        "spatial_correlation_max_lag", "periodic_fft_endpoint_mode",
    )
    baseline = results["none"]
    if baseline.get("version") != "m47_independent_ensemble_v1":
        raise ValueError("unexpected M47.2 generation result format")
    if baseline.get("held_out_reference_pairing") is not False:
        raise ValueError("M47.2 generated and reference fields must be unpaired")
    if baseline.get("periodic_fft_endpoint_mode") != "drop_max":
        raise ValueError("FFT endpoints must be dropped")
    if set(baseline.get("cases", {})) != set(CASES):
        raise ValueError("full M47.2 comparison must include all six cases")
    for mode, result in results.items():
        if (result.get("status") != "complete"
            or result.get("conditioning_mode") != mode):
            raise ValueError(f"{mode}: missing or incomplete generation results")
        if set(result.get("cases", {})) != set(CASES):
            raise ValueError(f"{mode}: missing cases")
        for key in shared_keys:
            if result.get(key) != baseline.get(key):
                raise ValueError(f"{mode}: mismatched ensemble setup {key}")
        for case in CASES:
            row = result["cases"][case]
            reference = baseline["cases"][case]
            case_keys = (
                "case", "reference_sample_ids", "generated_sampling_keys",
                "source_grid_shape", "unique_grid_shape",
                "physical_grid_spacing_m", "reference_velocity_scale_m_per_s",
                "reference_count", "generated_count", "reference_density_valid_count",
                "reference_thermodynamically_valid_count", "reference_mean_velocity_spectrum",
                "reference_mean_dilatational_energy_fraction",
                "reference_mean_velocity_correlation", "reference_seam_jump_over_std",
            )
            for key in case_keys:
                if row.get(key) != reference.get(key):
                    raise ValueError(f"{mode}/{case}: mismatched reference population or {key}")
            if (row["generated_count"] != result["samples_per_case"]
                or row["reference_count"] != 7):
                raise ValueError(f"{mode}/{case}: incorrect population size")
            for field in CHANNELS:
                estimate = row["one_point_wasserstein"][field]
                value = estimate["wasserstein_over_ref_std"]
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ValueError(f"{mode}/{case}/{field}: invalid Wasserstein metric")
            for name in ("generated_density_valid_count", "generated_thermodynamically_valid_count"):
                if not 0 <= row[name] <= result["samples_per_case"]:
                    raise ValueError(f"{mode}/{case}: invalid {name}")


def compare(root: Path, *, basename: str) -> dict[str, Any]:
    results = {}
    for mode in MODES:
        path = root / mode / basename / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing M47.2 completed ensemble: {path}")
        results[mode] = json.loads(path.read_text())
    check_contract(results)
    rows = []
    aggregates = []
    for mode in MODES:
        source = results[mode]
        for case in CASES:
            report = source["cases"][case]
            record = {
                "mode": mode,
                "case": case,
                "rho_nonpositive_fraction_mean": report["generated_mean_bad_density_fraction"],
                "density_valid_fields": report["generated_density_valid_count"],
                "thermo_valid_fields": report["generated_thermodynamically_valid_count"],
                "population_size": report["generated_count"],
                "relative_velocity_spectrum_l1": report["relative_velocity_spectrum_l1"],
                "dilatational_energy_fraction": report["generated_mean_dilatational_energy_fraction"],
                "reference_dilatational_energy_fraction": report["reference_mean_dilatational_energy_fraction"],
                "spatial_correlation_mae": report["vector_correlation_mean_absolute_error"],
                "mean_generated_periodic_seam_jump_over_std": finite_mean([
                    report["generated_seam_jump_over_std"][name] for name in CHANNELS
                ]),
            }
            for name in CHANNELS:
                record[f"{name}_W1_over_ref_std"] = report["one_point_wasserstein"][name]["wasserstein_over_ref_std"]
            rows.append(record)
        matched = [row for row in rows if row["mode"] == mode]
        valid = sum(r["density_valid_fields"] for r in matched)
        total = sum(r["population_size"] for r in matched)
        mean_w1 = finite_mean([
            r[f"{name}_W1_over_ref_std"] for r in matched for name in CHANNELS
        ])
        aggregates.append({
            "mode": mode,
            "conditioned_training": mode != "none",
            "cases": len(matched),
            "samples": total,
            "density_valid_fields": valid,
            "thermodynamically_valid_fields": sum(r["thermo_valid_fields"] for r in matched),
            "density_valid_field_fraction": valid / total,
            "mean_normalized_conservative_field_W1": mean_w1,
            "mean_relative_velocity_spectrum_l1": finite_mean([
                r["relative_velocity_spectrum_l1"] for r in matched
            ]),
            "mean_spatial_correlation_mae": finite_mean([
                r["spatial_correlation_mae"] for r in matched
            ]),
            "mean_generated_periodic_seam_jump_over_std": finite_mean([
                r["mean_generated_periodic_seam_jump_over_std"] for r in matched
            ]),
        })
    return {
        "version": "m47_independent_ensemble_compare_v1",
        "status": "matched_three_arm_generation_comparison",
        "condition_order": list(MODES),
        "same_sampling_initial_conditions": True,
        "same_held_out_reference_populations": True,
        "sampling_steps": results["none"]["sampling_steps"],
        "heun_nfe_per_sample": results["none"]["heun_model_evaluations_per_generated_field"],
        "sampling_seed": results["none"]["gaussian_sampling_seed"],
        "note": "Test references are unpaired. Spectra excluded for fields with nonpositive density, not clipped.",
        "interpretation_limits": [
            "Eight or fewer generated fields per case remain a small Monte Carlo population.",
            "Seven held-out snapshots per case may be temporally correlated.",
            "No independent seed replication or unseen Mach/Reynolds conditions.",
            "Resolution and Reynolds labels remain confounded.",
            "Only two training epochs, convergence not established.",
            "Physical observables must not be inferred from velocity-prediction MSE alone.",
        ],
        "per_case_rows": rows,
        "overall_rows": aggregates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m47_generated"))
    parser.add_argument("--basename", default="ensemble_s42_h50_n8_seed20261116")
    args = parser.parse_args()
    outcome = compare(args.root, basename=args.basename)
    output_path = args.root / f"comparison_{args.basename}.json"
    csv_path = args.root / f"comparison_{args.basename}.csv"
    if output_path.exists() or csv_path.exists():
        raise FileExistsError("M47.2 refuses to overwrite an existing comparison")
    output_path.write_text(json.dumps(outcome, indent=2, allow_nan=False) + "\n")
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(outcome["per_case_rows"][0]))
        writer.writeheader()
        writer.writerows(outcome["per_case_rows"])
    print(f"M47.2 matched generation analysis: {output_path}", flush=True)
    for row in outcome["overall_rows"]:
        print(f"{row['mode']:8} density_valid={row['density_valid_field_fraction']:.3f} "
              f"mean_normalized_W1={row['mean_normalized_conservative_field_W1']} "
              f"mean_spectrum_L1={row['mean_relative_velocity_spectrum_l1']}",
              flush=True)


if __name__ == "__main__":
    main()
