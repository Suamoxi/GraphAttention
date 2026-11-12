"""M45.8.1: compare four matched static/dynamic and coarse-context-radius runs.

Read-only aggregation; no checkpoint is loaded or modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from scripts.m45_checkpoint_diagnostics import _write_csv


def _csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _normalized_shape_rmse(generated: list[float], reference: list[float]) -> float:
    if min(generated + reference) <= 0:
        raise ValueError("conditional energies must be positive")
    gbar, rbar = mean(generated), mean(reference)
    return math.sqrt(mean(
        (g / gbar - r / rbar)**2 for g, r in zip(generated, reference)
    ))


def score_job(folder: Path) -> tuple[dict, list[dict]]:
    train = json.loads((folder / "summary.json").read_text())
    eval_report = json.loads((folder / "diagnostics" / "summary.json").read_text())
    if (train.get("schema") != "graph_attention.m45_8_local_train.v1"
            or eval_report.get("schema") != "graph_attention.m45_8_local_evaluate.v1"):
        raise ValueError(f"incompatible training/evaluation summaries in {folder}")
    s, e = train["settings"], eval_report["settings"]
    if (s["mode"] != e["mode"]
            or s["generator"] != e["generator"]
            or s["validation_seed"] != e["validation_seed"]):
        raise ValueError(f"checkpoint/evaluation mismatch: {folder}")

    bin_rows = _csv(folder / "diagnostics" / "conditional_summary.csv")
    by_dist = {}
    for dist in ("generated", "reference"):
        rows = sorted(
            (r for r in bin_rows if r["distribution"] == dist),
            key=lambda r: int(r["bin"]),
        )
        if len(rows) != 5:
            raise ValueError("expected five conditional strain bins")
        by_dist[dist] = [float(r["mean_kinetic_energy"]) for r in rows]
    g, r = by_dist["generated"], by_dist["reference"]
    curves = [
        {
            "mode": s["mode"], "context_hops": s["generator"]["context_hops"],
            "bin": i, "generated_energy": g[i], "reference_energy": r[i],
            "energy_ratio": g[i] / r[i] if r[i] > 0 else None,
        }
        for i in range(5)
    ]
    spectra = _csv(folder / "diagnostics" / "spectra.csv")
    by_spectrum = {}
    for row in spectra:
        key = (row["distribution"], int(row["shell_index"]))
        by_spectrum.setdefault(key, []).append(float(row["E_k"]))
    spectral_error = []
    cutoff = math.pi * s["coarse_side"]
    max_reference = max(mean(v) for (dist, _), v in by_spectrum.items()
                        if dist == "reference")
    for (dist, shell), vals in by_spectrum.items():
        if dist != "reference" or not vals:
            continue
        kphys = shell * 2 * math.pi
        ref = mean(vals)
        generated = mean(by_spectrum.get(("generated", shell), []))
        # Omit empty/tiny spectral shells from log comparisons.
        if kphys >= cutoff and ref > max_reference * 1e-6 and generated > 0:
            spectral_error.append(math.log(generated / ref)**2)
    if not spectral_error:
        raise ValueError("no comparable subgrid spectral shells")

    corr_rows = _csv(folder / "diagnostics" / "spatial_correlations.csv")
    by_corr = {}
    for row in corr_rows:
        if int(row["lag"]) in (1, 2, 3, 4):
            key = (row["distribution"], row["axis"], int(row["lag"]))
            by_corr.setdefault(key, []).append(float(row["correlation"]))
    corr_errors = []
    for (dist, axis, lag), vals in by_corr.items():
        if dist == "reference":
            generated = by_corr.get(("generated", axis, lag), [])
            if len(generated) != len(vals):
                raise ValueError("incomplete paired spatial correlations")
            corr_errors.append((mean(generated) - mean(vals))**2)
    result = {
        "mode": s["mode"], "context_hops": s["generator"]["context_hops"],
        "train_fields": s["train_fields"], "train_steps": s["steps"],
        "width": s["width"], "depth": s["depth"],
        "coarse_side": s["coarse_side"], "refinement": s["refinement"],
        "normalization_fields": s["normalization_fields"],
        "residual_policy": s["residual_policy"],
        "data_seed": s["dataset_seed"], "residual_seed": s["residual_seed"],
        "model_init_seed": s["init_seed"], "schedule_seed": s["schedule_seed"],
        "validation_seed": s["validation_seed"],
        "eval_seed": e["eval_seed"],
        "eval_fields": e["val_fields"], "draws": e["draws"],
        "heun_steps": e["heun_steps"],
        "heldout_fm_mse": train["results"]["heldout_fm_loss"],
        "residual_energy_ratio": (
            eval_report["results"]["generated"]["mean_residual_energy"]
            / eval_report["results"]["reference"]["mean_residual_energy"]
        ),
        "total_energy_ratio": (
            eval_report["results"]["generated"]["mean_total_energy"]
            / eval_report["results"]["reference"]["mean_total_energy"]
        ),
        "conditional_shape_rmse": _normalized_shape_rmse(g, r),
        "conditional_energy_relative_error": mean(
            abs(gg / rr - 1) for gg, rr in zip(g, r)
        ),
        "conditional_reference_high_low_ratio": r[-1] / r[0],
        "conditional_generated_high_low_ratio": g[-1] / g[0],
        "subgrid_log_spectral_rmse": math.sqrt(mean(spectral_error)),
        "spatial_corr_rmse_lags1to4": math.sqrt(mean(corr_errors)),
        "max_coarse_defect": eval_report["results"]["generated"]["max_coarse_defect"],
        "generated_divergence_rms": eval_report["results"]["generated"]["mean_fine_divergence_rms"],
        "reference_divergence_rms": eval_report["results"]["reference"]["mean_fine_divergence_rms"],
    }
    return result, curves


def aggregate(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing existing output {output}")
    jobs = sorted(root.glob("task_*"))
    if len(jobs) != 4:
        raise ValueError(f"expected four completed tasks, found {len(jobs)}")
    scores, curves = [], []
    for path in jobs:
        score, graph = score_job(path)
        scores.append(score)
        curves.extend(graph)
    keys = {(r["mode"], r["context_hops"]) for r in scores}
    if keys != {("static", 1), ("dynamic", 1), ("static", 2), ("dynamic", 2)}:
        raise ValueError(f"missing or duplicate model/context comparison: {keys}")
    for key in ("train_fields", "train_steps", "width", "depth",
                "coarse_side", "refinement", "normalization_fields",
                "residual_policy", "data_seed", "residual_seed",
                "model_init_seed", "schedule_seed", "validation_seed",
                "eval_seed", "eval_fields", "draws", "heun_steps"):
        if len({r[key] for r in scores}) != 1:
            raise ValueError(f"mismatched comparison setting {key}")
    output.mkdir(parents=True)
    _write_csv(output / "scorecard.csv", scores, list(scores[0]))
    _write_csv(output / "conditional_curves.csv", curves, list(curves[0]))
    report = {
        "schema": "graph_attention.m45_8_local_comparison.v1",
        "scope": "statistical HIT-inspired synthetic fields, not DNS/LES",
        "settings": {
            "coarse_context_hops": [1, 2], "modes": ["static", "dynamic"],
            "training_fields": scores[0]["train_fields"],
            "training_updates": scores[0]["train_steps"],
            "evaluation_fields": scores[0]["eval_fields"],
            "draws_per_field": scores[0]["draws"],
            "heun_steps": scores[0]["heun_steps"],
        },
        "results": scores,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.root, args.output_dir)


if __name__ == "__main__":
    main()
