"""Aggregate the matched depth=2 versus depth=4 M45.6 full-all study.

Reads existing job directories only. Identical seed and train-sample count
identify matched datasets; duplicate keys are rejected, not silently averaged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(root: Path, output_dir: Path, *, exclude_jobs: frozenset[str] = frozenset()) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    paths = sorted(root.glob("job_*/summary.json"))
    available_jobs = {path.parent.name for path in paths}
    unknown = exclude_jobs - available_jobs
    if unknown:
        raise ValueError(f"excluded jobs not present in input root: {sorted(unknown)}")
    rows = []
    for path in paths:
        if path.parent.name in exclude_jobs:
            continue
        data = json.loads(path.read_text())
        settings = data["settings"]
        if settings.get("training_scope") != "full_all":
            continue
        layers = (settings["fine_pre_layers"], settings["fine_post_layers"],
                  settings["coarse_static_layers"], settings["coarse_dynamic_layers"])
        if len(set(layers)) != 1 or layers[0] not in (2, 4):
            continue
        if settings.get("val_seed") != 7002026:
            raise ValueError(f"unexpected validation seed: {path}")
        csv_path = path.parent / "diagnostics" / "generation_metrics.csv"
        if not csv_path.is_file():
            raise ValueError(f"missing fixed-coarse generation metrics: {csv_path}")
        with csv_path.open(newline="") as f:
            metrics = list(csv.DictReader(f))
        for result in data["results"]:
            if result["case"] != "shared_mode":
                continue
            metric = [r for r in metrics if r["case"] == "shared_mode"
                      and r["mode"] == result["mode"]
                      and r["variant"] == "normal" and int(r["heun_steps"]) == 25]
            if len(metric) != 1:
                raise ValueError(f"missing or duplicated normal 25-step metrics in {path}")
            row = {"job": path.parent.name, "seed": settings["seed"],
                   "train_samples": settings["train_samples"], "depth": layers[0],
                   "mode": result["mode"], "fm_loss": result["heldout_fm_loss"],
                   "generated_m2": float(metric[0]["mode_second_moment"]),
                   "analytic_m2": float(metric[0]["analytic_mode_second_moment"]),
                   "parameters": result["parameters"], "updates": settings["steps"],
                   "width": settings["width"], "refinement": settings["refinement"]}
            row["recovered_pct"] = 100 * row["generated_m2"] / row["analytic_m2"]
            rows.append(row)
    indexed = {}
    for row in rows:
        key = (row["seed"], row["train_samples"], row["depth"], row["mode"])
        if key in indexed:
            raise ValueError(f"duplicate run {key}: {indexed[key]['job']}, {row['job']}")
        indexed[key] = row
    matched = []
    for samples in sorted({r["train_samples"] for r in rows}):
        for seed in sorted({r["seed"] for r in rows}):
            for mode in ("static", "dynamic"):
                low = indexed.get((seed, samples, 2, mode))
                high = indexed.get((seed, samples, 4, mode))
                if low is None or high is None:
                    continue
                for field in ("updates", "width", "refinement"):
                    if low[field] != high[field]:
                        raise ValueError(f"mismatched {field} for {seed}/{samples}/{mode}")
                if low["analytic_m2"] != high["analytic_m2"]:
                    raise ValueError("analytical reference changed across depths")
                matched.append({"train_samples": samples, "seed": seed, "mode": mode,
                                "job_depth2": low["job"], "job_depth4": high["job"],
                                "fm_depth2": low["fm_loss"], "fm_depth4": high["fm_loss"],
                                "fm_improvement_pct": 100 * (low["fm_loss"] - high["fm_loss"]) / low["fm_loss"],
                                "recovered_depth2_pct": low["recovered_pct"],
                                "recovered_depth4_pct": high["recovered_pct"]})
    grouped = defaultdict(list)
    for row in matched:
        grouped[(row["train_samples"], row["mode"])].append(row)
    aggregates = []
    for (samples, mode), group in sorted(grouped.items()):
        gains = [r["fm_improvement_pct"] for r in group]
        aggregates.append({"train_samples": samples, "mode": mode,
                           "matched_seeds": len(group),
                           "mean_fm_improvement_pct": mean(gains),
                           "sd_fm_improvement_pct": stdev(gains) if len(gains) > 1 else None,
                           "mean_modal_recovery_depth2_pct": mean(x["recovered_depth2_pct"] for x in group),
                           "mean_modal_recovery_depth4_pct": mean(x["recovered_depth4_pct"] for x in group)})
    output_dir.mkdir(parents=True)
    _write(output_dir / "per_job.csv", rows)
    _write(output_dir / "matched_depth.csv", matched)
    _write(output_dir / "aggregates.csv", aggregates)
    summary = {"schema": "graph_attention.m45_6_depth_aggregate.v1",
               "root": str(root), "n_runs": len(rows),
               "excluded_jobs": sorted(exclude_jobs),
               "n_matched": len(matched), "aggregates": aggregates}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-job", action="append", default=[],
                        help="Explicit run-directory name to exclude (repeatable); "
                             "duplicates still error unless excluded")
    args = parser.parse_args()
    print(json.dumps(aggregate(
        args.root, args.output_dir, exclude_jobs=frozenset(args.exclude_job),
    ), indent=2))


if __name__ == "__main__":
    main()
