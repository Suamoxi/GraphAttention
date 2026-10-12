"""Summarize completed M41.1 per-layout GH200 JSON reports as one CSV.

Does not run any GPU work. Refuses to mix checkpoints, samples, or precision.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _comparison_contract(report: dict) -> tuple:
    generation = report["generation"]
    return (
        report["sample_id"],
        report["num_nodes"],
        report["backend"],
        report["precision"],
        json.dumps(report["artifact_sha256"], sort_keys=True),
        generation["steps"],
        generation["seed"],
    )


def _max_numeric(values) -> float:
    valid = [value for value in values if value is not None]
    return max(valid) if valid else float("nan")


def _report_row(report: dict) -> dict:
    original = report["original"]
    partitioned = report["partitioned"]
    generation = report["generation"]
    layers = report["errors"]["layers"]
    partitions = report["partitions"]
    original_generation = generation["original"]["median_ms"]
    partitioned_generation = generation["partitioned"]["median_ms"]
    return {
        "blocks": "x".join(map(str, report["blocks"])),
        "num_partitions": report["num_partitions"],
        "num_nodes": report["num_nodes"],
        "nodewise_work_ratio": report["nodewise_work_ratio"],
        "total_halo_copies": sum(p["halo"] for p in partitions),
        "max_halo_per_partition": max(p["halo"] for p in partitions),
        "plan_build_s": report["plan_build_seconds"],
        "plan_allocated_gib": report["plan_persistent_allocated_gib"],
        "original_forward_median_ms": original["median_ms"],
        "partitioned_forward_median_ms": partitioned["median_ms"],
        "forward_overhead_pct": 100 * (
            partitioned["median_ms"] / original["median_ms"] - 1
        ),
        "original_heun_median_ms": original_generation,
        "partitioned_heun_median_ms": partitioned_generation,
        "heun_overhead_pct": 100 * (
            partitioned_generation / original_generation - 1
        ),
        "original_forward_peak_allocated_gib": original["peak_allocated_gib"],
        "partitioned_forward_peak_allocated_gib": partitioned["peak_allocated_gib"],
        "original_incremental_peak_gib": original["incremental_peak_allocated_gib"],
        "partitioned_incremental_peak_gib": partitioned["incremental_peak_allocated_gib"],
        "max_layer_abs_error": _max_numeric(item["max_abs"] for item in layers),
        "max_layer_relative_l2": _max_numeric(item["relative_l2"] for item in layers),
        "output_max_abs_error": report["errors"]["output"]["max_abs"],
        "heun_max_abs_error": report["errors"]["heun"]["max_abs"],
        "passed": report["passed"],
        "evidence": report["evidence"],
    }


def summarize(input_dir: Path) -> list[dict]:
    paths = sorted(input_dir.glob("blocks_*.json"))
    if not paths:
        raise FileNotFoundError(f"no blocks_*.json reports in {input_dir}")
    reports = [json.loads(path.read_text()) for path in paths]
    contract = _comparison_contract(reports[0])
    if any(_comparison_contract(report) != contract for report in reports[1:]):
        raise ValueError("M41.1 reports use different checkpoints, samples, or settings")
    blocks = [tuple(report["blocks"]) for report in reports]
    if len(set(blocks)) != len(blocks):
        raise ValueError("duplicate M41.1 partition layouts")
    return [
        _report_row(report)
        for report in sorted(
            reports, key=lambda report: (report["num_partitions"], tuple(report["blocks"]))
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    rows = summarize(args.input_dir)
    output = args.output_csv or args.input_dir / "comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison CSV: {output}")
    for row in rows:
        print(
            f"{row['blocks']:>7s} P={row['num_partitions']:>3d} "
            f"forward={row['partitioned_forward_median_ms']:.2f} ms "
            f"({row['forward_overhead_pct']:+.1f}%) "
            f"Heun={row['partitioned_heun_median_ms']:.2f} ms "
            f"({row['heun_overhead_pct']:+.1f}%) "
            f"max_layer_abs={row['max_layer_abs_error']:.3e} "
            f"pass={row['passed']}"
        )
    if not all(row["passed"] for row in rows):
        raise SystemExit("At least one M41.1 numerical validation failed")


if __name__ == "__main__":
    main()
