"""Compare full-block M41.1 and owned-query M41.2 on identical M37 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _shared_contract(report: dict) -> tuple:
    generation = report["generation"]
    return (
        report["sample_id"], report["num_nodes"], report["backend"], report["precision"],
        json.dumps(report["artifact_sha256"], sort_keys=True),
        generation["steps"], generation["seed"],
    )


def _contract(report: dict) -> tuple:
    return _shared_contract(report) + (
        tuple(report["blocks"]), report["num_partitions"],
    )


def compare(directory: Path) -> list[dict]:
    baselines = sorted(directory.glob("full_blocks_*.json"))
    candidates = sorted(directory.glob("owned_blocks_*.json"))
    if not baselines or len(baselines) != len(candidates):
        raise ValueError("M41.2 requires paired full_blocks_*.json and owned_blocks_*.json")
    rows = []
    shared_contract = None
    for baseline_path in baselines:
        suffix = baseline_path.name.removeprefix("full_")
        candidate_path = directory / ("owned_" + suffix)
        if not candidate_path.is_file():
            raise ValueError(f"missing M41.2 candidate for {baseline_path.name}")
        baseline = json.loads(baseline_path.read_text())
        candidate = json.loads(candidate_path.read_text())
        if _contract(baseline) != _contract(candidate):
            raise ValueError(f"comparison contract mismatch for {suffix}")
        if shared_contract is None:
            shared_contract = _shared_contract(baseline)
        elif _shared_contract(baseline) != shared_contract:
            raise ValueError("M41.2 layouts use different checkpoints or samples")
        if (
            baseline.get("executor") != "layerwise_full"
            or candidate.get("executor") != "owned_query"
        ):
            raise ValueError("M41.2 executor labels do not match the requested comparison")
        full_forward = baseline["partitioned"]["median_ms"]
        owned_forward = candidate["partitioned"]["median_ms"]
        full_heun = baseline["generation"]["partitioned"]["median_ms"]
        owned_heun = candidate["generation"]["partitioned"]["median_ms"]
        rows.append({
            "blocks": "x".join(map(str, candidate["blocks"])),
            "partitions": candidate["num_partitions"],
            "read_node_ratio": candidate["nodewise_work_ratio"],
            "m37_forward_median_ms": candidate["original"]["median_ms"],
            "m41_1_forward_median_ms": full_forward,
            "m41_2_forward_median_ms": owned_forward,
            "owned_vs_full_forward_speedup": full_forward / owned_forward,
            "m41_2_vs_m37_forward_ratio": owned_forward / candidate["original"]["median_ms"],
            "m41_1_heun_median_ms": full_heun,
            "m41_2_heun_median_ms": owned_heun,
            "owned_vs_full_heun_speedup": full_heun / owned_heun,
            "m41_1_forward_peak_gib": baseline["partitioned"]["peak_allocated_gib"],
            "m41_2_forward_peak_gib": candidate["partitioned"]["peak_allocated_gib"],
            "m41_1_plan_gib": baseline["plan_persistent_allocated_gib"],
            "m41_2_plan_gib": candidate["plan_persistent_allocated_gib"],
            "m41_2_max_layer_abs": max(
                item["max_abs"] if item["max_abs"] is not None else float("inf")
                for item in candidate["errors"]["layers"]
            ),
            "m41_2_output_max_abs": candidate["errors"]["output"]["max_abs"],
            "m41_2_heun_max_abs": candidate["errors"]["heun"]["max_abs"],
            "m41_1_passed": baseline["passed"],
            "m41_2_passed": candidate["passed"],
            "passed": baseline["passed"] and candidate["passed"],
        })
    return sorted(rows, key=lambda row: (row["partitions"], row["blocks"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = compare(args.input_dir)
    output = args.input_dir / "comparison.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"M41.2 comparison CSV: {output}")
    for row in rows:
        print(
            f"{row['blocks']:>7s} P={row['partitions']:>3d} "
            f"full={row['m41_1_forward_median_ms']:8.2f}ms "
            f"owned={row['m41_2_forward_median_ms']:8.2f}ms "
            f"speedup={row['owned_vs_full_forward_speedup']:.3f}x "
            f"Heun speedup={row['owned_vs_full_heun_speedup']:.3f}x "
            f"pass={row['passed']}"
        )
    if not all(row["passed"] for row in rows):
        raise SystemExit("M41.2 numerical gate failed for at least one layout")


if __name__ == "__main__":
    main()
