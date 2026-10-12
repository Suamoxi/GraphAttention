"""Strict matched-pair GH200 summary: M41.2 GPU-resident vs M41.3 CPU-streamed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _contract(report: dict) -> tuple:
    generation = report["generation"]
    return (
        report["sample_id"],
        report["num_nodes"],
        report["backend"],
        report["precision"],
        json.dumps(report["artifact_sha256"], sort_keys=True),
        tuple(report["blocks"]),
        report["num_partitions"],
        generation["steps"],
        generation["seed"],
    )


def compare(directory: Path) -> list[dict]:
    streamed = sorted(directory.glob("streamed_blocks_*.json"))
    reference = sorted(directory.glob("owned_blocks_*.json"))
    if not streamed or len(streamed) != len(reference):
        raise ValueError("M41.3 requires paired streamed_blocks_*.json and owned_blocks_*.json")
    rows = []
    common_sample = None
    for stream_path in streamed:
        suffix = stream_path.name.removeprefix("streamed_")
        owned_path = directory / ("owned_" + suffix)
        if not owned_path.exists():
            raise ValueError(f"missing M41.2 paired report for {suffix}")
        old, new = json.loads(owned_path.read_text()), json.loads(stream_path.read_text())
        if _contract(old) != _contract(new):
            raise ValueError(f"checkpoint, topology, or sampler mismatch for {suffix}")
        if old.get("executor") != "owned_query" or new.get("executor") != "streamed":
            raise ValueError("unexpected M41.2 or M41.3 executor label")
        if new.get("hidden_storage") != "cpu_double_buffer":
            raise ValueError("M41.3 report is missing host-streaming evidence")
        sample = _contract(new)[:5] + _contract(new)[-2:]
        if common_sample is None:
            common_sample = sample
        elif common_sample != sample:
            raise ValueError("M41.3 layouts are not from the same checkpoint and sample")
        old_forward = old["partitioned"]["median_ms"]
        new_forward = new["partitioned"]["median_ms"]
        old_heun = old["generation"]["partitioned"]["median_ms"]
        new_heun = new["generation"]["partitioned"]["median_ms"]
        rows.append({
            "blocks": "x".join(map(str, new["blocks"])),
            "partitions": new["num_partitions"],
            "read_node_ratio": new["nodewise_work_ratio"],
            "max_partition_read_nodes": new["max_partition_read_nodes"],
            "m37_forward_median_ms": new["original"]["median_ms"],
            "m41_2_forward_median_ms": old_forward,
            "m41_3_forward_median_ms": new_forward,
            "m41_3_vs_m41_2_speedup": old_forward / new_forward,
            "m41_3_vs_m37_forward_ratio": new_forward / new["original"]["median_ms"],
            "m41_2_heun_median_ms": old_heun,
            "m41_3_heun_median_ms": new_heun,
            "m41_3_vs_m41_2_heun_speedup": old_heun / new_heun,
            "m41_2_peak_allocated_gib": old["partitioned"]["peak_allocated_gib"],
            "m41_3_peak_allocated_gib": new["partitioned"]["peak_allocated_gib"],
            "m41_3_host_hidden_two_buffer_gib": new["host_hidden_two_buffer_gib"],
            "m41_3_max_layer_abs": max(
                error["max_abs"] if error["max_abs"] is not None else float("inf")
                for error in new["errors"]["layers"]
            ),
            "m41_3_output_max_abs": new["errors"]["output"]["max_abs"],
            "m41_3_heun_max_abs": new["errors"]["heun"]["max_abs"],
            "m41_2_passed": old["passed"],
            "m41_3_passed": new["passed"],
            "passed": old["passed"] and new["passed"],
        })
    return sorted(rows, key=lambda row: (row["partitions"], row["blocks"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = compare(args.input_dir)
    output = args.input_dir / "comparison.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"M41.3 comparison: {output}")
    for row in rows:
        print(
            f"{row['blocks']:>7s}: P={row['partitions']:>3d}, "
            f"owned={row['m41_2_forward_median_ms']:8.2f}ms, "
            f"streamed={row['m41_3_forward_median_ms']:8.2f}ms, "
            f"ratio={row['m41_3_vs_m41_2_speedup']:.3f}x, "
            f"passed={row['passed']}"
        )
    if not all(row["passed"] for row in rows):
        raise SystemExit("M41.3 numerical gate failed; do not interpret speedups as valid")


if __name__ == "__main__":
    main()
