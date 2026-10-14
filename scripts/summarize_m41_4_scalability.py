"""Aggregate M41.4 GH200 repeated/sweep trials and compare saved outputs.

No performance claims are made from incomplete runs. All input reports are
checked for consistent checkpoint, seed and graph identity. GPU memory is
PyTorch allocator peak; CPU peak is Linux process RSS, not total node RAM.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np


def _output_difference(reference: Path, candidate: Path, *, atol: float, rtol: float,
                       chunk_rows: int = 65536) -> dict:
    """Compare all saved output elements in bounded chunks; not just means/std."""
    a = np.load(reference, mmap_mode="r")
    b = np.load(candidate, mmap_mode="r")
    if a.shape != b.shape or a.dtype != b.dtype or a.ndim != 2:
        raise ValueError("saved forward outputs differ in shape or dtype")
    squared_error = squared_reference = 0.0
    largest = 0.0
    violations = 0
    for left in range(0, len(a), chunk_rows):
        right = min(len(a), left + chunk_rows)
        ref = np.asarray(a[left:right], dtype=np.float64)
        got = np.asarray(b[left:right], dtype=np.float64)
        if not np.isfinite(ref).all() or not np.isfinite(got).all():
            raise ValueError("nonfinite saved full-resolution forward output")
        delta = got - ref
        largest = max(largest, float(np.max(np.abs(delta))))
        squared_error += float(np.sum(delta * delta))
        squared_reference += float(np.sum(ref * ref))
        violations += int(np.count_nonzero(np.abs(delta) > atol + rtol * np.abs(ref)))
    return {
        "reference": str(reference), "candidate": str(candidate),
        "max_abs": largest,
        "rms": math.sqrt(squared_error / a.size),
        "relative_l2": math.sqrt(squared_error / max(squared_reference, 1e-60)),
        "violations": violations, "num_elements": int(a.size),
        "atol": atol, "rtol": rtol, "passed": violations == 0,
    }


def _stats(values: list[float]) -> dict:
    if not values:
        raise ValueError("no successful forward measurements")
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values), "max": max(values),
    }


def summarize(root: Path, *, require_layouts: tuple[int, ...] = (),
              require_repeated: bool = False,
              atol: float = 2e-4, rtol: float = 2e-4) -> dict:
    root = Path(root).resolve(strict=True)
    manifests = sorted((root / "benchmarks").rglob("benchmark.json"))
    if not manifests:
        raise FileNotFoundError("no completed or partial benchmark manifests")
    all_runs = []
    groups = []
    incomplete = []
    seeds = set()
    checkpoints = set()
    graphs = set()
    for file in manifests:
        m = json.loads(file.read_text())
        signature = m["signature"]
        series = file.parent.parent.name
        layout = int(signature["layout"])
        seeds.add(signature["seed"])
        graphs.add(signature["base_prepared_dir"])
        if m.get("checkpoint_sha256"):
            checkpoints.add(m["checkpoint_sha256"])
        if m.get("status") != "complete":
            incomplete.append(str(file))
            continue
        for mode in signature["strategies"]:
            reports = []
            for repeat in range(1, int(signature["repeats"]) + 1):
                name = f"{mode}_r{repeat:02d}"
                record = m["runs"].get(name)
                if not record:
                    incomplete.append(str(file) + ":" + name)
                    continue
                report = json.loads(Path(record["forward_file"]).read_text())
                if (report["status"] != "full_highres_forward_passed"
                        or report["partition_count"] != layout
                        or report["checkpoint_sha256"] != m["checkpoint_sha256"]
                        or report.get("seed") != signature["seed"]):
                    raise ValueError("benchmark report signature mismatch: " + name)
                reports.append(report)
                all_runs.append({"series": series, "layout": layout,
                                 "mode": mode, "repeat": repeat, **report})
            if reports:
                get = lambda key: [float(r[key]) for r in reports]
                rss = [float(r["host_memory_after_forward"]["rss_peak_gib"])
                       for r in reports]
                prepared = Path(signature["prepared_dir"])
                parts = json.loads((prepared / "partitions" / "partitions.json").read_text())
                owned = sum(v["owned_nodes"] for v in parts["parts"])
                read = sum(v["read_nodes"] for v in parts["parts"])
                if owned != reports[0]["nodes"]:
                    raise ValueError("partition ownership is not a complete node cover")
                groups.append({
                    "series": series, "partitions": layout,
                    "blocks": "x".join(map(str, signature["blocks"])),
                    "strategy": mode, "runs": len(reports),
                    "distinct_output_sha256": len({r.get("output_sha256") for r in reports}),
                    "forward_seconds": _stats(get("forward_seconds")),
                    "plan_seconds": _stats(get("plan_load_seconds")),
                    "peak_cpu_rss_gib": _stats(rss),
                    "peak_gpu_allocated_gib": _stats(get("gpu_peak_allocated_gib")),
                    "peak_gpu_reserved_gib": _stats(get("gpu_peak_reserved_gib")),
                    "max_owned_nodes": max(v["owned_nodes"] for v in parts["parts"]),
                    "max_read_nodes": max(v["read_nodes"] for v in parts["parts"]),
                    "total_halo_nodes_across_parts": read - owned,
                    "halo_read_amplification": read / owned,
                })
    if len(seeds) != 1 or len(graphs) != 1 or len(checkpoints) != 1:
        raise ValueError("experiments use inconsistent seed, graph source or checkpoint")
    available_sweep = {g["partitions"] for g in groups
                       if g["series"] == "sweep" and g["strategy"] == "lazy"}
    for p in require_layouts:
        if p not in available_sweep:
            incomplete.append(f"missing completed sweep layout {p}")
    if require_repeated and not ({"preload", "lazy"} <= {
        g["strategy"] for g in groups if g["series"] == "repeated" and g["partitions"] == 64
    }):
        incomplete.append("missing completed 64-partition paired repetition")
    ref = root / "benchmarks" / "repeated" / "layout_064" / "lazy_r01" / "forward.npy"
    if not ref.is_file():
        ref = root / "benchmarks" / "sweep" / "layout_064" / "lazy_r01" / "forward.npy"
    comparisons = []
    if ref.is_file():
        for file in sorted((root / "benchmarks").rglob("forward.npy")):
            comparisons.append(_output_difference(ref, file, atol=atol, rtol=rtol))
    else:
        incomplete.append("missing full-output 64-partition lazy reference")
    groups.sort(key=lambda x: (x["series"], x["partitions"], x["strategy"]))
    report = {
        "status": "complete" if not incomplete else "incomplete",
        "num_forward_runs": len(all_runs),
        "checkpoint_sha256": next(iter(checkpoints)),
        "seed": next(iter(seeds)),
        "groups": groups,
        "cross_layout_output_comparisons": comparisons,
        "all_saved_outputs_within_tolerance": bool(comparisons) and all(
            c["passed"] for c in comparisons
        ),
        "missing_or_incomplete": incomplete,
        "interpretation": (
            "Same 257^3 graph and checkpoint. Only partition count and lazy/preload "
            "topology loading strategy change. CPU memory is peak process RSS; "
            "GPU memory is PyTorch allocator metrics. Comparisons are partition-to-"
            "partition deterministic output consistency, not unpartitioned "
            "257^3 M37 parity or physical accuracy."
        ),
    }
    fields = ["series", "partitions", "blocks", "strategy", "runs",
              "forward_median_s", "forward_mean_s", "forward_stddev_s",
              "peak_cpu_rss_median_gib", "peak_cpu_rss_max_gib",
              "peak_gpu_allocated_median_gib", "peak_gpu_reserved_median_gib",
              "plan_median_s", "max_owned_nodes", "max_read_nodes",
              "halo_read_amplification", "distinct_output_sha256"]
    rows = []
    for g in groups:
        rows.append({
            "series": g["series"], "partitions": g["partitions"],
            "blocks": g["blocks"], "strategy": g["strategy"],
            "runs": g["runs"],
            "forward_median_s": g["forward_seconds"]["median"],
            "forward_mean_s": g["forward_seconds"]["mean"],
            "forward_stddev_s": g["forward_seconds"]["stddev"],
            "peak_cpu_rss_median_gib": g["peak_cpu_rss_gib"]["median"],
            "peak_cpu_rss_max_gib": g["peak_cpu_rss_gib"]["max"],
            "peak_gpu_allocated_median_gib": g["peak_gpu_allocated_gib"]["median"],
            "peak_gpu_reserved_median_gib": g["peak_gpu_reserved_gib"]["median"],
            "plan_median_s": g["plan_seconds"]["median"],
            "max_owned_nodes": g["max_owned_nodes"],
            "max_read_nodes": g["max_read_nodes"],
            "halo_read_amplification": g["halo_read_amplification"],
            "distinct_output_sha256": g["distinct_output_sha256"],
        })
    (root / "scaling_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (root / "scaling_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "# M41.4 257-cube computational scalability",
        "", f"Status: {report['status']}; completed trials: {len(all_runs)}.", "",
        "| Series | Partitions | Strategy | Repeats | Forward median (s) | CPU peak median (GiB) | GPU allocated median (GiB) | Halo read factor |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        markdown.append(
            f"| {r['series']} | {r['partitions']} | {r['strategy']} | {r['runs']} | "
            f"{r['forward_median_s']:.2f} | {r['peak_cpu_rss_median_gib']:.2f} | "
            f"{r['peak_gpu_allocated_median_gib']:.2f} | {r['halo_read_amplification']:.4f} |"
        )
    markdown.extend(["", "## Saved full-output comparison (relative to 64-partition lazy)", ""])
    for c in comparisons:
        markdown.append(f"- {Path(c['candidate']).parent.relative_to(root)}: "
                        f"max_abs={c['max_abs']:.6g}, relative_l2={c['relative_l2']:.6g}, "
                        f"violations={c['violations']}/{c['num_elements']}")
    markdown += ["", "Comparisons are between partitioning strategies, not an unpartitioned M37 reference."]
    if incomplete:
        markdown += ["", "Missing/incomplete trials:"] + ["- " + x for x in incomplete]
    (root / "scaling_summary.md").write_text("\n".join(markdown) + "\n")
    print("\n".join(markdown), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--require-layouts", nargs="*", type=int, default=[])
    p.add_argument("--require-repeated", action="store_true")
    p.add_argument("--atol", type=float, default=2e-4)
    p.add_argument("--rtol", type=float, default=2e-4)
    args = p.parse_args()
    report = summarize(args.root, require_layouts=tuple(args.require_layouts),
                       require_repeated=args.require_repeated,
                       atol=args.atol, rtol=args.rtol)
    if report["status"] != "complete" or not report["all_saved_outputs_within_tolerance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
