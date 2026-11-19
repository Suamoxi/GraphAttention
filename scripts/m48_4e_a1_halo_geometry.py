"""M48.4e A1: CPU-only exact-halo geometry audit versus depth and core side.

No GPU or CFD field values are read. Every measurement calls the same
build_targeted_two_scale_patch geometry builder as M48 patch training.
Each configuration runs in an isolated, memory/time-bounded subprocess.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.m48_4e_gpu_capacity import (
    authentic_teacher_grids, build_geometry, positive_ints, write_json,
)

SCHEMA = "graph_attention.m48_4e_a1_halo_geometry.v1"
ORIGINS = ("seam", "interior")


def origin_for(fine_side: int, kind: str) -> tuple[int, int, int]:
    if kind == "seam":
        return (fine_side - 3, fine_side // 3 + 1, fine_side - 2)
    if kind == "interior":
        return (fine_side // 4,) * 3
    raise ValueError("unknown origin kind")


def geometry_metrics(report: dict, depth: int) -> dict:
    """Derive dimensionless coverage and edge/node work proxies from exact counts.

    Full-domain support counts are analytic for r=2 periodic cubic grids:
    degree 6 (one hop), 18 (exactly two hops), 27/8 nonzero trilinear
    interpolation edges per fine node. These are operation-count proxies,
    NOT measured GPU times or a complete layer-specific pruning analysis.
    """
    mode, side, p = report["mode"], report["fine_side"], report["owned_core_side"]
    if mode not in ("static", "dynamic") or depth < 1 or side not in (48, 80):
        raise ValueError("invalid geometry comparison")
    full_f, full_c = side**3, (side // 2)**3
    owned, fine, coarse = (report[k] for k in (
        "owned_nodes_per_patch", "fine_read_nodes", "coarse_read_nodes"))
    if not (owned == p**3 and owned <= fine <= full_f and 0 < coarse <= full_c):
        raise ValueError("invalid exact read-set sizes")
    fl, fd, cl, cd, cross = (report[k] for k in (
        "fine_local_edges", "fine_dilated_edges", "coarse_local_edges",
        "coarse_dilated_edges", "cross_scale_edges"))
    if min(fl, fd, cl, cd, cross) < 0:
        raise ValueError("negative attention edge count")
    if fl > 6 * fine or fd > 18 * fine or cl > 6 * coarse or cd > 18 * coarse:
        raise ValueError("attention support exceeds periodic Cartesian degrees")
    if cross > 8 * fine:
        raise ValueError("invalid trilinear cross-scale edge count")

    n_local, n_dilated = (depth + 1) // 2, depth // 2
    coarse_passes, cross_passes = (2, 2) if mode == "dynamic" else (1, 1)
    node_work = depth * (2 * fine + coarse_passes * coarse)
    full_node_work = depth * (2 * full_f + coarse_passes * full_c)
    edge_work = (2 * (n_local * fl + n_dilated * fd)
                 + coarse_passes * (n_local * cl + n_dilated * cd)
                 + cross_passes * cross)
    full_cross = (27 * full_f) // 8
    full_edge_work = (2 * (n_local * 6 * full_f + n_dilated * 18 * full_f)
                      + coarse_passes * (n_local * 6 * full_c + n_dilated * 18 * full_c)
                      + cross_passes * full_cross)
    return {
        "fine_total_nodes": full_f, "coarse_total_nodes": full_c,
        "fine_halo_nodes": fine - owned,
        "fine_read_fraction": fine / full_f,
        "coarse_read_fraction": coarse / full_c,
        "owned_fraction": owned / full_f,
        "fine_read_per_owned": fine / owned,
        "coarse_read_per_owned": coarse / owned,
        "idealized_full_coverage_fine_read_amplification": fine / owned,
        "idealized_full_coverage_coarse_read_amplification": coarse * full_f / (owned * full_c),
        "node_work_proxy_per_patch": node_work,
        "edge_work_proxy_per_patch": edge_work,
        "full_grid_node_work_proxy": full_node_work,
        "full_grid_edge_work_proxy": full_edge_work,
        "idealized_full_coverage_node_work_ratio": node_work * full_f / (owned * full_node_work),
        "idealized_full_coverage_edge_work_ratio": edge_work * full_f / (owned * full_edge_work),
        "fine_read_covers_entire_domain": fine == full_f,
        "coarse_read_covers_entire_domain": coarse == full_c,
    }


def summarize(rows: list[dict], expected: int) -> dict:
    """Fail closed: all requested config/origin pairs must have exact receipts."""
    ok = [r for r in rows if r.get("status") == "ok"]
    errors = [r for r in rows if r.get("status") != "ok"]
    by_configuration: dict[tuple, list[dict]] = {}
    for row in ok:
        key = (row["mode"], row["fine_side"], row["depth"], row["owned_core_side"])
        by_configuration.setdefault(key, []).append(row)
    aggregates = []
    for (mode, side, depth, core), group in sorted(by_configuration.items()):
        ag = {"mode": mode, "fine_side": side, "depth": depth,
              "owned_core_side": core, "origin_count": len(group),
              "origin_kinds": sorted(r["origin_kind"] for r in group)}
        for col in ("fine_read_fraction", "coarse_read_fraction", "fine_read_per_owned",
                    "idealized_full_coverage_node_work_ratio",
                    "idealized_full_coverage_edge_work_ratio", "fine_read_nodes",
                    "coarse_read_nodes", "edge_work_proxy_per_patch",
                    "single_graph_tensor_bytes", "geometry_build_seconds"):
            ag[f"{col}_min"] = min(r[col] for r in group)
            ag[f"{col}_max"] = max(r[col] for r in group)
        aggregates.append(ag)
    return {"status": "PASS" if len(ok) == expected and not errors else "ERROR",
            "successful_trials": len(ok), "failed_trials": len(errors),
            "requested_trials": expected, "aggregates": aggregates}


def worker(args) -> None:
    """No field access. Constrain CPU address space before building geometry."""
    if args.max_worker_virtual_gib:
        import resource
        limit = int(args.max_worker_virtual_gib * 2**30)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    try:
        side, depth, core = args.fine_side, args.depth, args.core_side
        geom, raw = build_geometry(
            side, core, depth, args.mode, args.periodic_length,
            origin=origin_for(side, args.origin_kind),
        )
        metrics = geometry_metrics(raw, depth)
        row = {**raw, **metrics, "depth": depth, "origin_kind": args.origin_kind,
               "status": "ok", "no_gpu_or_physical_fields": True}
        del geom
    except (MemoryError, RuntimeError, ValueError) as exc:
        row = {"status": "geometry_error", "error": f"{type(exc).__name__}: {exc}"}
    write_json(args.worker_output, row)
    if row["status"] != "ok":
        raise RuntimeError(row["error"])


def run(args) -> dict:
    depths, cores, sides = (positive_ints(args.depths),
                            positive_ints(args.core_sides), positive_ints(args.fine_sides))
    kinds = tuple(x.strip() for x in args.origins.split(","))
    if (args.mode not in ("static", "dynamic") or not kinds or len(set(kinds)) != len(kinds)
            or any(k not in ORIGINS for k in kinds) or any(n not in (48, 80) for n in sides)
            or any(p > min(sides) for p in cores) or args.worker_timeout_seconds < 1
            or args.max_worker_virtual_gib <= 0):
        raise ValueError("invalid A1 halo sweep configuration")
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite A1 audit receipts")
    grids, provenance = authentic_teacher_grids(args.ms_sr_root, args.normalization_file)
    args.output_dir.mkdir(parents=True)
    count = len(depths) * len(cores) * len(sides) * len(kinds)
    meta = {"schema": SCHEMA, "status": "RUNNING", "mode": args.mode,
            "no_gpu_or_physical_fields": True,
            "provenance": provenance, "teacher_grids": grids,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "requested": {"depths_per_subnetwork": depths, "fine_sides": sides,
                          "core_sides": cores, "origins": kinds,
                          "worker_timeout_seconds": args.worker_timeout_seconds,
                          "max_worker_virtual_gib": args.max_worker_virtual_gib},
            "requested_trials": count, "trials": [], "aggregates": []}
    write_json(args.output_dir / "summary.json", meta)
    for side in sides:
        for depth in depths:
            for core in cores:
                for kind in kinds:
                    trial = f"{args.mode}_n{side}_d{depth}_p{core}_{kind}"
                    receipt = args.output_dir / "trials" / (trial + ".json")
                    log = args.output_dir / "trials" / (trial + ".log")
                    receipt.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [sys.executable, "-m", "scripts.m48_4e_a1_halo_geometry",
                           "--worker", "--mode", args.mode,
                           "--fine-side", str(side), "--core-side", str(core),
                           "--depth", str(depth), "--origin-kind", kind,
                           "--periodic-length", repr(grids[side]["periodic_length"]),
                           "--max-worker-virtual-gib", str(args.max_worker_virtual_gib),
                           "--worker-output", str(receipt)]
                    t0 = time.perf_counter()
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True,
                                                timeout=args.worker_timeout_seconds, check=False)
                        log.write_text(result.stdout + "\n=== stderr ===\n" + result.stderr)
                        if receipt.is_file():
                            row = json.loads(receipt.read_text())
                            if result.returncode != 0 and row.get("status") == "ok":
                                row = {"status": "unexpected_worker_exit", "exit_code": result.returncode}
                        else:
                            row = {"status": "missing_worker_receipt", "exit_code": result.returncode}
                    except subprocess.TimeoutExpired:
                        row = {"status": "worker_timeout", "limit_seconds": args.worker_timeout_seconds}
                        log.write_text("Worker exceeded the A1 time limit.\n")
                    row.update({"trial_id": trial, "mode": args.mode, "fine_side": side,
                                "depth": depth, "owned_core_side": core, "origin_kind": kind,
                                "worker_wall_seconds": time.perf_counter() - t0})
                    write_json(receipt, row)
                    meta["trials"].append(row)
                    write_json(args.output_dir / "summary.json", meta)
                    print(f"{trial}: {row['status']}"
                          + (f" fine={row['fine_read_nodes']}/{side**3}, "
                             f"coarse={row['coarse_read_nodes']}/{(side//2)**3}, "
                             f"amplification={row['fine_read_per_owned']:.2f}x"
                             if row["status"] == "ok" else ""), flush=True)
    aggregate = summarize(meta["trials"], count)
    meta.update(aggregate)
    write_json(args.output_dir / "summary.json", meta)
    with (args.output_dir / "trials.csv").open("w", newline="") as f:
        cols = ["trial_id", "status", "mode", "fine_side", "depth", "owned_core_side",
                "origin_kind", "fine_read_nodes", "coarse_read_nodes", "owned_nodes_per_patch",
                "fine_read_fraction", "coarse_read_fraction", "fine_read_per_owned",
                "fine_local_edges", "fine_dilated_edges", "coarse_local_edges",
                "coarse_dilated_edges", "cross_scale_edges", "edge_work_proxy_per_patch",
                "idealized_full_coverage_node_work_ratio",
                "idealized_full_coverage_edge_work_ratio", "single_graph_tensor_bytes",
                "geometry_build_seconds", "worker_wall_seconds", "error"]
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(meta["trials"])
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("static", "dynamic"))
    parser.add_argument("--ms-sr-root", type=Path, default=Path("/scratch/ai/MS_SR"))
    parser.add_argument("--normalization-file", type=Path,
                        default=Path("configs/m48/teacher_44_w1p25_l1_q4_q8.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fine-sides", default="48,80")
    parser.add_argument("--depths", default="2,4,6,8")
    parser.add_argument("--core-sides", default="12,16,20,24")
    parser.add_argument("--origins", default="seam,interior")
    parser.add_argument("--worker-timeout-seconds", type=int, default=300)
    parser.add_argument("--max-worker-virtual-gib", type=float, default=48.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fine-side", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--core-side", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--depth", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--origin-kind", choices=ORIGINS, help=argparse.SUPPRESS)
    parser.add_argument("--periodic-length", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if (args.worker_output is None or args.fine_side is None or args.core_side is None
                or args.depth is None or args.origin_kind is None or args.periodic_length is None
                or not math.isfinite(args.periodic_length) or args.periodic_length <= 0):
            parser.error("worker requires explicit authenticated grid, depth and receipt")
        worker(args)
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required")
        result = run(args)
        print(json.dumps({"status": result["status"], "successful_trials": result["successful_trials"],
                          "requested_trials": result["requested_trials"],
                          "output": str(args.output_dir)}, indent=2), flush=True)
        if result["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
