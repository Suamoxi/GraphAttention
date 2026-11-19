"""Three-arm real-HDF5 M37 segmented-conditioning GPU pilot.

Tests two different CFD graphs in one optimizer update, comparing packed
advanced indexing, opt-in expand/cat segmented broadcast, and sequential
single-graph gradient accumulation. The exact same packed Flow Matching
noise, per-graph time and equal-sample loss are used in all three arms.
A fresh subprocess per arm isolates CUDA caches; a CPU audit compares
every first-step gradient and post-AdamW parameter for parity.

Probe-only state normalization; no M47 scientific trainer/config change.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from scripts.benchmark_m47_segmented_worker import MODES
from scripts.m43_b1_memory_gate import _revision, atomic_json, inspect_case

METRICS = ("forward_seconds", "backward_seconds", "optimizer_seconds", "step_seconds")
BACKEND = "torch_sparse_triton"


def select_cases(root: Path, scenario: str) -> list[dict]:
    a = inspect_case(root, "Re54/Ma0.3:65")
    if scenario == "two_snapshots_65":
        snapshots = sorted(Path(a["snapshot_file"]).parent.glob("*.h5"))
        if len(snapshots) < 2:
            raise ValueError("need at least two real 65-cube HDF5 snapshots")
        middle = len(snapshots) // 2
        other = middle + 1 if middle + 1 < len(snapshots) else middle - 1
        b = dict(a)
        b["snapshot_file"] = str(snapshots[other].resolve(strict=True))
        b["snapshot_selection"] = "adjacent_distinct_snapshot"
        if b["snapshot_file"] == a["snapshot_file"]:
            raise RuntimeError("two snapshots must be distinct")
    elif scenario == "mixed_65_129":
        b = inspect_case(root, "Re94/Ma0.3:129")
    else:
        raise ValueError("unknown scenario")
    return [a, b]


def medians(report: dict, steps: int) -> dict | None:
    records = report.get("per_step", [])
    if report.get("status") != "ok" or len(records) != steps:
        return None
    stable = records[1:]
    return {
        "warmup_discarded": 1,
        "measured_steps": len(stable),
        "median_seconds": {
            key: statistics.median(float(x[key]) for x in stable)
            for key in METRICS
        },
        "peak_allocated_gib": max(float(x["peak_allocated_gib"]) for x in stable),
        "peak_reserved_gib": max(float(x["peak_reserved_gib"]) for x in stable),
        "losses": [float(x["loss"]) for x in records],
    }


def audit_parity(reference_path: Path, other_path: Path) -> dict:
    """Compare all named first-step parameter gradients and updated weights."""
    baseline = torch.load(reference_path, map_location="cpu", weights_only=True)
    candidate = torch.load(other_path, map_location="cpu", weights_only=True)
    categories = {
        "gradients": (5e-4, 5e-3),
        "post_update_parameters": (3e-6, 2e-4),
    }
    result = {"status": "pass", "categories": {}}
    for key, (atol, rtol) in categories.items():
        reference = baseline[key]
        testing = candidate[key]
        if set(reference) != set(testing):
            return {"status": "fail", "error": f"parameter names differ in {key}"}
        mismatch = 0
        worst_abs = 0.0
        worst_name = None
        total_elements = 0
        for name, expected in reference.items():
            actual = testing[name]
            if expected is None or actual is None:
                if expected is not None or actual is not None:
                    mismatch += 1
                    worst_name = name
                continue
            if expected.shape != actual.shape:
                return {"status": "fail", "error": f"{key}:{name} shape differs"}
            if not bool(torch.isfinite(expected).all()) or not bool(torch.isfinite(actual).all()):
                return {"status": "fail", "error": f"{key}:{name} nonfinite data"}
            diff = (actual - expected).abs()
            bad = diff > (atol + rtol * expected.abs())
            mismatch += int(bad.sum())
            total_elements += expected.numel()
            deviation = float(diff.max()) if diff.numel() else 0.0
            if deviation > worst_abs:
                worst_abs, worst_name = deviation, name
        result["categories"][key] = {
            "atol": atol, "rtol": rtol,
            "mismatched_elements": mismatch, "elements_compared": total_elements,
            "max_absolute_difference": worst_abs,
            "worst_parameter": worst_name,
        }
        if mismatch:
            result["status"] = "fail"
    return result


def run(*, root: Path, scenario: str, out: Path, steps: int, seed: int,
        backend: str) -> dict:
    if steps < 3:
        raise ValueError("at least three steps required")
    cases = select_cases(root, scenario)
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "protocol": "m47_segmented_multigraph_v1",
        "status": "preflight",
        "revision": _revision(),
        "scenario": scenario, "cases": cases,
        "steps": steps, "seed": seed, "backend": backend,
        "loss_weighting": "equal physical graphs; mean channel MSE within each graph",
        "scope": "computational GPU benchmark; probe-only per-graph normalization",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "run_order": list(MODES),
    }
    atomic_json(out / "manifest.json", manifest)
    print(f"ARTIFACT_MANIFEST={out / 'manifest.json'}", flush=True)
    print(f"ARTIFACT_SUMMARY={out / 'summary.json'}", flush=True)
    summary = dict(manifest, status="running", modes={})

    for mode in MODES:
        folder = out / mode
        folder.mkdir()
        report_path = folder / "report.json"
        audit_path = folder / "audit.pt"
        log_path = folder / "process.log"
        print(f"ARTIFACT_{mode.upper()}_REPORT={report_path}", flush=True)
        print(f"ARTIFACT_{mode.upper()}_AUDIT={audit_path}", flush=True)
        print(f"ARTIFACT_{mode.upper()}_LOG={log_path}", flush=True)
        cmd = [
            sys.executable, "-u", "-m", "scripts.benchmark_m47_segmented_worker",
            "--manifest", str(out / "manifest.json"),
            "--mode", mode, "--report", str(report_path),
            "--audit", str(audit_path), "--steps", str(steps),
            "--seed", str(seed), "--backend", backend,
        ]
        begin = time.perf_counter()
        with log_path.open("w") as stream:
            stream.write(f"mode={mode}; revision={manifest['revision']}\n")
            stream.flush()
            code = subprocess.run(
                cmd, stdout=stream, stderr=subprocess.STDOUT, check=False,
            ).returncode
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        item = {
            "status": report.get("status", "missing_report"),
            "exit_code": code,
            "process_wall_seconds": time.perf_counter() - begin,
            "report": str(report_path), "log": str(log_path),
            "audit": str(audit_path) if audit_path.is_file() else None,
            "graph": report.get("graph"),
            "normalized_state_sha256": report.get("normalized_state_sha256"),
            "initial_weights_sha256": report.get("initial_weights_sha256"),
            "parameter_count": report.get("parameter_count"),
            "measurements": medians(report, steps),
            "error": report.get("error"),
            "last_stage": report.get("last_stage"),
        }
        summary["modes"][mode] = item
        atomic_json(out / "progress.json", summary)
        print(f"[SEGMENTED] mode={mode} status={item['status']} "
              f"exit={code} metrics={item['measurements']}", flush=True)

    reference = summary["modes"]["sequential_accum"]
    result = {}
    accepted = all(
        item["status"] == "ok" and item["exit_code"] == 0
        and item["measurements"] is not None and item["audit"] is not None
        for item in summary["modes"].values()
    )
    if accepted:
        for mode in ("packed_index", "packed_segmented"):
            candidate = summary["modes"][mode]
            matching = all(
                candidate[key] is not None and candidate[key] == reference[key]
                for key in (
                    "graph", "normalized_state_sha256",
                    "initial_weights_sha256", "parameter_count",
                )
            )
            losses_a = reference["measurements"]["losses"]
            losses_b = candidate["measurements"]["losses"]
            max_loss = max(abs(a - b) for a, b in zip(losses_a, losses_b))
            audit = audit_parity(Path(reference["audit"]), Path(candidate["audit"]))
            result[mode] = {
                "matching_input_topologies_weights": matching,
                "max_absolute_loss_difference": max_loss,
                "first_step_grads_and_adamw_audit": audit,
                "sequential_to_packed_step_speedup": (
                    reference["measurements"]["median_seconds"]["step_seconds"]
                    / candidate["measurements"]["median_seconds"]["step_seconds"]
                ),
                "peak_allocated_gib": candidate["measurements"]["peak_allocated_gib"],
            }
            if not (matching and max_loss <= 1e-4 and audit["status"] == "pass"):
                accepted = False
        a = summary["modes"]["packed_index"]["measurements"]["median_seconds"]["step_seconds"]
        b = summary["modes"]["packed_segmented"]["measurements"]["median_seconds"]["step_seconds"]
        result["packed_index_to_segmented_speedup"] = a / b
    summary["comparison"] = result
    summary["status"] = "completed_not_accepted" if accepted else "failed_or_review_required"
    atomic_json(out / "summary.json", summary)
    print(f"[SEGMENTED] status={summary['status']}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/archivenfs/phlusim/HIT_DNS_COMP"))
    parser.add_argument("--scenario", choices=("two_snapshots_65", "mixed_65_129"),
                        default="two_snapshots_65")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("torch_sparse", "torch_sparse_triton"),
                        default=BACKEND)
    args = parser.parse_args()
    result = run(
        root=args.root, scenario=args.scenario, out=args.out,
        steps=args.steps, seed=args.seed, backend=args.backend,
    )
    if result["status"] != "completed_not_accepted":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
