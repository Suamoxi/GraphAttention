"""Paired real-mesh M37 training-step benchmark for CSR PyTorch vs Triton backward.

Reuses the M43-B1 real HDF5 / Flow Matching diagnostic, including its probe-only
per-snapshot z-score and 10-layer checkpointed DiNAT-DiT. Each backend runs
in a fresh CUDA process with the same snapshot, graph, seed and optimizer.
This is NOT scientific training, and never saves model checkpoints.
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

from scripts.m43_b1_memory_gate import _revision, atomic_json, inspect_case


BACKENDS = ("torch_sparse", "torch_sparse_triton")
METRICS = ("forward_seconds", "backward_seconds", "optimizer_seconds", "step_seconds")


def measured_steps(report: dict, *, warmup: int, expected_steps: int) -> dict | None:
    readings = report.get("per_step", [])
    if report.get("status") != "ok" or len(readings) != expected_steps:
        return None
    stable = readings[warmup:]
    if not stable:
        return None
    return {
        "warmup_excluded": warmup,
        "measured_steps": len(stable),
        "median_seconds": {
            key: statistics.median(float(step[key]) for step in stable)
            for key in METRICS
        },
        "peak_allocated_gib": max(float(step["peak_allocated_gib"]) for step in stable),
        "peak_reserved_gib": max(float(step["peak_reserved_gib"]) for step in stable),
        "losses": [float(step["loss_probe_only"]) for step in readings],
    }


def run_pair(*, root: Path, case_spec: str, output_dir: Path,
             steps: int, seed: int) -> dict:
    if steps < 3:
        raise ValueError("at least three steps required: one JIT warmup and two measurements")
    case = inspect_case(root, case_spec)
    out = output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "protocol": "triton_full_m37_real_hdf5_v1",
        "status": "running",
        "scope": "performance feasibility, probe-only scaling; no scientific training",
        "code_revision": _revision(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "case": case,
        "steps_per_backend": steps,
        "warmup_steps_excluded": 1,
        "seed": seed,
        "model": {
            "architecture": "AlternatingDilatedGeometricDiT",
            "hidden_dim": 128, "num_heads": 4, "num_layers": 10,
            "activation_checkpointing": True,
        },
        "run_order": list(BACKENDS),
        "backends": {},
    }
    atomic_json(out / "preflight.json", summary)
    print("ARTIFACT_PREFLIGHT=" + str(out / "preflight.json"), flush=True)
    print("ARTIFACT_SUMMARY=" + str(out / "summary.json"), flush=True)

    for backend in BACKENDS:
        folder = out / backend
        folder.mkdir()
        report_path = folder / "report.json"
        process_log = folder / "process.log"
        print(f"ARTIFACT_{backend.upper()}_REPORT={report_path}", flush=True)
        print(f"ARTIFACT_{backend.upper()}_LOG={process_log}", flush=True)
        command = [
            sys.executable, "-u", "-m", "scripts.m43_b1_training_step",
            "--case", case["case"],
            "--snapshot", case["snapshot_file"],
            "--mesh", case["mesh_file"],
            "--side", str(case["resolution"]),
            "--output", str(report_path),
            "--steps", str(steps), "--seed", str(seed),
            "--hidden-dim", "128", "--layers", "10", "--heads", "4",
            "--activation-checkpointing",
            "--sparse-backend", backend,
        ]
        start = time.perf_counter()
        with process_log.open("w") as log:
            log.write(f"backend={backend} revision={summary['code_revision']}\n")
            log.flush()
            process = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        measured = measured_steps(report, warmup=1, expected_steps=steps)
        summary["backends"][backend] = {
            "status": report.get("status", "missing_report"),
            "exit_code": process.returncode,
            "process_wall_seconds": time.perf_counter() - start,
            "report": str(report_path),
            "process_log": str(process_log),
            "graph": report.get("graph"),
            "model_parameters": report.get("model_parameters"),
            "initial_weights_sha256": report.get("initial_weights_sha256"),
            "gpu": report.get("gpu"),
            "measurements": measured,
            "error": report.get("error"),
            "last_stage": report.get("last_stage"),
        }
        atomic_json(out / "progress.json", summary)
        print(
            f"[FULL CSR] {backend}: status={summary['backends'][backend]['status']}, "
            f"exit={process.returncode}, metrics={measured}",
            flush=True,
        )

    baseline = summary["backends"][BACKENDS[0]]
    candidate = summary["backends"][BACKENDS[1]]
    a, b = baseline["measurements"], candidate["measurements"]
    same_graph = baseline["graph"] is not None and baseline["graph"] == candidate["graph"]
    same_weights = (
        baseline["initial_weights_sha256"] is not None
        and baseline["initial_weights_sha256"] == candidate["initial_weights_sha256"]
    )
    if (
        a is not None and b is not None and same_graph and same_weights
        and baseline["exit_code"] == 0 and candidate["exit_code"] == 0
        and baseline["model_parameters"] == candidate["model_parameters"]
    ):
        summary["comparison"] = {
            "speedup": {
                key: a["median_seconds"][key] / b["median_seconds"][key]
                for key in METRICS
            },
            "max_absolute_probe_loss_difference": max(
                abs(x - y) for x, y in zip(a["losses"], b["losses"])
            ),
            "same_graph_and_inputs": True,
            "same_initial_weights": True,
            "same_model_parameter_count": True,
        }
        summary["status"] = "completed_not_accepted"
    else:
        summary["status"] = "failed_or_incomplete"
        summary["comparison"] = {
            "same_graph_and_inputs": same_graph,
            "same_initial_weights": same_weights,
        }
    atomic_json(out / "summary.json", summary)
    print(f"[FULL CSR] status={summary['status']} summary={out / 'summary.json'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/archivenfs/phlusim/HIT_DNS_COMP"))
    parser.add_argument("--case", default="Re54/Ma0.3:65")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_pair(
        root=args.root, case_spec=args.case, output_dir=args.output_dir,
        steps=args.steps, seed=args.seed,
    )
    if report["status"] != "completed_not_accepted":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
