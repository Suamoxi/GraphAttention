"""Paired CUDA operator profile of one full real-HDF5 M37 optimizer step.

Two backends in fresh CUDA processes, one unprofiled warmup followed by
one separately profiled full AdamW update. The original scientific
M47.4 trainer and numerical configurations remain untouched.
Profiling changes timing: do not use profiler durations for speedups.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.m43_b1_memory_gate import _revision, atomic_json, inspect_case


BACKENDS = ("torch_sparse", "torch_sparse_triton")


def run_profiles(*, root: Path, case_spec: str, output_dir: Path, seed: int = 42) -> dict:
    case = inspect_case(root, case_spec)
    out = output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "protocol": "triton_full_m37_operator_profile_v1",
        "status": "running",
        "scope": "diagnostic GPU operator attribution, no comparable wall timings or scientific training",
        "revision": _revision(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "case": case,
        "profile_step": 2,
        "seed": seed,
        "model": {
            "architecture": "AlternatingDilatedGeometricDiT",
            "layers": 10, "hidden_dim": 128, "heads": 4,
            "checkpointed": True,
        },
        "backends": {},
    }
    atomic_json(out / "preflight.json", result)
    print(f"ARTIFACT_PREFLIGHT={out / 'preflight.json'}", flush=True)
    print(f"ARTIFACT_SUMMARY={out / 'summary.json'}", flush=True)
    for backend in BACKENDS:
        folder = out / backend
        folder.mkdir()
        report_path = folder / "report.json"
        profile_path = folder / "operator_profile.json"
        log_path = folder / "process.log"
        print(f"ARTIFACT_{backend.upper()}_REPORT={report_path}", flush=True)
        print(f"ARTIFACT_{backend.upper()}_PROFILE={profile_path}", flush=True)
        print(f"ARTIFACT_{backend.upper()}_LOG={log_path}", flush=True)
        args = [
            sys.executable, "-u", "-m", "scripts.m43_b1_training_step",
            "--case", case["case"],
            "--snapshot", case["snapshot_file"],
            "--mesh", case["mesh_file"],
            "--side", str(case["resolution"]),
            "--steps", "2", "--seed", str(seed),
            "--hidden-dim", "128", "--layers", "10", "--heads", "4",
            "--activation-checkpointing", "--sparse-backend", backend,
            "--operator-profile-step", "2",
            "--operator-profile-output", str(profile_path),
            "--output", str(report_path),
        ]
        with log_path.open("w") as stream:
            stream.write(f"backend={backend} revision={result['revision']}\n")
            stream.flush()
            proc = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, check=False)
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        profile = json.loads(profile_path.read_text()) if profile_path.is_file() else {}
        outcome = {
            "status": report.get("status", "no_report"),
            "exit_code": proc.returncode,
            "report": str(report_path),
            "operator_profile": str(profile_path),
            "process_log": str(log_path),
            "graph": report.get("graph"),
            "initial_weights_sha256": report.get("initial_weights_sha256"),
            "model_parameters": report.get("model_parameters"),
            "losses": [step["loss_probe_only"] for step in report.get("per_step", [])],
            "operator_profile_status": profile.get("status", "missing"),
            "profiled_step": profile.get("step"),
            "total_self_device_us": profile.get("total_self_device_us"),
            "top_cuda_ops": [
                entry for entry in profile.get("top_operators", [])
                if entry["self_device_us"] > 0
            ][:20],
            "error": report.get("error"),
        }
        result["backends"][backend] = outcome
        atomic_json(out / "progress.json", result)
        print(
            f"[FULL CSR PROFILE] {backend}: status={outcome['status']} "
            f"profile={outcome['operator_profile_status']} "
            f"operators={len(profile.get('top_operators', []))}",
            flush=True,
        )
    old, new = (result["backends"][key] for key in BACKENDS)
    matched = (
        old["graph"] is not None and old["graph"] == new["graph"]
        and old["initial_weights_sha256"] is not None
        and old["initial_weights_sha256"] == new["initial_weights_sha256"]
        and old["model_parameters"] == new["model_parameters"]
    )
    completed = all(
        item["exit_code"] == 0 and item["status"] == "ok"
        and item["operator_profile_status"] == "complete"
        and item["profiled_step"] == 2
        and (item["total_self_device_us"] or 0) > 0
        for item in (old, new)
    )
    result["comparison"] = {
        "same_graph_and_inputs_and_initial_weights": matched,
        "max_absolute_probe_loss_difference": (
            max(abs(a-b) for a,b in zip(old["losses"],new["losses"]))
            if len(old["losses"]) == len(new["losses"]) == 2 else None
        ),
        "timing_is_diagnostic_only": True,
    }
    result["status"] = "completed_not_accepted" if completed and matched else "failed_or_incomplete"
    atomic_json(out / "summary.json", result)
    print(f"[FULL CSR PROFILE] status={result['status']} summary={out/'summary.json'}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/archivenfs/phlusim/HIT_DNS_COMP"))
    parser.add_argument("--case", default="Re54/Ma0.3:65")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    opts = parser.parse_args()
    report = run_profiles(
        root=opts.root, case_spec=opts.case, output_dir=opts.output_dir, seed=opts.seed,
    )
    if report["status"] != "completed_not_accepted":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
