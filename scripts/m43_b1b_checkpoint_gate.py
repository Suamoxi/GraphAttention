"""M43-B1b: baseline vs non-reentrant checkpointed M37 full-graph training.

Runs baseline and checkpointed real HDF5 training steps for 65^3 and
129^3 in four fresh CUDA subprocesses on ONE Grace allocation.
For failed forwards, per-block memory snapshots survive in child reports.
No scientific model is trained and no trained weights are saved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from scripts.m43_b1_memory_gate import (
    DEFAULT_CASES, _revision, atomic_json, inspect_case, parse_case,
)

MODES = ("baseline", "checkpointed")


def compare_65_loss(baseline: dict, checkpointed: dict) -> dict:
    """Short-probe diagnostic loss parity; not a trained-model quality metric."""
    if baseline.get("status") != "ok" or checkpointed.get("status") != "ok":
        return {"status": "unavailable"}
    a = baseline["per_step"]
    b = checkpointed["per_step"]
    if len(a) != len(b):
        return {"status": "step_count_mismatch"}
    pairs = []
    for old, new in zip(a, b):
        previous = float(old["loss_probe_only"])
        current = float(new["loss_probe_only"])
        absolute = abs(previous - current)
        tolerance = 2.0e-4 + 2.0e-4 * abs(previous)
        pairs.append({
            "step": old["step"], "baseline_loss": previous,
            "checkpoint_loss": current,
            "abs_difference": absolute, "tolerance": tolerance,
            "within_tolerance": absolute <= tolerance,
        })
    return {
        "status": "passed" if all(p["within_tolerance"] for p in pairs)
                  else "numerical_mismatch",
        "pairs": pairs,
    }


def run_gate(
    *, root: Path, output_dir: Path, specs: tuple[str, ...] = DEFAULT_CASES,
    steps: int = 2, seed: int = 42, hidden_dim: int = 128,
    num_layers: int = 10, num_heads: int = 4,
) -> dict:
    if (steps < 1 or not specs
        or len({parse_case(spec)[0] for spec in specs}) != len(specs)):
        raise ValueError("steps must be >=1 and cases must be unique")
    cases = [inspect_case(root, spec) for spec in specs]
    if len(cases) != 2 or {case["resolution"] for case in cases} != {65, 129}:
        raise ValueError("M43-B1b paired diagnostic requires exactly 65^3 and 129^3")
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    signature = {
        "root": str(Path(root).resolve(strict=True)), "cases": cases,
        "steps": steps, "seed": seed, "hidden_dim": hidden_dim,
        "num_layers": num_layers, "num_heads": num_heads,
        "revision": _revision(), "profile_blocks": True,
        "modes": list(MODES),
        "schedule": ["65_baseline", "65_checkpointed",
                     "129_baseline", "129_checkpointed"],
    }
    lock = out / ".active.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"concurrent/stale B1b lock: {lock}") from exc
    with os.fdopen(fd, "w") as stream:
        stream.write(f"{socket.gethostname()}, pid {os.getpid()}\n")
    try:
        signature_file = out / "signature.json"
        if signature_file.exists():
            if json.loads(signature_file.read_text()) != signature:
                raise ValueError("resume signature mismatch; use new output directory")
        else:
            atomic_json(signature_file, signature)
        atomic_json(out / "preflight.json", {
            "status": "complete", "host": socket.gethostname(),
            "cases": cases,
        })

        reports: dict[str, dict[str, dict]] = {}
        for case in sorted(cases, key=lambda entry: entry["resolution"]):
            label = f"{case['case'].replace('/', '_')}_{case['resolution']}"
            reports[label] = {}
            for mode in MODES:
                folder = out / "runs" / label / mode
                folder.mkdir(parents=True, exist_ok=True)
                report_file = folder / "report.json"
                log_file = folder / "process.log"
                if report_file.exists():
                    previous = json.loads(report_file.read_text())
                    if previous.get("status") in {
                        "ok", "cuda_oom", "failed", "cuda_unavailable", "process_failed"
                    }:
                        if (previous.get("snapshot_file") != case["snapshot_file"]
                            or previous.get("revision") != signature["revision"]
                            or previous.get("seed") != seed
                            or previous.get("num_steps") != steps
                            or bool(previous.get("activation_checkpointing")) !=
                                (mode == "checkpointed")
                            or not previous.get("profile_blocks")):
                            raise ValueError(f"resume report mismatch: {report_file}")
                        print(f"[M43-B1b] existing {label}/{mode} => {previous['status']}",
                              flush=True)
                        reports[label][mode] = previous
                        continue
                cmd = [
                    sys.executable, "-u", "-m", "scripts.m43_b1_training_step",
                    "--case", case["case"],
                    "--snapshot", case["snapshot_file"],
                    "--mesh", case["mesh_file"],
                    "--side", str(case["resolution"]),
                    "--steps", str(steps), "--seed", str(seed),
                    "--hidden-dim", str(hidden_dim),
                    "--layers", str(num_layers),
                    "--heads", str(num_heads),
                    "--profile-blocks",
                    "--output", str(report_file),
                ]
                if mode == "checkpointed":
                    cmd.append("--activation-checkpointing")
                print(f"[M43-B1b] start {label}/{mode} on {socket.gethostname()}",
                      flush=True)
                start = time.perf_counter()
                with log_file.open("a") as stream:
                    stream.write(
                        f"\n[M43-B1b] launch {label}/{mode} "
                        f"host={socket.gethostname()}\n"
                    )
                    stream.flush()
                    proc = subprocess.run(
                        cmd, stdout=stream, stderr=subprocess.STDOUT, check=False,
                    )
                elapsed = time.perf_counter() - start
                record = (
                    json.loads(report_file.read_text()) if report_file.exists() else {}
                )
                valid = {"ok", "cuda_oom", "failed", "cuda_unavailable"}
                if (record.get("status") not in valid
                    or (proc.returncode != 0 and record.get("status") == "ok")):
                    record["status"] = "process_failed"
                    record["reason"] = (
                        "Worker terminated/incomplete before producing an outcome. "
                        "Inspect subprocess log for CPU OOM, signal or traceback."
                    )
                record.update({
                    "case": case["case"], "expected_side": case["resolution"],
                    "snapshot_file": case["snapshot_file"],
                    "revision": signature["revision"],
                    "seed": seed, "num_steps": steps,
                    "profile_blocks": True,
                    "activation_checkpointing": mode == "checkpointed",
                    "subprocess_exit_code": proc.returncode,
                    "subprocess_wall_seconds": elapsed,
                    "process_log": str(log_file),
                })
                atomic_json(report_file, record)
                reports[label][mode] = record
                atomic_json(out / "progress.json", {
                    "host": socket.gethostname(),
                    "completed": {
                        f"{label}/{mode}": r["status"]
                        for label, entries in reports.items()
                        for mode, r in entries.items()
                    },
                })
                print(f"[M43-B1b] done {label}/{mode}: {record['status']}",
                      flush=True)

        by_side = {
            case["resolution"]: reports[
                f"{case['case'].replace('/', '_')}_{case['resolution']}"
            ] for case in cases
        }
        loss_check = compare_65_loss(
            by_side[65]["baseline"], by_side[65]["checkpointed"],
        )
        expected_steps = lambda r: (r.get("status") == "ok"
                                     and len(r.get("per_step", [])) == steps)
        baseline65 = expected_steps(by_side[65]["baseline"])
        ckpt65 = expected_steps(by_side[65]["checkpointed"])
        ckpt129 = expected_steps(by_side[129]["checkpointed"])
        baseline129_expected = by_side[129]["baseline"].get("status") in {
            "ok", "cuda_oom",
        }
        all_passed = (baseline65 and ckpt65 and ckpt129
                      and baseline129_expected and loss_check["status"] == "passed")
        summary = {
            "status": "checkpoint_passed" if all_passed else "checkpoint_failed",
            "host": socket.gethostname(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "signature": signature, "runs": reports,
            "baseline_65_loss_check": loss_check,
            "baseline_129_status": by_side[129]["baseline"]["status"],
            "checkpoint_129_status": by_side[129]["checkpointed"]["status"],
            "notes": [
                "Synchronized per-block observers perturb forward timings in all modes.",
                "Block memory records are incremental and survive caught CUDA OOM.",
                "One snapshot z-score is probe-only and unsuitable for scientific training.",
                "Checkpointing preserves original M37 weight and attention definitions.",
                "Success means two full optimizer steps at each checkpointed resolution.",
                "No 600-snapshot mixed-resolution training was attempted.",
            ],
        }
        atomic_json(out / "summary.json", summary)
        print(json.dumps({
            "status": summary["status"],
            "by_case": {
                label: {
                    mode: {
                        "status": r["status"],
                        "peak_gib": r.get("gpu_final", {}).get("peak_allocated_gib"),
                        "completed_steps": len(r.get("per_step", [])),
                        "block_records": len(r.get("forward_block_memory", [])),
                    } for mode, r in modes.items()
                } for label, modes in reports.items()
            },
            "loss_parity": loss_check,
        }, indent=2), flush=True)
        return summary
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/archivenfs/phlusim/HIT_DNS_COMP"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--heads", type=int, default=4)
    a = parser.parse_args()
    outcome = run_gate(
        root=a.root, output_dir=a.output_dir,
        specs=tuple(a.case or DEFAULT_CASES),
        steps=a.steps, seed=a.seed,
        hidden_dim=a.hidden_dim, num_layers=a.layers, num_heads=a.heads,
    )
    if outcome["status"] != "checkpoint_passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
