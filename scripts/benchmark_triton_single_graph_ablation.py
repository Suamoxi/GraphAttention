"""Real-HDF5 M37 ablation: packed batch_index vs native single-graph expand.

Four fresh CUDA processes: original and Triton CSR backends in each mode.
Reuses the M43-B1 checkpointed Flow Matching runner, fixed seeds and one
unchanged real snapshot. M47 scientific training is NOT started or altered.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from scripts.benchmark_triton_full_training import BACKENDS, METRICS, run_pair
from scripts.m43_b1_memory_gate import atomic_json


def run_ablation(*, root: Path, case: str, output_dir: Path,
                 steps: int = 4, seed: int = 42) -> dict:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "protocol": "m47_triton_single_graph_broadcast_ablation_v1",
        "status": "running",
        "case": case,
        "steps": steps,
        "seed": seed,
        "modes": {},
        "note": (
            "Packed graph B=1 vs mathematically equivalent single-graph path; "
            "probe-only state normalization, not scientific M47 training. "
            "Do not compare CUDA-profiled times with these unprofiled medians."
        ),
    }
    atomic_json(output_dir / "progress.json", result)
    print(f"ARTIFACT_COMPARISON={output_dir / 'comparison.json'}", flush=True)
    for mode, unbatched in (("packed", False), ("single_graph", True)):
        print(f"ARTIFACT_{mode.upper()}_SUMMARY={output_dir / mode / 'summary.json'}", flush=True)
        for backend in BACKENDS:
            print(
                f"ARTIFACT_{mode.upper()}_{backend.upper()}_REPORT="
                f"{output_dir / mode / backend / 'report.json'}", flush=True,
            )
            print(
                f"ARTIFACT_{mode.upper()}_{backend.upper()}_LOG="
                f"{output_dir / mode / backend / 'process.log'}", flush=True,
            )
        result["modes"][mode] = run_pair(
            root=root, case_spec=case, output_dir=output_dir / mode,
            steps=steps, seed=seed, single_graph_forward=unbatched,
        )
        atomic_json(output_dir / "progress.json", result)

    packed = result["modes"]["packed"]
    expanded = result["modes"]["single_graph"]
    comparison = {}
    valid = all(
        part["status"] == "completed_not_accepted" for part in (packed, expanded)
    )
    for backend in BACKENDS:
        old = packed["backends"][backend]
        new = expanded["backends"][backend]
        old_m = old["measurements"]
        new_m = new["measurements"]
        matched = (
            old["graph"] is not None and old["graph"] == new["graph"]
            and old["initial_weights_sha256"] is not None
            and old["initial_weights_sha256"] == new["initial_weights_sha256"]
            and old["model_parameters"] == new["model_parameters"]
        )
        losses = (
            [abs(a-b) for a, b in zip(old_m["losses"], new_m["losses"])]
            if old_m is not None and new_m is not None else None
        )
        max_loss_diff = max(losses) if losses else None
        numeric_gate = max_loss_diff is not None and max_loss_diff <= 1.0e-4
        comparison[backend] = {
            "matched_graph_inputs_and_initial_weights": matched,
            "maximum_absolute_probe_loss_difference": max_loss_diff,
            "loss_parity_gate_atol": 1.0e-4,
            "loss_parity_passed": numeric_gate,
            "packed_to_single_graph_speedup": {
                metric: old_m["median_seconds"][metric] / new_m["median_seconds"][metric]
                for metric in METRICS
            } if old_m and new_m else None,
            "packed_peak_allocated_gib": (
                old_m["peak_allocated_gib"] if old_m else None
            ),
            "single_graph_peak_allocated_gib": (
                new_m["peak_allocated_gib"] if new_m else None
            ),
        }
        valid = valid and matched and numeric_gate
    result["comparison"] = comparison
    result["status"] = "completed_not_accepted" if valid else "failed_or_review_required"
    atomic_json(output_dir / "comparison.json", result)
    print(f"[SINGLE GRAPH ABLATION] status={result['status']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/archivenfs/phlusim/HIT_DNS_COMP"))
    parser.add_argument("--case", default="Re54/Ma0.3:65")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_ablation(
        root=args.root, case=args.case, output_dir=args.output_dir,
        steps=args.steps, seed=args.seed,
    )
    if result["status"] != "completed_not_accepted":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
