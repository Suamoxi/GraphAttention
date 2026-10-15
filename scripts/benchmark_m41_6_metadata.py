"""M41.6 controlled 32/64 metadata-loading benchmark: baseline/selective/mmap.

All workflows reuse exactly the M41.4 physical graph, original M37 model,
initial seed and per-layout partition assignment. Normal forward timings
use distinct processes without profiling. Subsequent diagnostic forwards
separate CPU NPZ decoding, index conversions, CSR and displacement assembly.

Experimental caching is DISK mmap only; no all-partitions CPU preload.
All 6 first-run full outputs are compared to legacy baseline, elementwise.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from scripts.benchmark_m41_4_scalability import _validate_report
from scripts.benchmark_m41_5_paired_profile import _prepared_layouts, _profile_brief
from scripts.summarize_m41_4_scalability import _output_difference, _stats
from graph_attention.models.selective_disk_dinat import validate_mmap_cache

LAYOUTS = (32, 64)
MODES = ("baseline", "selective", "mmap")


def trial_order(repeats: int) -> list[tuple[int, int, str]]:
    if repeats < 2:
        raise ValueError("at least two repeats required")
    result = []
    for repeat in range(1, repeats+1):
        # Rotate mode order and reverse layout order on alternating rounds.
        rotated = MODES[(repeat-1) % 3:] + MODES[:(repeat-1) % 3]
        for layout in (LAYOUTS if repeat % 2 else tuple(reversed(LAYOUTS))):
            for mode in rotated:
                result.append((repeat, layout, mode))
    return result


def _atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    os.replace(tmp, path)


def _verify_cache(prepared: Path, cache: Path) -> dict:
    plan = json.loads((prepared/"prepare.json").read_text())
    return validate_mmap_cache(
        cache, partition_dir=prepared/"partitions",
        expected_nodes=plan["num_nodes"],
        expected_partitions=plan["num_partitions"],
    )


def run_one(*, prepared: Path, cache: Path, run_dir: Path,
            checkpoint: str, directory: Path,
            layout: int, mode: str, num_nodes: int,
            seed: int, diagnostic: bool, save_full: bool) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory/"forward.json"
    if report_path.is_file():
        if save_full and not (directory/"forward.npy").is_file():
            raise FileNotFoundError(f"saved report lacks required full output: {directory}")
        report = json.loads(report_path.read_text())
        print(f"[M41.6] reuse {directory}", flush=True)
    else:
        if any(directory.iterdir()):
            raise FileExistsError(f"partial or failed run at {directory}; inspect before retrying")
        command = [
            sys.executable, "-u", "-m", "scripts.m41_4_generic_large", "infer",
            "--prepared-dir", str(prepared), "--run-dir", str(run_dir),
            "--checkpoint", checkpoint, "--output-dir", str(directory),
            "--mode", "forward", "--metadata-mode", mode, "--seed", str(seed),
        ]
        if mode == "mmap":
            command.extend(["--metadata-cache-dir", str(cache)])
        if diagnostic:
            command.append("--profile-stages")
        if save_full:
            command.append("--save-forward")
        print(f"[M41.6] {layout=}, {mode=}, {diagnostic=}", flush=True)
        with (directory/"process.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    check=False)
        if result.returncode:
            raise RuntimeError(f"run failed {result.returncode}, log: {directory/'process.log'}")
        if not report_path.is_file():
            raise FileNotFoundError(f"missing output report: {report_path}")
        report = json.loads(report_path.read_text())
    _validate_report(report, num_nodes=num_nodes, partitions=layout,
                     preload=False, seed=seed)
    if report.get("metadata_mode") != mode:
        raise ValueError("metadata mode mismatch in trial report")
    if bool(report.get("diagnostic_profile_enabled")) != diagnostic:
        raise ValueError("diagnostic flag mismatch in trial report")
    if diagnostic and "stage_profile" not in report:
        raise ValueError("missing profiler detail")
    if not report.get("output_sha256"):
        raise ValueError("missing output fingerprint")
    return report


def benchmark(*, base_dir: Path, layout_root: Path,
              cache_root: Path, run_dir: Path, checkpoint: str,
              output_dir: Path, repeats: int = 3, seed: int = 5678) -> dict:
    order = trial_order(repeats)
    base, prepared = _prepared_layouts(base_dir, layout_root)
    run_dir = Path(run_dir).resolve(strict=True)
    for name in (checkpoint, "resolved_config.yaml", "standardizers.pt"):
        if not (run_dir/name).is_file():
            raise FileNotFoundError(run_dir/name)
    cache_root = Path(cache_root).resolve(strict=True)
    cached = {p: cache_root/f"layout_{p:03d}" for p in LAYOUTS}
    for p in LAYOUTS:
        _verify_cache(prepared[p], cached[p])
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out/".active.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("benchmark already running or stale lock: " + str(lock)) from exc
    with os.fdopen(fd, "w") as file:
        file.write(f"{socket.gethostname()}, pid {os.getpid()}\n")
    try:
        signature = {
            "base_dir": str(Path(base_dir).resolve(strict=True)),
            "prepared_32": str(prepared[32]), "prepared_64": str(prepared[64]),
            "cache_32": str(cached[32].resolve(strict=True)),
            "cache_64": str(cached[64].resolve(strict=True)),
            "run_dir": str(run_dir), "checkpoint": checkpoint,
            "seed": seed, "repeats": repeats, "modes": list(MODES),
            "layouts": list(LAYOUTS),
        }
        progress_path = out/"progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress["signature"] != signature:
                raise ValueError("different experiment signature; use a new output directory")
        else:
            progress = {
                "status": "incomplete", "signature": signature, "completed": [],
                "host": socket.gethostname(),
                "trial_order": [f"r{r:02d}_p{p:03d}_{m}" for r,p,m in order],
            }
            _atomic_json(progress_path, progress)
        observed = {p: {m: [] for m in MODES} for p in LAYOUTS}
        ckpt_digest = None
        for rep, layout, mode in order:
            key = f"r{rep:02d}_p{layout:03d}_{mode}"
            folder = out/"timings"/f"layout_{layout:03d}"/mode/f"repeat_{rep:02d}"
            record = run_one(
                prepared=prepared[layout], cache=cached[layout],
                run_dir=run_dir, checkpoint=checkpoint, directory=folder,
                layout=layout, mode=mode, num_nodes=base["num_nodes"],
                seed=seed, diagnostic=False, save_full=(rep == 1),
            )
            digest = record["checkpoint_sha256"]
            if ckpt_digest is None:
                ckpt_digest = digest
            elif ckpt_digest != digest:
                raise ValueError("checkpoint digest changed during benchmark")
            observed[layout][mode].append(record)
            if key not in progress["completed"]:
                progress["completed"].append(key)
            _atomic_json(progress_path, progress)

        diagnostic_profiles = {}
        for layout in LAYOUTS:
            diagnostic_profiles[str(layout)] = {}
            for mode in MODES:
                folder = out/"profiles"/f"layout_{layout:03d}"/mode
                record = run_one(
                    prepared=prepared[layout], cache=cached[layout],
                    run_dir=run_dir, checkpoint=checkpoint, directory=folder,
                    layout=layout, mode=mode, num_nodes=base["num_nodes"],
                    seed=seed, diagnostic=True, save_full=False,
                )
                if record["checkpoint_sha256"] != ckpt_digest:
                    raise ValueError("diagnostic checkpoint digest mismatch")
                brief = _profile_brief(record)
                brief["metadata_detail_seconds"] = record["stage_profile"].get(
                    "metadata_detail_seconds", {}
                )
                brief["metadata_detail_calls"] = record["stage_profile"].get(
                    "metadata_detail_calls", {}
                )
                diagnostic_profiles[str(layout)][mode] = brief
                key = f"profile_p{layout:03d}_{mode}"
                if key not in progress["completed"]:
                    progress["completed"].append(key)
                _atomic_json(progress_path, progress)

        summaries = {}
        for layout in LAYOUTS:
            summaries[str(layout)] = {}
            for mode in MODES:
                records = observed[layout][mode]
                summaries[str(layout)][mode] = {
                    "runs": len(records),
                    "forward_seconds": _stats([r["forward_seconds"] for r in records]),
                    "process_rss_peak_gib": _stats([
                        r["host_memory_after_forward"]["rss_peak_gib"]
                        for r in records
                    ]),
                    "gpu_allocated_peak_gib": _stats([
                        r["gpu_peak_allocated_gib"] for r in records
                    ]),
                    "gpu_reserved_peak_gib": _stats([
                        r["gpu_peak_reserved_gib"] for r in records
                    ]),
                    "plan_seconds": _stats([
                        r["plan_load_seconds"] for r in records
                    ]),
                }
        full_outputs = {
            (p, m): out/"timings"/f"layout_{p:03d}"/m/"repeat_01"/"forward.npy"
            for p in LAYOUTS for m in MODES
        }
        reference = full_outputs[(64, "baseline")]
        comparisons = {}
        for (p, m), candidate in full_outputs.items():
            comparisons[f"{p}_{m}"] = _output_difference(
                reference, candidate, atol=2e-4, rtol=2e-4,
            )
        good = all(v["passed"] for v in comparisons.values())
        summary = {
            "status": "complete" if good else "numerical_mismatch",
            "host": socket.gethostname(), "signature": signature,
            "checkpoint_sha256": ckpt_digest, "num_nodes": base["num_nodes"],
            "unprofiled_runs": repeats * len(LAYOUTS) * len(MODES),
            "diagnostic_runs": len(LAYOUTS) * len(MODES),
            "unprofiled": summaries, "profiles": diagnostic_profiles,
            "full_output_checks": comparisons,
            "all_first_outputs_within_tolerance": good,
            "notes": [
                "All three modes execute the same M37 checkpoint and exact graph topology.",
                "Metadata profile detail stages are nested, separate from top-level stage totals.",
                "CPU RSS is peak process memory, GPU memory is PyTorch allocated/reserved peak.",
                "Profiled stage times insert CUDA synchronizations and must not replace ordinary latency.",
                "Mmap cache is immutable per-partition static CSR/displacement on disk; not CPU preload.",
                "Only the first trial per layout/mode is saved for elementwise comparisons.",
            ],
        }
        _atomic_json(out/"summary.json", summary)
        progress["status"] = summary["status"]
        _atomic_json(progress_path, progress)
        print(json.dumps({
            "status": summary["status"],
            "unprofiled": summaries,
            "profile_stage_s": {
                p: {m: vals["stage_seconds"]
                    for m,vals in group.items()}
                for p,group in diagnostic_profiles.items()
            },
            "profile_metadata_detail_s": {
                p: {m: vals["metadata_detail_seconds"]
                    for m,vals in group.items()}
                for p,group in diagnostic_profiles.items()
            },
            "output_check": {k: v["max_abs"] for k,v in comparisons.items()},
        }, indent=2), flush=True)
        if not good:
            raise ValueError("saved full outputs exceed numerical tolerance")
        return summary
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--layout-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5678)
    a = parser.parse_args()
    benchmark(
        base_dir=a.base_dir, layout_root=a.layout_root,
        cache_root=a.cache_root, run_dir=a.run_dir,
        checkpoint=a.checkpoint, output_dir=a.output_dir,
        repeats=a.repeats, seed=a.seed,
    )


if __name__ == "__main__":
    main()
