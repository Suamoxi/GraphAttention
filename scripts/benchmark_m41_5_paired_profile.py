"""M41.5 matched-host repeated 32-vs-64 latency and stage diagnostics.

Uses the ALREADY PREPARED M41.4 generic unstructured one-hop CSR and
partition-local two-hop files. No global graph rebuild and no model changes.
Six paired, unprofiled forward repetitions per layout, AB/BA crossover order,
then one distinct synchronized diagnostic forward per layout.

Each invocation is a fresh subprocess, enabling clean process peak RSS
and GPU allocator measurements. The diagnostic run adds synchronization
and MUST NOT be used for unprofiled performance comparisons.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

from scripts.benchmark_m41_4_scalability import _validate_report
from scripts.m41_4_repartition import LAYOUTS, layout_path
from scripts.summarize_m41_4_scalability import _output_difference, _stats


LAYOUT_PAIR = (32, 64)


def paired_order(repeats: int) -> tuple[tuple[int, int], ...]:
    if repeats < 2:
        raise ValueError("at least two paired repetitions required")
    return tuple(
        (rep, layout)
        for rep in range(1, repeats + 1)
        for layout in (LAYOUT_PAIR if rep % 2 else tuple(reversed(LAYOUT_PAIR)))
    )


def _atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _prepared_layouts(base_dir: Path, layout_root: Path) -> tuple[dict, dict[int, Path]]:
    base_dir = Path(base_dir).resolve(strict=True)
    root = _load_json(base_dir / "prepare.json")
    if root.get("status") != "generic_topology_ready" or root["num_partitions"] != 64:
        raise ValueError("immutable M41.4 64-partition base is missing")
    if tuple(root["blocks"]) != LAYOUTS[64]:
        raise ValueError("base does not have expected (4,4,4) layout")
    paths = {64: base_dir, 32: layout_path(layout_root, 32).resolve(strict=True)}
    for layout, folder in paths.items():
        report = _load_json(folder / "prepare.json")
        if report["status"] != "generic_topology_ready":
            raise ValueError(f"layout {layout} is not complete")
        if report["num_partitions"] != layout or tuple(report["blocks"]) != LAYOUTS[layout]:
            raise ValueError(f"layout {layout} manifest does not match expected block geometry")
        for name in (
            "num_nodes", "num_onehop_edges", "num_partition_local_edges",
            "num_partition_twohop_edges", "coordinate_divisor",
            "include_self_attention", "include_local_self_attention",
        ):
            if report[name] != root[name]:
                raise ValueError(f"layout {layout} disagrees on immutable graph field {name}")
        if (folder / "coords.npy").resolve() != (base_dir / "coords.npy").resolve():
            raise ValueError(f"layout {layout} must reuse source coordinates")
        if (folder / "csr").resolve() != (base_dir / "csr").resolve():
            raise ValueError(f"layout {layout} must reuse the one-hop CSR")
        if not (folder / "partitions" / "partitions.json").is_file():
            raise ValueError(f"layout {layout} lacks halo partitions")
    return root, paths


def _run_forward(
    prepared: Path, run_dir: Path, checkpoint: str,
    directory: Path, *, seed: int,
    num_nodes: int, layout: int,
    diagnostic: bool, save_full_output: bool,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "forward.json"
    if report_path.is_file():
        if save_full_output and not (directory / "forward.npy").is_file():
            raise FileNotFoundError(f"existing {report_path} lacks required full output")
        report = _load_json(report_path)
        print(f"[M41.5] Reusing complete {report_path}", flush=True)
    else:
        if any(directory.iterdir()):
            raise FileExistsError(
                f"{directory} contains partial artifacts without a complete forward report; "
                "inspect manually before retry"
            )
        command = [
            sys.executable, "-u", "-m", "scripts.m41_4_generic_large", "infer",
            "--prepared-dir", str(prepared), "--run-dir", str(run_dir),
            "--checkpoint", checkpoint, "--mode", "forward",
            "--output-dir", str(directory), "--seed", str(seed),
        ]
        if diagnostic:
            command.append("--profile-stages")
        if save_full_output:
            command.append("--save-forward")
        print(f"[M41.5] Running {layout} partitions; "
              f"diagnostic={diagnostic}, dir={directory}", flush=True)
        with (directory / "process.log").open("w") as stream:
            process = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False,
            )
        if process.returncode:
            raise RuntimeError(
                f"forward failed (exit {process.returncode}): "
                f"{directory / 'process.log'}"
            )
        if not report_path.is_file():
            raise FileNotFoundError(f"forward exited zero without {report_path}")
        report = _load_json(report_path)
    _validate_report(
        report, num_nodes=num_nodes, partitions=layout,
        preload=False, seed=seed,
    )
    if bool(report.get("diagnostic_profile_enabled", False)) != diagnostic:
        raise ValueError(f"unexpected profiler mode in {report_path}")
    if diagnostic and "stage_profile" not in report:
        raise ValueError("requested synchronized stage profile was not recorded")
    if not diagnostic and "stage_profile" in report:
        raise ValueError("unprofiled benchmark must not contain diagnostic stage timers")
    if not report.get("output_sha256"):
        raise ValueError("forward must contain a deterministic output digest")
    return report


def _profile_brief(report: dict) -> dict:
    profile = report["stage_profile"]
    seconds = profile["stage_seconds"]
    wall = float(profile["profiled_forward_wall_seconds"])
    if wall <= 0 or sum(seconds.values()) <= 0 or sum(seconds.values()) > wall * 1.04:
        raise ValueError("diagnostic stage durations do not fit forward wall time")
    layer_seconds = profile["layer_wall_seconds"]
    if not layer_seconds or any(float(v) <= 0 for v in layer_seconds.values()):
        raise ValueError("profile has no completed transformer layer measurements")
    return {
        "profiled_forward_seconds": report["forward_seconds"],
        "local_vs_exact_two_hop_layer_wall_seconds": {
            "local": sum(float(v) for k, v in layer_seconds.items() if int(k) % 2 == 0),
            "exact_two_hop": sum(float(v) for k, v in layer_seconds.items() if int(k) % 2 == 1),
        },
        "local_vs_exact_two_hop_attention_seconds": {
            "local": sum(float(v.get("gpu_attention_context", 0.0))
                         for k, v in profile["layers_stage_seconds"].items()
                         if int(k) % 2 == 0),
            "exact_two_hop": sum(float(v.get("gpu_attention_context", 0.0))
                               for k, v in profile["layers_stage_seconds"].items()
                               if int(k) % 2 == 1),
        },
        "stage_seconds": seconds,
        "stage_share_of_profiled_forward_pct": {
            name: 100.0 * value / wall for name, value in seconds.items()
        },
        "stage_calls": profile["stage_calls"],
        "layer_wall_seconds": layer_seconds,
        "layers_stage_seconds": profile["layers_stage_seconds"],
        "synchronization_calls": profile["synchronization_calls"],
        "profile_timing_method": profile["timing_method"],
        "cpu_peak_rss_gib": report["host_memory_after_forward"]["rss_peak_gib"],
        "gpu_peak_allocated_gib": report["gpu_peak_allocated_gib"],
        "gpu_peak_reserved_gib": report["gpu_peak_reserved_gib"],
    }


def paired_benchmark(
    *, base_dir: Path, layout_root: Path,
    run_dir: Path, checkpoint: str,
    output_dir: Path, repeats: int = 6, seed: int = 5678,
) -> dict:
    order = paired_order(repeats)
    root, prepared = _prepared_layouts(base_dir, layout_root)
    run_dir = Path(run_dir).resolve(strict=True)
    for filename in (checkpoint, "resolved_config.yaml", "standardizers.pt"):
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(run_dir / filename)
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".active.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"another M41.5 job is active or a lock needs inspection: {lock}"
        ) from exc
    with os.fdopen(lock_fd, "w") as stream:
        stream.write(f"host={socket.gethostname()}, pid={os.getpid()}\n")
    try:
        signature = {
            "base_dir": str(Path(base_dir).resolve(strict=True)),
            "prepared_32": str(prepared[32]),
            "prepared_64": str(prepared[64]),
            "run_dir": str(run_dir), "checkpoint": checkpoint,
            "seed": seed, "repeats": repeats, "layouts": [32, 64],
            "profile": "one_separate_diagnostic_forward_per_layout",
        }
        manifest_file = out / "progress.json"
        if manifest_file.is_file():
            progress = _load_json(manifest_file)
            if progress["signature"] != signature:
                raise ValueError("existing progress signature differs; use a new output directory")
        else:
            progress = {
                "status": "incomplete", "signature": signature,
                "host": socket.gethostname(), "completed": [],
                "trial_order": [f"r{r:02d}_p{p:03d}" for r, p in order],
            }
            _atomic_json(manifest_file, progress)
        reports: dict[int, list[dict]] = {32: [], 64: []}
        checkpoint_hash = None
        for rep, layout in order:
            key = f"r{rep:02d}_p{layout:03d}"
            folder = out / "timings" / f"layout_{layout:03d}" / f"repeat_{rep:02d}"
            record = _run_forward(
                prepared[layout], run_dir, checkpoint, folder,
                seed=seed, num_nodes=root["num_nodes"], layout=layout,
                diagnostic=False, save_full_output=(rep == 1),
            )
            digest = record["checkpoint_sha256"]
            if checkpoint_hash is None:
                checkpoint_hash = digest
            elif digest != checkpoint_hash:
                raise ValueError("checkpoint differs across runs")
            reports[layout].append(record)
            if key not in progress["completed"]:
                progress["completed"].append(key)
            _atomic_json(manifest_file, progress)
            print(f"[M41.5] {key} forward {record['forward_seconds']:.3f}s", flush=True)

        profiles = {}
        for layout in LAYOUT_PAIR:
            folder = out / "profiles" / f"layout_{layout:03d}"
            record = _run_forward(
                prepared[layout], run_dir, checkpoint, folder,
                seed=seed, num_nodes=root["num_nodes"], layout=layout,
                diagnostic=True, save_full_output=False,
            )
            if record["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("profiling checkpoint differs from unprofiled forwards")
            profiles[str(layout)] = _profile_brief(record)
            key = f"profile_p{layout:03d}"
            if key not in progress["completed"]:
                progress["completed"].append(key)
            _atomic_json(manifest_file, progress)

        first_32 = out / "timings/layout_032/repeat_01/forward.npy"
        first_64 = out / "timings/layout_064/repeat_01/forward.npy"
        comparison = _output_difference(
            first_64, first_32, atol=2e-4, rtol=2e-4,
        )
        aggregate = {}
        for layout in LAYOUT_PAIR:
            runs = reports[layout]
            aggregate[str(layout)] = {
                "repetitions": len(runs),
                "forward_seconds": _stats([r["forward_seconds"] for r in runs]),
                "host_peak_rss_gib": _stats([
                    r["host_memory_after_forward"]["rss_peak_gib"] for r in runs
                ]),
                "gpu_peak_allocated_gib": _stats([
                    r["gpu_peak_allocated_gib"] for r in runs
                ]),
                "gpu_peak_reserved_gib": _stats([
                    r["gpu_peak_reserved_gib"] for r in runs
                ]),
                "plan_load_seconds": _stats([
                    r["plan_load_seconds"] for r in runs
                ]),
            }
        paired_differences = [
            {
                "repeat": i + 1,
                "seconds_32_minus_64": reports[32][i]["forward_seconds"] -
                reports[64][i]["forward_seconds"],
            } for i in range(repeats)
        ]
        summary = {
            "status": "complete" if comparison["passed"] else "output_mismatch",
            "signature": signature, "host": socket.gethostname(),
            "checkpoint_sha256": checkpoint_hash, "node_count": root["num_nodes"],
            "num_forward_runs": repeats * 2,
            "num_diagnostic_forwards": 2,
            "trial_order": progress["trial_order"],
            "unprofiled": aggregate,
            "paired_differences": paired_differences,
            "profile": profiles,
            "first_saved_output_comparison": comparison,
            "notes": [
                "Stage profile inserts CUDA synchronization; do not compare its total to unprofiled wall time.",
                "This is a fixed node-count and same-host test of 32 versus 64 partitions.",
                "All model computations, checkpoint, physical graph edges and random input seed are fixed.",
                "The full first-trial outputs are compared elementwise; other trials are not fully compared.",
                "One allocatable Grace GPU; no multi-GPU scaling or physical fidelity conclusions.",
            ],
        }
        _atomic_json(out / "summary.json", summary)
        progress["status"] = summary["status"]
        _atomic_json(manifest_file, progress)
        print(json.dumps({
            "status": summary["status"],
            "unprofiled": summary["unprofiled"],
            "profile_stage_seconds": {
                k: v["stage_seconds"] for k, v in profiles.items()
            },
            "output_difference": comparison,
        }, indent=2), flush=True)
        if not comparison["passed"]:
            raise ValueError("32 and 64 partition outputs differ beyond tolerance")
        return summary
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--layout-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--seed", type=int, default=5678)
    args = p.parse_args()
    paired_benchmark(
        base_dir=args.base_dir, layout_root=args.layout_root,
        run_dir=args.run_dir, checkpoint=args.checkpoint,
        output_dir=args.output_dir, repeats=args.repeats, seed=args.seed,
    )


if __name__ == "__main__":
    main()
