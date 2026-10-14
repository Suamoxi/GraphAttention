"""Controlled GH200 M41.4 forward benchmark using fresh Python processes.

Only measures computation and CPU/GPU resource use. The complete 257^3
one-hop CSR is reused; every layout uses its own exact owned-query halos.
Repeated runs alternate lazy/preloaded order to limit order bias.
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

from scripts.m41_4_repartition import LAYOUTS, layout_path


def _atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _plan(repeats: int, strategies: tuple[str, ...]) -> list[tuple[int, str]]:
    """Balanced crossover ordering; use a clean subprocess for each trial."""
    if repeats < 1 or not strategies or len(set(strategies)) != len(strategies):
        raise ValueError("repeats must be positive and strategies unique/nonempty")
    if not set(strategies) <= {"lazy", "preload"}:
        raise ValueError("strategies must be lazy and/or preload")
    # AB, BA, AB ...; always start from caller-specified canonical order.
    return [(r, mode) for r in range(1, repeats + 1)
            for mode in (strategies if r % 2 else tuple(reversed(strategies)))]


def _validate_report(report: dict, *, num_nodes: int, partitions: int,
                     preload: bool, seed: int) -> None:
    if (report.get("status") != "full_highres_forward_passed"
        or report.get("nodes") != num_nodes
        or report.get("output_shape") != [num_nodes, 5]
        or report.get("partition_count") != partitions
        or report.get("preload_partitions") is not preload
        or report.get("seed") != seed):
        raise ValueError("forward report does not match benchmark run configuration")
    for metric in ("forward_seconds", "gpu_peak_allocated_gib"):
        if not isinstance(report.get(metric), (float, int)) or report[metric] <= 0:
            raise ValueError(f"invalid {metric} in forward report")
    if report.get("host_memory_after_forward", {}).get("rss_peak_gib", 0) <= 0:
        raise ValueError("forward report is missing peak process RSS")
    if not isinstance(report.get("checkpoint_sha256"), str):
        raise ValueError("missing source checkpoint digest")


def benchmark(
    *, base_dir: Path, layout_root: Path, layout: int,
    run_dir: Path, checkpoint: str, output_dir: Path,
    repeats: int, strategies: tuple[str, ...], seed: int = 5678,
    save_first_full_output: bool = True,
) -> dict:
    if layout not in LAYOUTS:
        raise ValueError(f"unsupported layout {layout}")
    base_dir = Path(base_dir).resolve(strict=True)
    prepared = (base_dir if layout == 64 else layout_path(layout_root, layout)).resolve(strict=True)
    p = json.loads((prepared / "prepare.json").read_text())
    if (p["status"] != "generic_topology_ready"
            or p["num_partitions"] != layout
            or tuple(p["blocks"]) != LAYOUTS[layout]):
        raise ValueError("prepared layout is incomplete or has incorrect ownership blocks")
    base = json.loads((base_dir / "prepare.json").read_text())
    for metric in ("num_nodes", "num_onehop_edges", "num_partition_local_edges",
                   "num_partition_twohop_edges", "coordinate_divisor",
                   "include_self_attention", "include_local_self_attention"):
        if p[metric] != base[metric]:
            raise ValueError(f"layout changed physical graph or preprocessing: {metric}")
    if (prepared / "coords.npy").resolve() != (base_dir / "coords.npy").resolve():
        raise ValueError("layout must reuse original immutable coordinates")
    if (prepared / "csr").resolve() != (base_dir / "csr").resolve():
        raise ValueError("layout must reuse original immutable one-hop CSR")
    if not (prepared / "partitions" / "partitions.json").is_file():
        raise FileNotFoundError("missing owned-query halo manifest")
    plan = _plan(repeats, strategies)
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".active.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"another benchmark may be running: {lock}") from exc
    with os.fdopen(fd, "w") as stream:
        stream.write(f"host={socket.gethostname()} pid={os.getpid()}\n")
    try:
        signature = {
            "layout": layout, "blocks": list(LAYOUTS[layout]),
            "base_prepared_dir": str(base_dir), "prepared_dir": str(prepared),
            "run_dir": str(Path(run_dir).resolve(strict=True)),
            "checkpoint": checkpoint, "seed": seed,
            "repeats": repeats, "strategies": list(strategies),
            "save_first_full_output": bool(save_first_full_output),
        }
        manifest_file = out / "benchmark.json"
        if manifest_file.exists():
            doc = json.loads(manifest_file.read_text())
            if doc["signature"] != signature:
                raise ValueError("existing benchmark signature differs; use a new output directory")
        else:
            doc = {"status": "incomplete", "signature": signature,
                   "hostname": socket.gethostname(), "runs": {},
                   "trial_order": [f"{strategy}_r{r:02d}" for r, strategy in plan]}
            _atomic_json(manifest_file, doc)
        ckpt_digest = None
        for r, strategy in plan:
            key = f"{strategy}_r{r:02d}"
            directory = out / key
            directory.mkdir(exist_ok=True)
            report_file = directory / "forward.json"
            need_full_output = (
                save_first_full_output and r == 1
                and not (directory / "forward.npy").is_file()
            )
            if report_file.is_file() and not need_full_output:
                report = json.loads(report_file.read_text())
                print(f"[benchmark] reusing completed {key}", flush=True)
            else:
                command = [
                    sys.executable, "-u", "-m", "scripts.m41_4_generic_large", "infer",
                    "--prepared-dir", str(prepared),
                    "--run-dir", signature["run_dir"], "--checkpoint", checkpoint,
                    "--output-dir", str(directory), "--mode", "forward",
                    "--seed", str(seed),
                ]
                if strategy == "preload":
                    command.append("--preload-partitions")
                if save_first_full_output and r == 1:
                    command.append("--save-forward")
                print(f"[benchmark] {layout} partitions {key} on {socket.gethostname()}", flush=True)
                t0 = time.perf_counter()
                with (directory / "process.log").open("w") as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                            check=False)
                if result.returncode:
                    raise RuntimeError(
                        f"{key} failed (exit {result.returncode}); "
                        f"inspect {directory / 'process.log'}"
                    )
                if not report_file.is_file():
                    raise FileNotFoundError(f"forward exited zero without report: {key}")
                report = json.loads(report_file.read_text())
                print(f"[benchmark] {key} subprocess {time.perf_counter()-t0:.2f}s", flush=True)
            _validate_report(report, num_nodes=p["num_nodes"], partitions=layout,
                             preload=(strategy == "preload"), seed=seed)
            if ckpt_digest is None:
                ckpt_digest = report["checkpoint_sha256"]
            elif report["checkpoint_sha256"] != ckpt_digest:
                raise ValueError("checkpoint changed between trials")
            doc["runs"][key] = {
                "forward_file": str(report_file),
                "forward_seconds": report["forward_seconds"],
                "rss_peak_gib": report["host_memory_after_forward"]["rss_peak_gib"],
                "gpu_peak_allocated_gib": report["gpu_peak_allocated_gib"],
                "gpu_peak_reserved_gib": report["gpu_peak_reserved_gib"],
                "plan_load_seconds": report["plan_load_seconds"],
                "output_sha256": report.get("output_sha256"),
                "saved_full_output": (directory / "forward.npy").is_file(),
            }
            _atomic_json(manifest_file, doc)
        doc["status"] = "complete"
        doc["checkpoint_sha256"] = ckpt_digest
        doc["num_nodes"] = p["num_nodes"]
        doc["num_onehop_edges"] = p["num_onehop_edges"]
        doc["num_twohop_edges"] = p["num_partition_twohop_edges"]
        doc["max_partition_owned_nodes"] = p["max_partition_owned_nodes"]
        doc["max_partition_read_nodes"] = p["max_partition_read_nodes"]
        _atomic_json(manifest_file, doc)
        print(f"[benchmark] COMPLETE layout {layout}, {len(doc['runs'])} trials", flush=True)
        return doc
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--layout-root", type=Path, required=True)
    p.add_argument("--layout", type=int, choices=sorted(LAYOUTS), required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--strategies", nargs="+", choices=["lazy", "preload"], default=["lazy", "preload"])
    p.add_argument("--seed", type=int, default=5678)
    p.add_argument("--no-save-first-full-output", action="store_true")
    args = p.parse_args()
    benchmark(
        base_dir=args.base_dir, layout_root=args.layout_root,
        layout=args.layout, run_dir=args.run_dir, checkpoint=args.checkpoint,
        output_dir=args.output_dir, repeats=args.repeats,
        strategies=tuple(args.strategies), seed=args.seed,
        save_first_full_output=not args.no_save_first_full_output,
    )


if __name__ == "__main__":
    main()
