"""M43-B1: resumable, real-mesh 65^3/129^3 M37 training-memory gate.

Uses read-only HDF5 metadata inspection before any GPU work, then launches
each one-snapshot training probe in a separate process on the SAME Grace
allocation. Records both successes and memory failures for later decisions.
Does NOT start a multi-case training run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import h5py

DEFAULT_CASES = ("Re54/Ma0.3:65", "Re94/Ma0.3:129")
FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")
_CASE_RE = re.compile(r"^(Re[0-9]+)/(Ma[0-9]+(?:\.[0-9]+)?):([0-9]+)$")


def parse_case(spec: str) -> tuple[str, int]:
    match = _CASE_RE.fullmatch(spec)
    if match is None:
        raise ValueError(f"expected 'Re54/Ma0.3:65', got {spec!r}")
    relative = f"{match.group(1)}/{match.group(2)}"
    side = int(match.group(3))
    if side < 3:
        raise ValueError("mesh side must be >= 3")
    return relative, side


def choose_snapshot(case_directory: Path) -> tuple[Path, int]:
    roots = (case_directory / "SOLUT", case_directory / "RUN" / "SOLUT")
    matches = []
    for directory in roots:
        if directory.is_dir():
            found = sorted(p for p in directory.glob("*.h5") if p.is_file())
            if found:
                matches.append(found)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one nonempty SOLUT directory for {case_directory}, "
            f"found {len(matches)}; disambiguate manually"
        )
    files = matches[0]
    return files[len(files) // 2].resolve(strict=True), len(files)


def inspect_case(root: Path, spec: str) -> dict:
    case, side = parse_case(spec)
    case_directory = (Path(root).resolve(strict=True) / case)
    if not case_directory.is_dir():
        raise NotADirectoryError(case_directory)
    mesh_path = (case_directory / "MESH" / "mesh.mesh.h5").resolve(strict=True)
    snapshot, count = choose_snapshot(case_directory)
    with h5py.File(mesh_path, "r") as mesh:
        coord_counts = {
            axis: int(mesh[f"Coordinates/{axis}"].shape[0]) for axis in "xyz"
        }
        connect = mesh["Connectivity/hex->node"]
        if connect.size % 8:
            raise ValueError(f"non-hexahedral connectivity length in {mesh_path}")
        cells = int(connect.size // 8)
    expected_nodes = side**3
    expected_cells = (side-1)**3
    if len(set(coord_counts.values())) != 1 or next(iter(coord_counts.values())) != expected_nodes:
        raise ValueError(f"{case} expected {expected_nodes} nodes, got {coord_counts}")
    if cells != expected_cells:
        raise ValueError(f"{case} expected {expected_cells} cells, got {cells}")
    with h5py.File(snapshot, "r") as file:
        field_meta = {}
        for name in FIELDS:
            dset = file[f"GaseousPhase/{name}"]
            if dset.shape != (expected_nodes,):
                raise ValueError(
                    f"{case} {name} expected {(expected_nodes,)}, got {dset.shape}"
                )
            field_meta[name] = {"shape": list(dset.shape), "dtype": str(dset.dtype)}
    return {
        "case": case, "resolution": side, "nodes": expected_nodes,
        "hex_cells": expected_cells,
        "mesh_file": str(mesh_path), "snapshot_file": str(snapshot),
        "snapshot_count": count, "fields": field_meta,
        "snapshot_selection": "middle_of_sorted_solution_filenames",
    }


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def _revision() -> str | None:
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True,
        text=True, check=False,
    )
    return p.stdout.strip() if p.returncode == 0 else None


def run_memory_gate(
    *, root: Path, output_dir: Path, specs: tuple[str, ...],
    steps: int = 2, seed: int = 42,
    hidden_dim: int = 128, num_layers: int = 10, num_heads: int = 4,
    preflight_only: bool = False,
) -> dict:
    if steps < 1 or not specs or len({parse_case(x)[0] for x in specs}) != len(specs):
        raise ValueError("steps must be positive, cases nonempty and unique")
    prepared = [inspect_case(root, spec) for spec in specs]
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    signature = {
        "root": str(Path(root).resolve(strict=True)),
        "cases": prepared, "steps": steps, "seed": seed,
        "hidden_dim": hidden_dim, "num_layers": num_layers, "num_heads": num_heads,
        "probe_only_scaling": True,
        "revision": _revision(),
    }
    lock = out / ".active.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"an experiment is already running or lock is stale: {lock}") from exc
    with os.fdopen(fd, "w") as stream:
        stream.write(f"host={socket.gethostname()}, pid={os.getpid()}\n")
    try:
        signature_file = out / "signature.json"
        if signature_file.exists():
            if json.loads(signature_file.read_text()) != signature:
                raise ValueError("resume signature changed; choose another output directory")
        else:
            atomic_json(signature_file, signature)
        atomic_json(out / "preflight.json", {
            "status": "complete", "host": socket.gethostname(),
            "cases": prepared,
            "note": "Read-only shape/dtype and snapshot inspection, no graph construction",
        })
        if preflight_only:
            return {"status": "preflight_complete", "cases": prepared}

        summaries: dict[str, dict] = {}
        for case_info in prepared:
            case = case_info["case"]
            key = f"{case.replace('/', '_')}_{case_info['resolution']}"
            folder = out / "runs" / key
            folder.mkdir(parents=True, exist_ok=True)
            report_file = folder / "report.json"
            log_file = folder / "process.log"
            if report_file.is_file():
                existing = json.loads(report_file.read_text())
                if existing.get("status") in {
                    "ok", "cuda_oom", "failed", "cuda_unavailable", "process_failed",
                }:
                    if (existing.get("expected_side") != case_info["resolution"]
                        or existing.get("snapshot_file") != case_info["snapshot_file"]
                        or existing.get("seed") != seed
                        or existing.get("revision") != signature["revision"]):
                        raise ValueError(f"existing outcome does not match signature: {report_file}")
                    print(f"[M43-B1] Reusing completed {key}: {existing['status']}", flush=True)
                    summaries[key] = existing
                    continue
            command = [
                sys.executable, "-u", "-m", "scripts.m43_b1_training_step",
                "--case", case, "--snapshot", case_info["snapshot_file"],
                "--mesh", case_info["mesh_file"],
                "--side", str(case_info["resolution"]),
                "--steps", str(steps), "--seed", str(seed),
                "--hidden-dim", str(hidden_dim),
                "--layers", str(num_layers), "--heads", str(num_heads),
                "--output", str(report_file),
            ]
            print(f"[M43-B1] Starting {key}, steps={steps}", flush=True)
            start = time.perf_counter()
            with log_file.open("a") as stream:
                stream.write(
                    f"\n[M43-B1] New process start: host={socket.gethostname()}\n"
                )
                stream.flush()
                process = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT,
                    check=False,
                )
            elapsed = time.perf_counter() - start
            if report_file.is_file():
                record = json.loads(report_file.read_text())
            else:
                record = {}
            if (record.get("status") not in
                {"ok", "cuda_oom", "failed", "cuda_unavailable"}
                or (process.returncode != 0 and record.get("status") == "ok")):
                record["status"] = "process_failed"
                record["reason"] = (
                    "child exited before completing result; may be CPU OOM, "
                    "SIGKILL, environment or other process failure"
                )
            record.update({
                "case": case, "expected_side": case_info["resolution"],
                "snapshot_file": case_info["snapshot_file"],
                "mesh_file": case_info["mesh_file"],
                "seed": seed, "subprocess_exit_code": process.returncode,
                "subprocess_wall_seconds": elapsed,
                "process_log": str(log_file),
            })
            atomic_json(report_file, record)
            summaries[key] = record
            atomic_json(out / "progress.json", {
                "host": socket.gethostname(),
                "completed": {name: v["status"] for name, v in summaries.items()},
            })
            print(f"[M43-B1] {key}: {record['status']} (code {process.returncode})",
                  flush=True)
        good = all(r["status"] == "ok" and len(r.get("per_step", [])) == steps
                   for r in summaries.values())
        payload = {
            "status": "all_passed" if good else "not_all_passed",
            "host": socket.gethostname(),
            "signature": signature,
            "runs": summaries,
            "interpretation": (
                "Pass indicates two actual full-graph M37 backward+AdamW steps fit, "
                "not that scientific mixed-resolution training or its physics normalization "
                "has been validated. No automated long training starts."
            ),
        }
        atomic_json(out / "summary.json", payload)
        return payload
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/archivenfs/phlusim/HIT_DNS_COMP"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", default=None,
                        help="Repeat e.g. --case Re54/Ma0.3:65")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run_memory_gate(
        root=args.root, output_dir=args.output_dir,
        specs=tuple(args.case or DEFAULT_CASES), steps=args.steps,
        seed=args.seed, hidden_dim=args.hidden_dim,
        num_layers=args.layers, num_heads=args.heads,
        preflight_only=args.preflight_only,
    )
    print(json.dumps({
        "status": result["status"],
        "runs": {
            k: {
                "status": v["status"],
                "peak_cpu_rss_gib": v.get("peak_cpu_rss_gib"),
                "train_steps": v.get("per_step"),
            }
            for k, v in result.get("runs", {}).items()
        },
    }, indent=2), flush=True)
    if result["status"] not in {"all_passed", "preflight_complete"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
