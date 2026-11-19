"""M48.4e phase-A: isolated GPU-only capacity sweep for exact-halo TwoScaleDiNAT.

No DNS fields, CEDAR projections, training checkpoints or validation data are read.
Only an authenticated teacher manifest and frozen scaler identify the two grids.
Each subprocess builds an isolated, disconnected batch of synthetic five-channel
patches with the *actual* M48 geometry and static/dynamic forward/backward.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
from dataclasses import fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback

import torch


SCHEMA = "graph_attention.m48_4e_gpu_capacity.v1"
GRAPH_KEYS = ("fine_global", "coarse_global", "owned_fine_local", "fine_coords",
              "coarse_coords", "fine_edges", "fine_dilated", "coarse_edges",
              "coarse_dilated", "coarse_to_fine", "fine_to_coarse", "cross_weights")


def positive_ints(value: str) -> list[int]:
    result = [int(x.strip()) for x in value.split(",")]
    if not result or any(x <= 0 for x in result) or len(set(result)) != len(result):
        raise ValueError("expected nonempty comma-separated distinct positive integers")
    return sorted(result)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w") as fh:
        json.dump(obj, fh, indent=2, allow_nan=False, sort_keys=True)
        fh.write("\n")
    os.replace(temporary, path)


def graph_tensor_bytes(geometry) -> int:
    return sum(getattr(geometry, name).numel() * getattr(geometry, name).element_size()
               for name in GRAPH_KEYS)


def pack_repeated_geometry(geometry, batch_size: int):
    """Disjoint union of identical, original-global-ID exact halos.

    Unlike building B different halos, this explicitly isolates GPU capacity
    from CPU patch sampling. Each replica gets its own node features and FM t.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    nf = int(geometry.fine_coords.shape[0])
    nc = int(geometry.coarse_coords.shape[0])
    fo = torch.arange(batch_size, dtype=torch.long) * nf
    co = torch.arange(batch_size, dtype=torch.long) * nc

    def edges(value: torch.Tensor, source_offset: torch.Tensor,
              target_offset: torch.Tensor) -> torch.Tensor:
        offsets = torch.stack((source_offset, target_offset), dim=1).unsqueeze(-1)
        return (value.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1).contiguous()

    result = replace(
        geometry,
        fine_global=geometry.fine_global.repeat(batch_size),
        coarse_global=geometry.coarse_global.repeat(batch_size),
        owned_fine_local=(geometry.owned_fine_local[None, :] + fo[:, None]).reshape(-1),
        fine_coords=geometry.fine_coords.repeat(batch_size, 1),
        coarse_coords=geometry.coarse_coords.repeat(batch_size, 1),
        fine_edges=edges(geometry.fine_edges, fo, fo),
        fine_dilated=edges(geometry.fine_dilated, fo, fo),
        coarse_edges=edges(geometry.coarse_edges, co, co),
        coarse_dilated=edges(geometry.coarse_dilated, co, co),
        coarse_to_fine=edges(geometry.coarse_to_fine, co, fo),
        fine_to_coarse=edges(geometry.fine_to_coarse, fo, co),
        cross_weights=geometry.cross_weights.repeat(batch_size),
    )
    fine_batch = torch.arange(batch_size, dtype=torch.long).repeat_interleave(nf)
    coarse_batch = torch.arange(batch_size, dtype=torch.long).repeat_interleave(nc)
    return result, fine_batch, coarse_batch


def verify_packed_graph(geometry, fine_batch, coarse_batch) -> None:
    """Fail if any adjacency would accidentally couple independent samples."""
    for name, batch in (("fine_edges", fine_batch), ("fine_dilated", fine_batch),
                        ("coarse_edges", coarse_batch), ("coarse_dilated", coarse_batch)):
        e = getattr(geometry, name)
        if not torch.equal(batch[e[0]], batch[e[1]]):
            raise RuntimeError(f"{name}: cross-sample edge")
    for name, src, dst in (("coarse_to_fine", coarse_batch, fine_batch),
                           ("fine_to_coarse", fine_batch, coarse_batch)):
        e = getattr(geometry, name)
        if not torch.equal(src[e[0]], dst[e[1]]):
            raise RuntimeError(f"{name}: cross-sample edge")


def summarize_teacher_grids(records: list[dict]) -> dict[int, dict]:
    """Validate every teacher's geometry; choose one period *per GPU grid*.

    Distinct CFD cases with the same node count may have different physical
    box lengths. This GPU-only benchmark reuses a representative exact graph
    topology, never packs patches from those different physical cases.
    """
    grids: dict[int, dict] = {}
    for record in records:
        nf, nc = tuple(record["target_shape"]), tuple(record["input_shape"])
        if not (len(nf) == len(nc) == 3 and len(set(nf)) == len(set(nc)) == 1
                and nf[0] == 2 * nc[0] and nf[0] in (48, 80)):
            raise RuntimeError("unexpected q8->q4 teacher grid dimensions")
        if len(record["fine_spacing"]) != 3 or len(record["coarse_spacing"]) != 3:
            raise RuntimeError("teacher source has invalid 3D spacing metadata")
        length = float(record["fine_spacing"][0]) * nf[0]
        lengths = [float(x) * z for x, z in zip(record["fine_spacing"], nf)]
        lengths += [float(x) * z for x, z in zip(record["coarse_spacing"], nc)]
        if not math.isfinite(length) or length <= 0 or not all(
            math.isfinite(x) and math.isclose(x, length, rel_tol=1e-7, abs_tol=1e-12)
            for x in lengths
        ):
            raise RuntimeError("teacher source has inconsistent physical periodic lengths")
        grid = grids.setdefault(nf[0], {
            "fine_side": nf[0], "coarse_side": nc[0],
            "periodic_length": length,
            "representative_teacher_case": record["case"],
            "representative_snapshot_id": str(record["snapshot_id"]),
            "periodic_length_min": length, "periodic_length_max": length,
            "teacher_snapshots": 0, "case_periodic_lengths": {},
        })
        if grid["coarse_side"] != nc[0]:
            raise RuntimeError("inconsistent refinement within teacher grid family")
        grid["teacher_snapshots"] += 1
        grid["periodic_length_min"] = min(grid["periodic_length_min"], length)
        grid["periodic_length_max"] = max(grid["periodic_length_max"], length)
        grid["case_periodic_lengths"].setdefault(record["case"], []).append(length)
    if sorted(grids) != [48, 80]:
        raise RuntimeError("expected exactly the two M48 audited teacher grid families")
    for grid in grids.values():
        grid["teacher_case_count"] = len(grid["case_periodic_lengths"])
        unique_lengths = []
        for case_lengths in grid["case_periodic_lengths"].values():
            for length in case_lengths:
                if not any(math.isclose(length, x, rel_tol=1e-7, abs_tol=1e-12)
                           for x in unique_lengths):
                    unique_lengths.append(length)
        grid["distinct_periodic_lengths"] = sorted(unique_lengths)
    return grids


def authentic_teacher_grids(ms_sr_root: Path, normalization_file: Path) -> tuple[dict, dict]:
    """Read authenticated teacher *metadata*, never source or test fields."""
    from graph_attention.data.ms_sr_training import select_role_records
    from graph_attention.data.ms_sr_adapter import MANIFEST_SHA256

    frozen = json.loads(normalization_file.read_text())
    if frozen.get("manifest_sha256") != MANIFEST_SHA256 or frozen.get("source_count") != 44:
        raise RuntimeError("unexpected frozen M48 teacher normalization provenance")
    actual_manifest = sha256_file(ms_sr_root / "dataset" / "manifest.json")
    if actual_manifest != MANIFEST_SHA256:
        raise RuntimeError("MS_SR manifest does not match pinned M48 teacher manifest")
    records, seal = select_role_records(
        ms_sr_root, role="teacher", filter_id="w1p25_l1",
        case_prefix="", max_sources=0,
    )
    if len(records) != 44 or len({r["case"] for r in records}) != 22:
        raise RuntimeError("not all 44 authenticated teacher records are selected")
    if seal != frozen["selection_sha256"]:
        raise RuntimeError("frozen normalization teacher selection hash mismatch")
    grids = summarize_teacher_grids(records)
    return grids, {"manifest_sha256": actual_manifest, "selection_sha256": seal,
                   "normalization_sha256": sha256_file(normalization_file),
                   "teacher_snapshots": 44, "teacher_cases": 22,
                   "teacher_field_payloads_read": False}


def build_geometry(fine_side: int, core_side: int, depth: int,
                   mode: str, periodic_length: float):
    from graph_attention.geometry.two_scale_targeted import build_targeted_two_scale_patch
    from scripts.m48_ms_sr_train import _owned_cube

    if fine_side not in (48, 80) or core_side > fine_side or depth < 1:
        raise ValueError("unsupported M48 geometry configuration")
    # Deliberately cross the x and z periodic seams, and use an odd y origin.
    origin = (fine_side - 3, fine_side // 3 + 1, fine_side - 2)
    owned = _owned_cube(fine_side, core_side, origin)
    started = time.perf_counter()
    plan, geometry = build_targeted_two_scale_patch(
        coarse_side=fine_side // 2, refinement=2, owned_fine=owned,
        fine_pre_layers=depth, fine_post_layers=depth,
        coarse_static_layers=depth,
        coarse_dynamic_layers=depth if mode == "dynamic" else 0,
        periodic_length=periodic_length, dtype=torch.float32,
        max_fine_read=fine_side**3, max_coarse_read=(fine_side // 2)**3,
    )
    elapsed = time.perf_counter() - started
    if (not torch.equal(geometry.fine_global[geometry.owned_fine_local], owned)
            or owned.numel() != core_side**3):
        raise RuntimeError("owned core/global mapping differs from exact-halo geometry")
    report = {
        "fine_side": fine_side, "coarse_side": fine_side // 2,
        "mode": mode, "owned_core_side": core_side, "origin": list(origin),
        "fine_read_nodes": int(len(plan.fine_read)),
        "coarse_read_nodes": int(len(plan.coarse_read)),
        "owned_nodes_per_patch": int(len(owned)),
        "read_to_owned_ratio": len(plan.fine_read) / len(owned),
        "fine_local_edges": int(geometry.fine_edges.shape[1]),
        "fine_dilated_edges": int(geometry.fine_dilated.shape[1]),
        "coarse_local_edges": int(geometry.coarse_edges.shape[1]),
        "coarse_dilated_edges": int(geometry.coarse_dilated.shape[1]),
        "cross_scale_edges": int(geometry.coarse_to_fine.shape[1]),
        "single_graph_tensor_bytes": graph_tensor_bytes(geometry),
        "geometry_build_seconds": elapsed,
        "periodic_length": periodic_length,
    }
    return geometry, report


def pack_synthetic_inputs(geometry, microbatch: int, seed: int, device: torch.device):
    from scripts.m48_ms_sr_train import _geometry_device
    g, fine_idx, coarse_idx = pack_repeated_geometry(geometry, microbatch)
    # Validate topology before moving to GPU; no CEDAR, no real CFD fields.
    verify_packed_graph(g, fine_idx, coarse_idx)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    def normal(n: int):
        return torch.randn((n, 5), generator=gen, dtype=torch.float32)
    state = normal(len(g.fine_global))
    baseline = normal(len(g.fine_global))
    coarse = normal(len(g.coarse_global))
    target = normal(len(g.owned_fine_local))
    fm_time = (0.05 + 0.9 * torch.rand((microbatch,), generator=gen))
    start = time.perf_counter()
    result = {
        "geometry": _geometry_device(g, device), "fine_batch_index": fine_idx.to(device),
        "coarse_batch_index": coarse_idx.to(device),
        "state": state.to(device), "baseline": baseline.to(device),
        "coarse_state": coarse.to(device), "target": target.to(device),
        "time": fm_time.to(device),
    }
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


class NvmlSampler:
    """Optional low-overhead NVML utilization samples; never a test dependency."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.1):
        self.samples: list[tuple[int, int]] = []
        self.error = None
        self._stop = threading.Event()
        self._thread = None
        self._lib = None
        self._device = ctypes.c_void_p()
        self.interval_s = interval_s
        try:
            library = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
            lib = ctypes.CDLL(library)
            lib.nvmlInit_v2.restype = ctypes.c_int
            if lib.nvmlInit_v2() != 0:
                raise RuntimeError("NVML init rejected")
            get_handle = getattr(lib, "nvmlDeviceGetHandleByIndex_v2", None)
            if get_handle is None:
                get_handle = lib.nvmlDeviceGetHandleByIndex
            get_handle.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            get_handle.restype = ctypes.c_int
            if get_handle(device_index, ctypes.byref(self._device)) != 0:
                raise RuntimeError("NVML GPU selection rejected")
            lib.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
            self._lib = lib
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._lib = None

    def __enter__(self):
        if self._lib is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def _poll(self):
        class Utilization(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
        while not self._stop.is_set():
            data = Utilization()
            rc = self._lib.nvmlDeviceGetUtilizationRates(self._device, ctypes.byref(data))
            if rc == 0:
                self.samples.append((int(data.gpu), int(data.memory)))
            else:
                self.error = f"NVML utilization query error {rc}"
            self._stop.wait(self.interval_s)

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._lib is not None:
            self._lib.nvmlShutdown()

    def report(self):
        return {
            "nvml_mean_gpu_utilization_percent": (
                statistics.mean(x for x, _ in self.samples) if self.samples else None
            ),
            "nvml_mean_memory_utilization_percent": (
                statistics.mean(y for _, y in self.samples) if self.samples else None
            ),
            "nvml_sample_count": len(self.samples),
            "nvml_error": self.error,
        }


def _gpu_trial(args) -> dict:
    """One CUDA worker process; returns one JSON receipt. Never shares context."""
    if not torch.cuda.is_available():
        raise RuntimeError("M48.4e worker requires CUDA, not CPU emulation")
    from graph_attention.geometry.two_scale_partition import TwoScalePatchGeometry
    from graph_attention.models.two_scale_dinat import TwoScaleDiNAT
    from scripts.m48_ms_sr_train import _forward_args

    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    free_initial, _ = torch.cuda.mem_get_info(device)
    cache = torch.load(args.geometry_cache, map_location="cpu", weights_only=True)
    geom = TwoScalePatchGeometry(**cache["geometry"])
    report = cache["report"]
    baseline_meta = {"mode": args.mode, "fine_side": args.fine_side,
                     "owned_core_side": args.core_side, "gpu_microbatch": args.microbatch,
                     "network_width": args.width, "network_depth": args.depth,
                     "heads": args.heads, "seed": args.seed,
                     "cuda_device": torch.cuda.get_device_name(device),
                     "gpu_total_memory_bytes": total, "gpu_free_initial_bytes": free_initial,
                     "allowed_fraction": args.max_gpu_fraction,
                     "owned_nodes_per_patch": report["owned_nodes_per_patch"],
                     "fine_read_nodes_per_patch": report["fine_read_nodes"],
                     "coarse_read_nodes_per_patch": report["coarse_read_nodes"],
                     "edge_counts_per_patch": {k: report[k] for k in (
                         "fine_local_edges", "fine_dilated_edges", "coarse_local_edges",
                         "coarse_dilated_edges", "cross_scale_edges")},
                     "synthetic_inputs_only": True, "optimizer": "AdamW",
                     "precision": "fp32", "warmup_steps": args.warmup,
                     "measured_steps": args.measure}
    cost = report["single_graph_tensor_bytes"] * args.microbatch
    # CPU serialization + packing transient copies; conservative local guard.
    if cost * args.host_overhead_multiplier > args.max_host_pack_gib * 2**30:
        return {**baseline_meta, "status": "host_memory_budget_exceeded",
                "estimated_host_pack_bytes": int(cost * args.host_overhead_multiplier)}

    torch.manual_seed(args.seed)
    model = TwoScaleDiNAT(
        mode=args.mode, fine_width=args.width, coarse_width=args.width,
        num_heads=args.heads, fine_pre_layers=args.depth,
        fine_post_layers=args.depth, coarse_static_layers=args.depth,
        coarse_dynamic_layers=args.depth,
    ).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch, transfer_sec = pack_synthetic_inputs(geom, args.microbatch, args.seed + 1, device)

    def one_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["state"], baseline=batch["baseline"],
                     coarse_state=batch["coarse_state"], time=batch["time"],
                     fine_batch_index=batch["fine_batch_index"],
                     coarse_batch_index=batch["coarse_batch_index"],
                     **_forward_args(batch["geometry"]))
        squared = (pred[batch["geometry"].owned_fine_local] - batch["target"]).square()
        # All owned cubes have identical volume, so the full mean equals the
        # real training path's equal-patch-mean objective (without CPU splits).
        loss = squared.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss

    for _ in range(args.warmup):
        one_step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    # Timed loop includes forward, backward, optimizer, zero_grad, CUDA launch
    # overhead, and explicit final synchronization. Excludes graph construction,
    # CUDA transfer, CEDAR, dataset IO, and checkpointing.
    durations_wall, durations_cuda, losses = [], [], []
    with NvmlSampler(device_index=0) as nvml:
        for _ in range(args.measure):
            torch.cuda.synchronize(device)
            ev_start, ev_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            ev_start.record()
            loss = one_step()
            ev_end.record()
            torch.cuda.synchronize(device)
            durations_wall.append(time.perf_counter() - start)
            durations_cuda.append(ev_start.elapsed_time(ev_end) / 1000.0)
            losses.append(float(loss.detach().item()))
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError("nonfinite CUDA training loss")
    owned = args.microbatch * args.core_side**3
    mean_wall = statistics.mean(durations_wall)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    budget = min(total, free_initial) * args.max_gpu_fraction
    return {**baseline_meta,
            "status": "ok" if peak_reserved <= budget else "gpu_headroom_exceeded",
            "gpu_safe_budget_bytes": int(budget),
            "peak_allocated_bytes": int(peak_allocated),
            "peak_reserved_bytes": int(peak_reserved),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "gpu_transfer_and_packing_seconds": transfer_sec,
            "owned_nodes_per_batch": owned,
            "fine_read_nodes_per_batch": args.microbatch * report["fine_read_nodes"],
            "coarse_read_nodes_per_batch": args.microbatch * report["coarse_read_nodes"],
            "total_graph_edges_per_batch": args.microbatch * (
                report["fine_local_edges"] + report["fine_dilated_edges"] +
                report["coarse_local_edges"] + report["coarse_dilated_edges"] +
                2 * report["cross_scale_edges"]),
            "timed_update_seconds": durations_wall,
            "cuda_event_seconds": durations_cuda,
            "mean_update_seconds": mean_wall,
            "median_update_seconds": statistics.median(durations_wall),
            "mean_cuda_event_seconds": statistics.mean(durations_cuda),
            "owned_nodes_per_second": owned / mean_wall,
            "loss_first_last": [losses[0], losses[-1]],
            **nvml.report()}


def run_worker(args) -> None:
    try:
        outcome = _gpu_trial(args)
    except torch.cuda.OutOfMemoryError as exc:
        outcome = {"status": "cuda_oom", "error": str(exc)[:2000]}
    except MemoryError as exc:
        outcome = {"status": "host_memory_error", "error": str(exc)[:2000]}
    except Exception as exc:
        outcome = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-4000:]}
        write_json(args.worker_output, outcome)
        raise
    write_json(args.worker_output, outcome)


def _classify_worker(returncode: int, stdout: str, stderr: str, receipt: dict | None) -> dict:
    if receipt is not None:
        return receipt
    return {"status": "worker_crash", "returncode": returncode,
            "stderr_tail": stderr[-2400:], "stdout_tail": stdout[-800:]}


def _best_configurations(trials: list[dict]) -> list[dict]:
    """Best observed throughput within budget, separately for each mesh/core."""
    grouped = {}
    for row in trials:
        if row.get("status") != "ok":
            continue
        key = (row["mode"], row["fine_side"], row["owned_core_side"])
        if key not in grouped or row["owned_nodes_per_second"] > grouped[key]["owned_nodes_per_second"]:
            grouped[key] = row
    return [{key: row[key] for key in (
                "mode", "fine_side", "owned_core_side", "gpu_microbatch",
                "network_width", "network_depth", "owned_nodes_per_second",
                "peak_allocated_bytes", "peak_reserved_bytes", "gpu_total_memory_bytes")}
            for _, row in sorted(grouped.items())]


def _subprocess_command(args, grid: dict, core: int, micro: int, cache: Path,
                        output: Path) -> list[str]:
    return [sys.executable, "-m", "scripts.m48_4e_gpu_capacity", "--worker",
            "--mode", args.mode, "--fine-side", str(grid["fine_side"]),
            "--core-side", str(core), "--microbatch", str(micro),
            "--width", str(args.width), "--depth", str(args.depth),
            "--heads", str(args.heads), "--warmup", str(args.warmup),
            "--measure", str(args.measure), "--seed", str(args.seed),
            "--max-gpu-fraction", str(args.max_gpu_fraction),
            "--max-host-pack-gib", str(args.max_host_pack_gib),
            "--host-overhead-multiplier", str(args.host_overhead_multiplier),
            "--geometry-cache", str(cache), "--worker-output", str(output)]


def run_sweep(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU capacity sweep requires a CUDA device on Calypso")
    if (args.mode not in ("static", "dynamic") or args.depth < 1
            or args.width <= 0 or args.heads <= 0 or args.width % args.heads
            or args.warmup < 1 or args.measure < 2 or args.seed < 0
            or not (0 < args.max_gpu_fraction < 1)
            or args.max_host_pack_gib <= 0 or args.host_overhead_multiplier < 1
            or args.worker_timeout_seconds < 1):
        raise ValueError("invalid M48.4e GPU sweep configuration")
    sides, cores, batches = (positive_ints(args.fine_sides),
                             positive_ints(args.core_sides),
                             positive_ints(args.microbatches))
    if any(x not in (48, 80) for x in sides) or any(p > min(sides) for p in cores):
        raise ValueError("only current authenticated fine grids and valid cores supported")
    if args.output_dir.exists():
        raise FileExistsError("M48.4e refuses to overwrite or resume benchmark output")
    grids, provenance = authentic_teacher_grids(args.ms_sr_root, args.normalization_file)
    args.output_dir.mkdir(parents=True)
    run_info = {"schema": SCHEMA, "status": "RUNNING", "phase": "A_gpu_only",
                "no_training_performed": True,
                "synthetic_feature_inputs": True,
                "no_cedar_or_dns_field_evaluation": True,
                "provenance": provenance,
                "architecture": {"mode": args.mode, "width": args.width,
                                 "depth_each_stage": args.depth, "heads": args.heads},
                "device": {"name": torch.cuda.get_device_name(0),
                           "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
                "hyperparameters": {"fine_sides": sides, "core_sides": cores,
                                    "microbatches": batches, "warmup": args.warmup,
                                    "measure": args.measure, "seed": args.seed,
                                    "gpu_fraction": args.max_gpu_fraction},
                "trials": [], "best_safe_per_grid_core": []}
    import subprocess as _subprocess
    run_info["git_commit"] = _subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(args.output_dir / "summary.json", run_info)
    for side in sides:
        for core in cores:
            cache_path = args.output_dir / "geometry" / f"{args.mode}_n{side}_p{core}.pt"
            report_path = args.output_dir / "geometry" / f"{args.mode}_n{side}_p{core}.json"
            try:
                geom, report = build_geometry(side, core, args.depth, args.mode,
                                              grids[side]["periodic_length"])
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"geometry": {f.name: getattr(geom, f.name) for f in fields(geom)},
                            "report": report}, cache_path)
                write_json(report_path, report)
                del geom
            except Exception as exc:
                row = {"mode": args.mode, "fine_side": side, "owned_core_side": core,
                       "status": "geometry_error", "error": f"{type(exc).__name__}: {exc}"}
                run_info["trials"].append(row)
                write_json(args.output_dir / "summary.json", run_info)
                continue
            for micro in batches:
                trial_id = f"{args.mode}_n{side}_p{core}_b{micro}"
                path = args.output_dir / "trials" / f"{trial_id}.json"
                log = args.output_dir / "trials" / f"{trial_id}.log"
                started = time.perf_counter()
                cmd = _subprocess_command(args, grids[side], core, micro, cache_path, path)
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True,
                                          timeout=args.worker_timeout_seconds, check=False)
                    row = _classify_worker(proc.returncode, proc.stdout, proc.stderr,
                                           json.loads(path.read_text()) if path.is_file() else None)
                    log.write_text(proc.stdout + "\n=== stderr ===\n" + proc.stderr)
                    if proc.returncode and row["status"] == "ok":
                        row["status"] = "worker_exit_nonzero"
                        row["returncode"] = proc.returncode
                except subprocess.TimeoutExpired as exc:
                    row = {"status": "worker_timeout", "limit_seconds": args.worker_timeout_seconds}
                    log.write_text("Worker timed out.\n" + str(exc))
                row.update({"trial_id": trial_id, "mode": args.mode, "fine_side": side,
                            "owned_core_side": core, "gpu_microbatch": micro,
                            "worker_wall_seconds": time.perf_counter() - started})
                write_json(path, row)
                run_info["trials"].append(row)
                run_info["best_safe_per_grid_core"] = _best_configurations(run_info["trials"])
                write_json(args.output_dir / "summary.json", run_info)
                print(f"{trial_id}: {row['status']}"
                      + (f" {row['owned_nodes_per_second']:.0f} owned nodes/s"
                         if row["status"] == "ok" else ""), flush=True)
                if row["status"] != "ok":
                    # Larger batches are not qualified after a failed capacity
                    # trial; independently continue at the next core/grid.
                    break
    qualified = {(r["fine_side"], r["owned_core_side"]) for r in run_info["trials"]
                 if r.get("status") == "ok"}
    expected = {(side, core) for side in sides for core in cores}
    run_info["unqualified_grid_core_pairs"] = [
        {"fine_side": side, "owned_core_side": core}
        for side, core in sorted(expected - qualified)
    ]
    unexpected = {"error", "worker_crash", "worker_timeout", "worker_exit_nonzero",
                  "geometry_error", "host_memory_error"}
    if any(row.get("status") in unexpected for row in run_info["trials"]):
        run_info["status"] = "ERROR"
    else:
        run_info["status"] = "FINISHED" if qualified == expected else "INCOMPLETE"
    write_json(args.output_dir / "summary.json", run_info)
    with (args.output_dir / "trials.csv").open("w", newline="") as fh:
        cols = ["trial_id", "mode", "fine_side", "owned_core_side", "gpu_microbatch",
                "status", "owned_nodes_per_second", "mean_update_seconds",
                "peak_allocated_bytes", "peak_reserved_bytes", "gpu_total_memory_bytes",
                "fine_read_nodes_per_batch", "coarse_read_nodes_per_batch",
                "total_graph_edges_per_batch", "worker_wall_seconds"]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in run_info["trials"]:
            writer.writerow({key: row.get(key) for key in cols})
    return run_info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--metadata-only", action="store_true",
                        help="Verify teacher geometry without requesting CUDA or writing output")
    parser.add_argument("--mode", required=True, choices=("static", "dynamic"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ms-sr-root", type=Path, default=Path("/scratch/ai/MS_SR"))
    parser.add_argument("--normalization-file", type=Path,
                        default=Path("configs/m48/teacher_44_w1p25_l1_q4_q8.json"))
    parser.add_argument("--fine-sides", default="48,80")
    parser.add_argument("--core-sides", default="12,16,20,24")
    parser.add_argument("--microbatches", default="1,2,4,8,16,32,64,128,256")
    # Width 64/depth 2 is an explicitly tunable *engineering candidate*;
    # final production architecture has not yet been scientifically selected.
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measure", type=int, default=8)
    parser.add_argument("--seed", type=int, default=484)
    parser.add_argument("--max-gpu-fraction", type=float, default=0.85)
    parser.add_argument("--max-host-pack-gib", type=float, default=8.0)
    parser.add_argument("--host-overhead-multiplier", type=float, default=3.0)
    parser.add_argument("--worker-timeout-seconds", type=int, default=300)
    parser.add_argument("--geometry-cache", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--fine-side", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--core-side", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--microbatch", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.metadata_only:
        if args.worker:
            parser.error("--worker and --metadata-only are mutually exclusive")
        grids, provenance = authentic_teacher_grids(args.ms_sr_root, args.normalization_file)
        print(json.dumps({"schema": "graph_attention.m48_4e_teacher_geometry.v1",
                          "status": "PASS", "grids": grids, "provenance": provenance},
                         indent=2), flush=True)
        return
    if args.worker:
        if (args.geometry_cache is None or args.worker_output is None
                or args.fine_side is None or args.core_side is None or args.microbatch is None):
            parser.error("worker requires geometry cache, output, fine side, core side and microbatch")
        run_worker(args)
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required")
        result = run_sweep(args)
        print(json.dumps({"status": result["status"],
                          "trial_count": len(result["trials"]),
                          "safe_configurations": len(result["best_safe_per_grid_core"]),
                          "output": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
