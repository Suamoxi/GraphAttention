"""M48.4e A2: GPU comparison of full-grid vs complete exact-halo tiled updates.

Benchmark only. Synthetic standardized node fields, shared global FM time,
no CEDAR or DNS field reads. Isolated CUDA workers and fail-closed parity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from itertools import product
import statistics
import subprocess
import sys
import time
import traceback

import torch

from scripts.m48_4e_gpu_capacity import (
    authentic_teacher_grids, graph_tensor_bytes, positive_ints, write_json,
)

SCHEMA = "graph_attention.m48_4e_a2_full_vs_tiled.v1"
ACTIVE_INIT = "nonzero_dit_attention_v1"
VERIFICATION_STEPS = 2


class HostGraphBudgetExceeded(RuntimeError):
    """Bounded exclusion, not an unexpected worker crash."""


def activate_attention_paths(model) -> None:
    """Make every DiT attention block and final decoder output active.

    The production model is unchanged. Its DiT output head and adaLN gates
    are zero-initialized, which makes pre-training output parity vacuous.
    This deterministic *benchmark-only* activation enables full graph
    dependence without altering the underlying attention implementation.
    """
    stages = [model.fine_encoder, model.coarse_encoder, model.fine_decoder]
    if model.mode == "dynamic":
        stages.append(model.coarse_dynamic)
    with torch.no_grad():
        for stage in stages:
            for block in stage.blocks:
                torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
                torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)
        torch.nn.init.normal_(model.fine_decoder.final_layer.projection.weight, std=0.02)
        torch.nn.init.zeros_(model.fine_decoder.final_layer.projection.bias)


def required_gradient_groups(mode: str) -> tuple[str, ...]:
    groups = ["fine_encoder.attention", "coarse_encoder.attention",
              "fine_decoder.attention", "up"]
    if mode == "dynamic":
        groups.extend(["coarse_dynamic.attention", "down"])
    return tuple(groups)


def gradient_group_norms(parameters: list[tuple[str, int]],
                         flat_gradient: torch.Tensor, mode: str) -> dict[str, float]:
    """Derive active path norms directly from saved parameter-order gradients."""
    squared = {group: 0.0 for group in required_gradient_groups(mode)}
    offset = 0
    for name, count in parameters:
        gradient = flat_gradient[offset:offset + count]
        if gradient.numel() != count:
            raise ValueError("gradient parameter layout mismatch")
        offset += count
        group = name.split(".", 1)[0]
        if ".attention." in name:
            group += ".attention"
        if group in squared:
            squared[group] += float(gradient.to(torch.float64).square().sum())
    if offset != flat_gradient.numel():
        raise ValueError("gradient parameter count mismatch")
    return {name: math.sqrt(total) for name, total in squared.items()}



def tile_origins(side: int, core: int, max_tiles: int) -> list[tuple[int, int, int]]:
    """Nonoverlapping periodic owned cubes, each fine node owned exactly once."""
    if (side < 6 or core < 1 or side % core != 0 or max_tiles < 1):
        raise ValueError("full-coverage tiling requires a positive divisor of fine side")
    count = (side // core) ** 3
    if count > max_tiles:
        raise ValueError(f"full coverage needs {count} tiles, cap is {max_tiles}")
    return list(product(range(0, side, core), repeat=3))


def verify_ownership(side: int, geometries: list) -> None:
    """Check exact once-only loss ownership before CUDA evaluation."""
    counts = torch.zeros(side ** 3, dtype=torch.int32)
    for geom in geometries:
        owned = geom.fine_global[geom.owned_fine_local]
        if (not torch.equal(owned, torch.unique(owned)) or
                int(owned.min()) < 0 or int(owned.max()) >= side ** 3):
            raise RuntimeError("invalid owned fine indices")
        counts.index_add_(0, owned, torch.ones_like(owned, dtype=torch.int32))
    if not torch.all(counts == 1):
        raise RuntimeError("tiled loss does not own every global fine node exactly once")


def full_geometry(side: int, period: float):
    """Full two-scale periodic graph using the same M48 targeted builder.

    Zero dependency layers are intentional: with every fine node owned, the
    complete fine/coarse read sets and both cross-scale supports are present.
    The *network* still uses the requested nonzero depth.
    """
    from graph_attention.geometry.two_scale_targeted import build_targeted_two_scale_patch
    _, geom = build_targeted_two_scale_patch(
        coarse_side=side // 2, refinement=2,
        owned_fine=torch.arange(side**3, dtype=torch.long),
        fine_pre_layers=0, fine_post_layers=0,
        coarse_static_layers=0, coarse_dynamic_layers=0,
        periodic_length=period, max_fine_read=side**3,
        max_coarse_read=(side//2)**3,
    )
    if (len(geom.fine_global) != side**3 or
            len(geom.coarse_global) != (side//2)**3 or
            not torch.equal(geom.fine_global, torch.arange(side**3)) or
            not torch.equal(geom.coarse_global, torch.arange((side//2)**3))):
        raise RuntimeError("full geometry failed all-node support checks")
    return geom


def make_geometries(side: int, core: int, depth: int, mode: str,
                    period: float, max_tiles: int, strategy: str,
                    *, max_host_graph_bytes: int | None = None):
    from scripts.m48_4e_gpu_capacity import build_geometry
    started = time.perf_counter()
    if strategy == "full":
        geometries = [full_geometry(side, period)]
    elif strategy == "tiled":
        geometries = []
        cumulative = 0
        for origin in tile_origins(side, core, max_tiles):
            geom, _ = build_geometry(side, core, depth, mode, period, origin=origin)
            cumulative += graph_tensor_bytes(geom)
            if max_host_graph_bytes is not None and cumulative > max_host_graph_bytes:
                raise HostGraphBudgetExceeded(
                    f"tile geometry exceeds host cap after {len(geometries) + 1} tiles: "
                    f"{cumulative} > {max_host_graph_bytes} bytes")
            geometries.append(geom)
        verify_ownership(side, geometries)
    else:
        raise ValueError("strategy must be full or tiled")
    if (max_host_graph_bytes is not None
            and sum(graph_tensor_bytes(g) for g in geometries) > max_host_graph_bytes):
        raise HostGraphBudgetExceeded("full geometry exceeds host cap")
    return geometries, time.perf_counter() - started


def validate_48_parity_gate(path: Path | None, *, mode: str, depths: list[int],
                            width: int, heads: int, provenance: dict,
                            current_commit: str) -> dict:
    """Refuse 80^3 comparisons until matching 48^3 active parity passed."""
    if path is None or not path.is_file():
        raise ValueError("80^3 requires --parity-gate-summary from a passed active 48^3 run")
    report = json.loads(path.read_text())
    requested = report.get("requested", {})
    checks = report.get("comparisons", [])
    expected = {(48, d) for d in depths}
    qualified = {(row.get("fine_side"), row.get("depth")) for row in checks
                 if row.get("status") == "PASS"
                 and row.get("parity", {}).get("passed")
                 and len(row.get("parity", {}).get("steps", [])) == VERIFICATION_STEPS
                 and all(step.get("active_attention_gradients") and step.get("passed")
                         for step in row["parity"]["steps"])}
    valid = (report.get("schema") == SCHEMA and report.get("status") == "PASS"
             and report.get("active_initialization") == ACTIVE_INIT
             and report.get("required_verification_steps") == VERIFICATION_STEPS
             and report.get("source_commit") == current_commit
             and report.get("provenance") == provenance
             and requested.get("mode") == mode
             and requested.get("width") == width and requested.get("heads") == heads
             and expected <= qualified)
    if not valid:
        raise ValueError("48^3 active prediction/gradient parity gate is missing or mismatched")
    return {"status": "PASS", "source_path": str(path),
            "source_commit": current_commit, "qualified_depths": sorted(depths)}


def compare_arrays(reference: torch.Tensor, candidate: torch.Tensor,
                   *, atol: float, rtol: float) -> dict:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {"passed": False, "error": "shape or dtype mismatch"}
    difference = (reference - candidate).to(torch.float64)
    ref = reference.to(torch.float64)
    return {
        "passed": bool(torch.allclose(reference, candidate, rtol=rtol, atol=atol)),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(difference) /
                             torch.linalg.vector_norm(ref).clamp_min(1e-30)),
        "atol": atol, "rtol": rtol,
    }


def compare_verification(ref_path: Path, tile_path: Path) -> dict:
    """Fail-closed active-network parity, including after one AdamW update."""
    full = torch.load(ref_path, map_location="cpu", weights_only=True)
    tiled = torch.load(tile_path, map_location="cpu", weights_only=True)
    if (full.get("initialization") != ACTIVE_INIT
            or tiled.get("initialization") != ACTIVE_INIT):
        return {"passed": False, "error": "zero-head or legacy parity not qualified"}
    if (full.get("seed") != tiled.get("seed")
            or full.get("mode") != tiled.get("mode")
            or len(full.get("steps", [])) != VERIFICATION_STEPS
            or len(tiled.get("steps", [])) != VERIFICATION_STEPS):
        raise RuntimeError("different synthetic seeds, modes or verification step counts")
    required = required_gradient_groups(full["mode"])
    checked = []
    for index, (a, b) in enumerate(zip(full["steps"], tiled["steps"])):
        if a["parameters"] != b["parameters"]:
            raise RuntimeError("different model parameter layouts")
        pred = compare_arrays(a["prediction"], b["prediction"],
                              rtol=5e-4, atol=5e-4)
        grad = compare_arrays(a["gradient"], b["gradient"],
                              rtol=1e-2, atol=1e-4)
        # Absolute-only allclose can silently accept tiny inactive gradients.
        active_norms = []
        for step in (a, b):
            measured = gradient_group_norms(
                step["parameters"], step["gradient"], full["mode"])
            rms = float(step["prediction"].square().mean().sqrt())
            spatial_std = float(step["prediction"].std(dim=0).mean())
            active_norms.append((measured, rms, spatial_std))
        active = all(
            rms > 1e-5 and spatial_std > 1e-5
            and math.isfinite(rms) and math.isfinite(spatial_std)
            and all(math.isfinite(groups[group]) and groups[group] > 1e-12
                    for group in required)
            for groups, rms, spatial_std in active_norms)
        # Check both global and individual attention-path relative errors.
        grad_rel = grad.get("relative_l2", math.inf)
        gradients_well_matched = bool(grad["passed"] and grad_rel < 2e-3)
        passed = bool(active and pred["passed"] and pred.get("relative_l2", math.inf) < 1e-3
                      and gradients_well_matched)
        checked.append({"step": index, "passed": passed,
                        "active_attention_gradients": active,
                        "prediction": pred, "gradient": grad,
                        "full_gradient_group_norms": active_norms[0][0],
                        "tiled_gradient_group_norms": active_norms[1][0],
                        "full_prediction_rms": active_norms[0][1],
                        "tiled_prediction_rms": active_norms[1][1]})
    return {"passed": all(step["passed"] for step in checked),
            "initialization": ACTIVE_INIT, "steps": checked,
            "prediction": checked[0]["prediction"],
            "gradient": checked[0]["gradient"]}


def make_global_inputs(side: int, seed: int, device: torch.device) -> dict:
    """Identical entire-domain inputs across isolated worker strategies."""
    rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    def sample(nodes):
        return torch.randn((nodes, 5), generator=rng, dtype=torch.float32).to(device)
    return {"state": sample(side**3), "baseline": sample(side**3),
            "coarse_state": sample((side//2)**3), "target": sample(side**3),
            "time": torch.tensor([0.43], device=device, dtype=torch.float32)}


def gpu_step(model, optimizer, global_input: dict, geometries: list,
             strategy: str, *, capture: bool = False,
             stream_geometry: bool = False):
    from scripts.m48_ms_sr_train import _forward_args, _geometry_device
    device = global_input["state"].device
    n = global_input["state"].shape[0]
    optimizer.zero_grad(set_to_none=True)
    prediction = torch.empty((n, 5), dtype=torch.float32) if capture else None
    objective = None
    for original_geometry in geometries:
        geom = (_geometry_device(original_geometry, device)
                if stream_geometry else original_geometry)
        if strategy == "full":
            fine = global_input["state"]
            baseline = global_input["baseline"]
            coarse = global_input["coarse_state"]
        else:
            fine = global_input["state"][geom.fine_global]
            baseline = global_input["baseline"][geom.fine_global]
            coarse = global_input["coarse_state"][geom.coarse_global]
        pred = model(fine, baseline=baseline, coarse_state=coarse,
                     time=global_input["time"], **_forward_args(geom))
        owned_local = geom.owned_fine_local
        owned_global = geom.fine_global[owned_local]
        pred_owned = pred[owned_local]
        loss = (pred_owned - global_input["target"][owned_global]).square().mean()
        # One loss over every fine-grid node, regardless of tile size.
        weighted = loss * (owned_global.numel() / n)
        weighted.backward()
        if capture:
            prediction[owned_global.cpu()] = pred_owned.detach().cpu()
        objective = weighted.detach() if objective is None else objective + weighted.detach()
        del pred, pred_owned, fine, coarse, baseline, loss, weighted, geom

    verification = None
    if capture:
        names = [(name, p.numel()) for name, p in model.named_parameters()]
        grad = torch.cat([p.grad.detach().reshape(-1).cpu() if p.grad is not None
                          else torch.zeros(p.numel(), dtype=torch.float32)
                          for p in model.parameters()])
        group_norms = gradient_group_norms(names, grad, model.mode)
        spatial_std = prediction.std(dim=0).mean().item()
        verification = {"prediction": prediction, "gradient": grad,
                        "parameters": names,
                        "prediction_rms": float(prediction.square().mean().sqrt().item()),
                        "prediction_spatial_std": float(spatial_std),
                        "gradient_group_norms": group_norms}
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return objective, verification


def _gpu_worker(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("A2 requires CUDA; never substitute a CPU timing")
    from graph_attention.models.two_scale_dinat import TwoScaleDiNAT
    from scripts.m48_ms_sr_train import _geometry_device

    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("highest")
    total = int(torch.cuda.get_device_properties(device).total_memory)
    initial_free = int(torch.cuda.mem_get_info(device)[0])
    budget = int(min(total, initial_free) * args.max_gpu_fraction)
    torch.cuda.reset_peak_memory_stats(device)
    residency = ("cpu_streamed" if args.strategy == "tiled"
                 and (args.tiled_geometry_residency == "cpu"
                      or (args.tiled_geometry_residency == "auto" and args.fine_side >= 80))
                 else "gpu_cached")
    try:
        geom_cpu, build_seconds = make_geometries(
            args.fine_side, args.core_side, args.depth, args.mode,
            args.periodic_length, args.max_tiles, args.strategy,
            max_host_graph_bytes=int(args.max_host_graph_gib * 2**30))
    except HostGraphBudgetExceeded as exc:
        return {"status": "host_graph_budget_exceeded", "error": str(exc),
                "geometry_residency": residency}
    host_geom_bytes = sum(graph_tensor_bytes(g) for g in geom_cpu)
    if host_geom_bytes > args.max_host_graph_gib * (2**30):
        return {"status": "host_graph_budget_exceeded",
                "host_graph_bytes": host_geom_bytes,
                "geometry_build_seconds": build_seconds}
    start = time.perf_counter()
    if residency == "cpu_streamed":
        # Hold topology on host and transfer one tile at a time per update.
        # Prevents N_tile * graph_bytes from becoming permanently GPU-resident.
        geom_gpu = geom_cpu
    else:
        geom_gpu = [_geometry_device(g, device) for g in geom_cpu]
        del geom_cpu
    torch.cuda.synchronize(device)
    transfer_seconds = time.perf_counter() - start
    if torch.cuda.memory_reserved(device) > budget:
        return {"status": "gpu_headroom_exceeded_at_preload",
                "peak_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "gpu_safe_budget_bytes": budget}

    torch.manual_seed(args.seed)
    model = TwoScaleDiNAT(
        mode=args.mode, fine_width=args.width, coarse_width=args.width,
        num_heads=args.heads, fine_pre_layers=args.depth,
        fine_post_layers=args.depth, coarse_static_layers=args.depth,
        coarse_dynamic_layers=args.depth,
    ).to(device).train()
    activate_attention_paths(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    inputs = make_global_inputs(args.fine_side, args.seed, device)
    geometry_details = {
        "strategy": args.strategy, "mode": args.mode, "fine_side": args.fine_side,
        "core_side": args.core_side, "depth": args.depth, "width": args.width,
        "heads": args.heads, "tiles": len(geom_gpu),
        "geometry_build_seconds": build_seconds,
        "geometry_transfer_seconds": transfer_seconds,
        "host_graph_bytes": host_geom_bytes, "gpu_safe_budget_bytes": budget,
        "gpu_total_memory_bytes": total, "gpu_initial_free_bytes": initial_free,
        "synthetic_inputs_only": True, "no_cedar_or_dns_fields": True,
        "identical_global_fm_time": True,
        "geometry_residency": residency,
        "streamed_geometry_transfer_in_update_time": residency == "cpu_streamed",
        "initialization": ACTIVE_INIT,
    }

    first_loss = None
    captured_steps = []
    for i in range(args.warmup):
        loss, verification = gpu_step(model, optimizer, inputs, geom_gpu,
                                      args.strategy, capture=i < VERIFICATION_STEPS,
                                      stream_geometry=residency == "cpu_streamed")
        if i == 0:
            first_loss = float(loss.item())
        if verification is not None:
            captured_steps.append(verification)
        torch.cuda.synchronize(device)
        if torch.cuda.max_memory_reserved(device) > budget:
            return {**geometry_details, "status": "gpu_headroom_exceeded",
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}

    args.verification_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"initialization": ACTIVE_INIT, "mode": args.mode,
                "seed": args.seed, "steps": captured_steps},
               args.verification_output)
    times = []
    cuda_times = []
    losses = []
    for _ in range(args.measure):
        torch.cuda.synchronize(device)
        start_event, end_event = (torch.cuda.Event(enable_timing=True),
                                  torch.cuda.Event(enable_timing=True))
        t0 = time.perf_counter()
        start_event.record()
        loss, _ = gpu_step(model, optimizer, inputs, geom_gpu, args.strategy,
                           stream_geometry=residency == "cpu_streamed")
        end_event.record()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
        cuda_times.append(start_event.elapsed_time(end_event) / 1e3)
        losses.append(float(loss.item()))
        if not math.isfinite(losses[-1]):
            raise RuntimeError("nonfinite synthetic MSE")
        if torch.cuda.max_memory_reserved(device) > budget:
            return {**geometry_details, "status": "gpu_headroom_exceeded",
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
    mean_sec = statistics.mean(times)
    return {**geometry_details, "status": "ok",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "warmup_steps": args.warmup, "measured_updates": args.measure,
            "initial_loss": first_loss,
            "losses": losses, "wall_update_seconds": times,
            "cuda_update_seconds": cuda_times,
            "mean_wall_update_seconds": mean_sec,
            "median_wall_update_seconds": statistics.median(times),
            "mean_cuda_update_seconds": statistics.mean(cuda_times),
            "supervised_nodes_per_update": args.fine_side**3,
            "supervised_nodes_per_second": args.fine_side**3 / mean_sec}


def _worker(args) -> None:
    try:
        result = _gpu_worker(args)
    except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
        result = {"status": "cuda_or_host_oom", "error": str(exc)[:1500]}
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()[-4000:]}
    write_json(args.worker_output, result)
    if result["status"] == "error":
        raise RuntimeError(result["error"])


def _run_worker(args, grid: dict, depth: int, core: int, strategy: str,
                output_dir: Path) -> dict:
    trial_id = f"{args.mode}_n{grid['fine_side']}_d{depth}_p{core}_{strategy}"
    result_file = output_dir / "trials" / f"{trial_id}.json"
    verification = output_dir / "verification" / f"{trial_id}.pt"
    log_file = output_dir / "trials" / f"{trial_id}.log"
    cmd = [sys.executable, "-m", "scripts.m48_4e_a2_full_vs_tiled", "--worker",
           "--strategy", strategy, "--mode", args.mode,
           "--fine-side", str(grid["fine_side"]), "--core-side", str(core),
           "--depth", str(depth), "--width", str(args.width),
           "--heads", str(args.heads), "--seed", str(args.seed),
           "--periodic-length", str(grid["periodic_length"]),
           "--max-tiles", str(args.max_tiles), "--warmup", str(args.warmup),
           "--measure", str(args.measure),
           "--max-gpu-fraction", str(args.max_gpu_fraction),
           "--max-host-graph-gib", str(args.max_host_graph_gib),
           "--verification-output", str(verification),
           "--tiled-geometry-residency", args.tiled_geometry_residency,
           "--worker-output", str(result_file)]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.worker_timeout_seconds, check=False)
        row = json.loads(result_file.read_text()) if result_file.is_file() else {
            "status": "worker_crash", "returncode": proc.returncode}
        if proc.returncode and row.get("status") == "ok":
            row["status"] = "worker_exit_nonzero"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(proc.stdout[-8000:] + "\n=== stderr ===\n" + proc.stderr[-8000:])
    except subprocess.TimeoutExpired as exc:
        row = {"status": "worker_timeout", "limit_seconds": args.worker_timeout_seconds}
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(str(exc))
    row.update(trial_id=trial_id, strategy=strategy, fine_side=grid["fine_side"],
               mode=args.mode, core_side=core, depth=depth,
               worker_wall_seconds=time.perf_counter() - started)
    write_json(result_file, row)
    return row


def summarize_pair(full: dict, tiled: dict, verification: dict | None) -> dict:
    """Compare equal global supervised coverage and require numerical parity."""
    result = {"mode": full["mode"], "fine_side": full["fine_side"],
              "depth": full["depth"], "core_side": full["core_side"],
              "full_status": full["status"], "tiled_status": tiled["status"]}
    unexpected = {"error", "worker_crash", "worker_exit_nonzero"}
    if full["status"] in unexpected or tiled["status"] in unexpected:
        return {**result, "status": "ERROR"}
    if full["status"] != "ok" or tiled["status"] != "ok":
        # Capacity or bounded-time exclusions are not numerical speed results.
        return {**result, "status": "UNQUALIFIED"}
    if verification is None or not verification["passed"]:
        return {**result, "status": "PARITY_FAILED", "parity": verification}
    for row in (full, tiled):
        if row["supervised_nodes_per_update"] != full["fine_side"]**3:
            return {**result, "status": "COVERAGE_MISMATCH"}
    return {**result, "status": "PASS", "parity": verification,
            "full_mean_wall_seconds": full["mean_wall_update_seconds"],
            "tiled_mean_wall_seconds": tiled["mean_wall_update_seconds"],
            "full_over_tiled_speedup": (tiled["mean_wall_update_seconds"] /
                                        full["mean_wall_update_seconds"]),
            "full_peak_reserved_bytes": full["peak_reserved_bytes"],
            "tiled_peak_reserved_bytes": tiled["peak_reserved_bytes"],
            "full_geometry_setup_seconds": (full["geometry_build_seconds"] +
                                            full["geometry_transfer_seconds"]),
            "tiled_geometry_setup_seconds": (tiled["geometry_build_seconds"] +
                                             tiled["geometry_transfer_seconds"])}


def _write_csv(path: Path, data: list[dict]):
    columns = sorted(set().union(*(r.keys() for r in data))) if data else ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list))
                             else v for k, v in row.items()})


def run(args) -> dict:
    if (args.mode not in ("static", "dynamic")
            or args.tiled_geometry_residency not in ("auto", "cpu", "gpu")
            or args.width < 1 or args.heads < 1 or args.width % args.heads != 0
            or args.warmup < VERIFICATION_STEPS or args.measure < 2
            or args.seed < 0 or not 0 < args.max_gpu_fraction < 1
            or args.max_host_graph_gib <= 0 or args.worker_timeout_seconds < 1):
        raise ValueError("invalid A2 benchmark options")
    depths = positive_ints(args.depths)
    sides = positive_ints(args.fine_sides)
    cores = positive_ints(args.core_sides)
    if not set(sides) <= {48, 80}:
        raise ValueError("only authenticated 48/80 fine grids supported")
    for side, core in product(sides, cores):
        tile_origins(side, core, args.max_tiles)
    if args.output_dir.exists():
        raise FileExistsError("A2 refuses to overwrite or resume an output directory")
    grids, provenance = authentic_teacher_grids(args.ms_sr_root, args.normalization_file)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    gate = None
    if 80 in sides:
        gate = validate_48_parity_gate(
            args.parity_gate_summary, mode=args.mode, depths=depths,
            width=args.width, heads=args.heads, provenance=provenance,
            current_commit=head)
    args.output_dir.mkdir(parents=True)
    report = {"schema": SCHEMA, "status": "RUNNING", "source_commit": head,
              "no_scientific_training": True, "no_cedar_or_dns_field_evaluation": True,
              "same_global_fm_time_and_inputs": True,
              "active_initialization": ACTIVE_INIT,
              "required_verification_steps": VERIFICATION_STEPS,
              "parity_gate_48": gate,
              "provenance": provenance,
              "requested": {"mode": args.mode, "fine_sides": sides, "depths": depths,
                            "core_sides": cores, "width": args.width,
                            "heads": args.heads, "warmup": args.warmup,
                            "measure": args.measure, "max_tiles": args.max_tiles,
                            "tiled_geometry_residency": args.tiled_geometry_residency},
              "trials": [], "comparisons": []}
    write_json(args.output_dir / "summary.json", report)
    for side, depth, core in product(sides, depths, cores):
        rows = {}
        for strategy in ("full", "tiled"):
            row = _run_worker(args, grids[side], depth, core, strategy, args.output_dir)
            rows[strategy] = row
            report["trials"].append(row)
            write_json(args.output_dir / "summary.json", report)
            print(row["trial_id"], row["status"], flush=True)
        verification = None
        if rows["full"]["status"] == rows["tiled"]["status"] == "ok":
            try:
                a = args.output_dir / "verification" / (rows["full"]["trial_id"] + ".pt")
                b = args.output_dir / "verification" / (rows["tiled"]["trial_id"] + ".pt")
                verification = compare_verification(a, b)
            except Exception as exc:
                verification = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        comparison = summarize_pair(rows["full"], rows["tiled"], verification)
        report["comparisons"].append(comparison)
        write_json(args.output_dir / "summary.json", report)
        print("COMPARE", side, depth, core, comparison["status"], flush=True)
    _write_csv(args.output_dir / "trials.csv", report["trials"])
    _write_csv(args.output_dir / "comparisons.csv", report["comparisons"])
    report["qualified_pairs"] = sum(r["status"] == "PASS" for r in report["comparisons"])
    report["unqualified_pairs"] = len(report["comparisons"]) - report["qualified_pairs"]
    report["status"] = "PASS" if not report["unqualified_pairs"] else (
        "INCOMPLETE" if report["qualified_pairs"] else "NO_QUALIFIED_PAIRS")
    write_json(args.output_dir / "summary.json", report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "dynamic"), required=True)
    parser.add_argument("--fine-sides", default="48")
    parser.add_argument("--depths", default="2,4,8")
    parser.add_argument("--core-sides", default="24")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measure", type=int, default=2)
    parser.add_argument("--seed", type=int, default=484)
    parser.add_argument("--max-tiles", type=int, default=8)
    parser.add_argument("--tiled-geometry-residency", choices=("auto", "cpu", "gpu"),
                        default="auto")
    parser.add_argument("--parity-gate-summary", type=Path)
    parser.add_argument("--max-gpu-fraction", type=float, default=0.85)
    parser.add_argument("--max-host-graph-gib", type=float, default=12.0)
    parser.add_argument("--worker-timeout-seconds", type=int, default=1800)
    parser.add_argument("--ms-sr-root", type=Path, default=Path("/scratch/ai/MS_SR"))
    parser.add_argument("--normalization-file", type=Path, default=Path(
        "configs/m48/teacher_44_w1p25_l1_q4_q8.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--strategy", choices=("full", "tiled"))
    parser.add_argument("--fine-side", type=int)
    parser.add_argument("--core-side", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--periodic-length", type=float)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--verification-output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.metadata_only:
        grids, provenance = authentic_teacher_grids(args.ms_sr_root, args.normalization_file)
        print(json.dumps({"status": "PASS", "teacher_grids": grids,
                          "provenance": provenance}, sort_keys=True))
        return
    if args.worker:
        if (not args.strategy or not args.worker_output or not args.verification_output
                or not args.fine_side or not args.core_side or not args.depth
                or not args.periodic_length):
            raise ValueError("worker requires strategy, graph specification and receipt paths")
        _worker(args)
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --metadata-only")
    report = run(args)
    print("A2", report["status"], "qualified", report["qualified_pairs"],
          "of", len(report["comparisons"]))


if __name__ == "__main__":
    main()
