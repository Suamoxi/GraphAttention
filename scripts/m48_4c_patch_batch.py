"""M48.4c exact-halo configuration-C patch batching and profiling.

One independent full-domain CEDAR projected Gaussian source per patch.
A GPU microbatch is a disjoint union of exact TwoScaleDiNAT attention
patches; the optimizer objective is the equal-patch mean over the
effective batch, independent of GPU microbatch size.
"""
from __future__ import annotations

from dataclasses import replace
import math
import time

import torch

from graph_attention.data.ms_sr_adapter import flat_channels
from graph_attention.geometry.two_scale_targeted import build_targeted_two_scale_patch
from scripts.m48_ms_sr_train import _forward_args, _geometry_device, _owned_cube


def make_patch(
    prepared: tuple, args, *,
    fm_time: float, core_origin: tuple[int, int, int], source_seed: int,
    mode: str,
) -> dict:
    """Full-field FM task before extracting a targeted, exact dependency graph."""
    pair, cedar, coarse, baseline, x1 = prepared
    if mode not in ("static", "dynamic") or not 0.0 < fm_time < 1.0:
        raise ValueError("invalid patch mode or FM time")
    started = time.perf_counter()
    rng = torch.Generator(device="cpu").manual_seed(source_seed)
    eps = torch.randn(pair.fine.shape, dtype=torch.float64, generator=rng)
    x0_field = cedar.project(eps)
    defect = float(cedar.restrict(x0_field).abs().amax().item())
    if not math.isfinite(defect) or defect >= 1e-9:
        raise RuntimeError(f"full-field CEDAR projected noise defect: {defect}")
    x0 = flat_channels(x0_field)
    xt = (1 - fm_time) * x0 + fm_time * x1
    target = x1 - x0
    projection_seconds = time.perf_counter() - started

    started = time.perf_counter()
    owned = _owned_cube(pair.fine.shape[-1], args.core_side, core_origin)
    plan, geom = build_targeted_two_scale_patch(
        coarse_side=pair.coarse.shape[-1], refinement=pair.refinement,
        owned_fine=owned, fine_pre_layers=args.depth,
        fine_post_layers=args.depth, coarse_static_layers=args.depth,
        coarse_dynamic_layers=args.depth if mode == "dynamic" else 0,
        periodic_length=pair.periodic_length,
        max_fine_read=args.max_fine_read, max_coarse_read=args.max_coarse_read,
    )
    geometry_seconds = time.perf_counter() - started
    return {
        "state": xt[plan.fine_read].to(dtype=torch.float32),
        "baseline": baseline[plan.fine_read].to(dtype=torch.float32),
        "coarse_state": coarse[plan.coarse_read].to(dtype=torch.float32),
        "target": target[owned].to(dtype=torch.float32),
        "geometry": geom,
        "owned": owned,
        "fm_time": float(fm_time),
        "source_seed": int(source_seed),
        "core_origin": list(core_origin),
        "fine_nodes_read": int(plan.fine_read.numel()),
        "coarse_nodes_read": int(plan.coarse_read.numel()),
        "owned_fine_nodes": int(owned.numel()),
        "cedar_defect": defect,
        "projection_seconds": projection_seconds,
        "geometry_seconds": geometry_seconds,
    }


def pack_patches(patches: list[dict], device: torch.device) -> dict:
    """Make disconnected fine/coarse graphs with correct independent t values."""
    if not patches:
        raise ValueError("empty M48 patch microbatch")
    reference = patches[0]["geometry"]
    if any(
        p["geometry"].periodic_length != reference.periodic_length
        or p["geometry"].coarse_spacing != reference.coarse_spacing
        or p["geometry"].refinement != reference.refinement
        for p in patches
    ):
        raise ValueError("cannot pack patches with different spatial geometry scales")

    f_counts = [p["geometry"].fine_coords.shape[0] for p in patches]
    c_counts = [p["geometry"].coarse_coords.shape[0] for p in patches]
    f_offsets = [0]
    c_offsets = [0]
    for n in f_counts:
        f_offsets.append(f_offsets[-1] + n)
    for n in c_counts:
        c_offsets.append(c_offsets[-1] + n)

    def offset_edges(name: str, offsets0: list[int], offsets1: list[int]):
        return torch.cat([
            getattr(p["geometry"], name) + torch.tensor(
                [[offsets0[i]], [offsets1[i]]], dtype=torch.long
            )
            for i, p in enumerate(patches)
        ], dim=1)

    geom = replace(
        reference,
        fine_global=torch.cat([p["geometry"].fine_global for p in patches]),
        coarse_global=torch.cat([p["geometry"].coarse_global for p in patches]),
        owned_fine_local=torch.cat([
            p["geometry"].owned_fine_local + f_offsets[i]
            for i, p in enumerate(patches)
        ]),
        fine_coords=torch.cat([p["geometry"].fine_coords for p in patches]),
        coarse_coords=torch.cat([p["geometry"].coarse_coords for p in patches]),
        fine_edges=offset_edges("fine_edges", f_offsets, f_offsets),
        fine_dilated=offset_edges("fine_dilated", f_offsets, f_offsets),
        coarse_edges=offset_edges("coarse_edges", c_offsets, c_offsets),
        coarse_dilated=offset_edges("coarse_dilated", c_offsets, c_offsets),
        coarse_to_fine=offset_edges("coarse_to_fine", c_offsets, f_offsets),
        fine_to_coarse=offset_edges("fine_to_coarse", f_offsets, c_offsets),
        cross_weights=torch.cat([p["geometry"].cross_weights for p in patches]),
    )
    # Each component is a graph. IDs are disjoint despite overlapping
    # original physical locations (independent noise and FM time).
    fine_batches = torch.cat([
        torch.full((n,), i, dtype=torch.long) for i, n in enumerate(f_counts)
    ])
    coarse_batches = torch.cat([
        torch.full((n,), i, dtype=torch.long) for i, n in enumerate(c_counts)
    ])
    sizes = [p["target"].shape[0] for p in patches]
    if (not all(x > 0 for x in sizes)
            or geom.owned_fine_local.numel() != sum(sizes)):
        raise RuntimeError("packed owned-node indexing mismatch")
    return {
        "state": torch.cat([p["state"] for p in patches]).to(device),
        "baseline": torch.cat([p["baseline"] for p in patches]).to(device),
        "coarse_state": torch.cat([p["coarse_state"] for p in patches]).to(device),
        "target": torch.cat([p["target"] for p in patches]).to(device),
        "geometry": _geometry_device(geom, device),
        "fine_batch_index": fine_batches.to(device),
        "coarse_batch_index": coarse_batches.to(device),
        "time": torch.tensor([p["fm_time"] for p in patches],
                             dtype=torch.float32, device=device),
        "owned_sizes": sizes,
    }


def packed_loss(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """One loss per physical patch, not weighted by its halo size."""
    geometry = batch["geometry"]
    output = model(
        batch["state"], baseline=batch["baseline"],
        coarse_state=batch["coarse_state"], time=batch["time"],
        fine_batch_index=batch["fine_batch_index"],
        coarse_batch_index=batch["coarse_batch_index"],
        **_forward_args(geometry),
    )
    squared = (output[geometry.owned_fine_local] - batch["target"]).square()
    by_patch = torch.stack([
        values.mean() for values in squared.split(batch["owned_sizes"])
    ])
    if not bool(torch.isfinite(by_patch).all()):
        raise RuntimeError("nonfinite M48 patch training loss")
    return by_patch.mean(), by_patch


def accumulated_batch_step(
    model, optimizer, args, *, prepared: tuple,
    draws: list[tuple[float, tuple[int, int, int], int]],
    device: torch.device,
) -> dict:
    """Exactly one optimizer update across effective B=4 projected patches."""
    effective = len(draws)
    micro = args.microbatch_size
    if effective < 2 or micro < 1 or effective % micro:
        raise ValueError("effective batch must be divisible by GPU microbatch")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    samples = []
    forward_backward_seconds = 0.0
    for start in range(0, effective, micro):
        patches = [
            make_patch(
                prepared, args, fm_time=t, core_origin=origin,
                source_seed=seed, mode=model.mode,
            ) for t, origin, seed in draws[start:start + micro]
        ]
        started = time.perf_counter()
        batch = pack_patches(patches, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        packing_seconds = time.perf_counter() - started

        started = time.perf_counter()
        _, per_patch = packed_loss(model, batch)
        # per-patch reduction first, then global effective-batch mean.
        # One optimizer step after all microbatch gradients.
        loss = per_patch.sum() / effective
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_backward_seconds += time.perf_counter() - started
        for p, lv in zip(patches, per_patch.detach().cpu().tolist()):
            samples.append({
                "fm_time": p["fm_time"], "source_seed": p["source_seed"],
                "core_origin": p["core_origin"], "loss": lv,
                "fine_nodes_read": p["fine_nodes_read"],
                "coarse_nodes_read": p["coarse_nodes_read"],
                "owned_fine_nodes": p["owned_fine_nodes"],
                "projected_source_abs_defect": p["cedar_defect"],
                "projection_seconds": p["projection_seconds"],
                "geometry_seconds": p["geometry_seconds"],
                "packing_seconds_per_microbatch": packing_seconds,
            })
        del patches, batch, per_patch, loss

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise RuntimeError("nonfinite or zero accumulated gradient")
    optimizer.step()
    unique_owned = set()
    for p in samples:
        # Count overlap only within this same physical source; origin is
        # periodic and the core contains core_side**3 global unique IDs.
        side = prepared[0].fine.shape[-1]
        owned = _owned_cube(side, args.core_side, tuple(p["core_origin"]))
        unique_owned.update(owned.tolist())
    return {
        "loss": sum(p["loss"] for p in samples) / effective,
        "gradient_norm_preclip": float(norm),
        "effective_batch": effective,
        "gpu_microbatch_size": micro,
        "owned_fine_nodes": sum(p["owned_fine_nodes"] for p in samples),
        "unique_owned_fine_nodes": len(unique_owned),
        "fine_nodes_read": sum(p["fine_nodes_read"] for p in samples),
        "coarse_nodes_read": sum(p["coarse_nodes_read"] for p in samples),
        "projected_source_abs_defect": max(
            p["projected_source_abs_defect"] for p in samples
        ),
        "projection_seconds": sum(p["projection_seconds"] for p in samples),
        "geometry_seconds": sum(p["geometry_seconds"] for p in samples),
        "forward_backward_seconds": forward_backward_seconds,
        "patches": samples,
    }


def preflight_halos(records: list[dict], args, mode: str) -> dict:
    """Audit core and wrap-around exact read sets on every selected grid size.

    This tests the geometry implementation and cap independently of CFD
    noise, without changing the published CEDAR or reading locked data.
    """
    grids: dict[tuple[int, int], set[str]] = {}
    for record in records:
        c_shape = tuple(record["input_shape"])
        f_shape = tuple(record["target_shape"])
        if (len(c_shape) != 3 or len(f_shape) != 3
                or len(set(c_shape)) != 1 or len(set(f_shape)) != 1
                or f_shape[0] != 2 * c_shape[0]):
            raise ValueError("expected cubic q8->q4 teacher source geometry")
        grids.setdefault((c_shape[0], f_shape[0]), set()).add(record["case"])
    results = []
    for (nc, nf), cases in sorted(grids.items()):
        if args.core_side > nf:
            raise ValueError("owned core wider than fine lattice")
        for name, origin in [
            ("interior", (nf // 3,) * 3),
            ("global_periodic_seam", (nf - 2, nf // 2, nf - 3)),
        ]:
            started = time.perf_counter()
            owned = _owned_cube(nf, args.core_side, origin)
            plan, geom = build_targeted_two_scale_patch(
                coarse_side=nc, refinement=2, owned_fine=owned,
                fine_pre_layers=args.depth, fine_post_layers=args.depth,
                coarse_static_layers=args.depth,
                coarse_dynamic_layers=args.depth if mode == "dynamic" else 0,
                max_fine_read=args.max_fine_read,
                max_coarse_read=args.max_coarse_read,
            )
            if not torch.equal(
                geom.fine_global[geom.owned_fine_local], owned
            ) or owned.numel() != args.core_side**3:
                raise RuntimeError("owned global/local index mapping is incorrect")
            results.append({
                "coarse_side": nc, "fine_side": nf,
                "cases_on_grid": len(cases), "location": name,
                "origin": list(origin), "owned_fine_nodes": len(owned),
                "fine_nodes_read": len(plan.fine_read),
                "coarse_nodes_read": len(plan.coarse_read),
                "read_to_owned_fine_ratio": len(plan.fine_read) / len(owned),
                "fine_edges": int(geom.fine_edges.shape[1]),
                "fine_dilated_edges": int(geom.fine_dilated.shape[1]),
                "coarse_edges": int(geom.coarse_edges.shape[1]),
                "coarse_dilated_edges": int(geom.coarse_dilated.shape[1]),
                "cross_edges": int(geom.coarse_to_fine.shape[1]),
                "geometry_seconds": time.perf_counter() - started,
            })
    return {
        "schema": "graph_attention.m48_4c_halo_preflight.v1",
        "mode": mode, "core_side": args.core_side,
        "depth": args.depth, "cases": len({r["case"] for r in records}),
        "sources": len(records), "results": results,
    }
