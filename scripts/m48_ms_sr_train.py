"""M48.4 teacher-only, globally projected, partitioned Flow Matching training.

Engineering qualification runner. This does NOT implement global generative
inference and does NOT use the locked test split. All MS_SR I/O is read-only.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from graph_attention.data.ms_sr_adapter import MANIFEST_SHA256, flat_channels, load_ms_sr_pair
from graph_attention.data.ms_sr_training import (
    checked_normalization, fit_teacher_normalization, record_identity,
    select_role_records,
)
from graph_attention.geometry.two_scale_targeted import build_targeted_two_scale_patch
from graph_attention.models.two_scale_dinat import TwoScaleDiNAT


def _write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write("\n")


def _geometry_device(geometry, device: torch.device):
    return replace(geometry, **{
        f.name: getattr(geometry, f.name).to(device)
        for f in fields(geometry)
        if isinstance(getattr(geometry, f.name), torch.Tensor)
    })


def _forward_args(geometry):
    return {
        "fine_coords": geometry.fine_coords,
        "coarse_coords": geometry.coarse_coords,
        "fine_edges": geometry.fine_edges,
        "fine_dilated": geometry.fine_dilated,
        "coarse_edges": geometry.coarse_edges,
        "coarse_dilated": geometry.coarse_dilated,
        "coarse_to_fine": geometry.coarse_to_fine,
        "fine_to_coarse": geometry.fine_to_coarse,
        "cross_weights": geometry.cross_weights,
        "refinement": geometry.refinement,
        "periodic_length": geometry.periodic_length,
        "coarse_spacing": geometry.coarse_spacing,
    }


def _owned_cube(side: int, width: int, origin: tuple[int, int, int]) -> torch.Tensor:
    """Global-ID periodic owned core; no artificial boundaries for attention."""
    if width < 1 or width > side:
        raise ValueError("invalid owned core width")
    axes = [(torch.arange(width, dtype=torch.long) + x) % side for x in origin]
    xyz = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return torch.unique((xyz[:, 0] * side + xyz[:, 1]) * side + xyz[:, 2], sorted=True)


def _prepare_pair(args, record, scales):
    pair, cedar, checks = load_ms_sr_pair(
        args.ms_sr_root, role="teacher", filter_id=record["filter_id"],
        fine_factor=4, coarse_factor=8,
        case_prefix=record["case"], snapshot_case=record["case"],
        snapshot_id=record["snapshot_id"],
        expected_manifest_sha256=args.manifest_sha256,
    )
    if any(v["max_component_scaled_linf"] > 1e-9 for v in checks.values()):
        raise RuntimeError("real source failed CEDAR physical consistency")
    mean, std, residual_scale = scales
    coarse = (flat_channels(pair.coarse) - mean) / std
    baseline = (flat_channels(pair.baseline) - mean) / std
    x1 = flat_channels(pair.residual) / residual_scale
    return pair, cedar, coarse, baseline, x1


def _one_step(
    model, optimizer, args, *, pair, cedar, coarse, baseline, x1,
    source_seed: int, core_origin: tuple[int, int, int], fm_time: float,
    device: torch.device,
):
    """Global FM preprocessing followed by core-owned network loss."""
    # Separate, reproducible CPU noise for every training step and snapshot.
    rng = torch.Generator(device="cpu").manual_seed(source_seed)
    eps = torch.randn(pair.fine.shape, dtype=torch.float64, generator=rng)
    source = cedar.project(eps)
    defect = cedar.restrict(source).abs().amax().item()
    if not defect < 1e-9:
        raise RuntimeError(f"globally projected FM source violates C: {defect}")
    x0 = flat_channels(source)
    xt = (1 - fm_time) * x0 + fm_time * x1
    target = x1 - x0
    nf = pair.fine.shape[-1]
    owned = _owned_cube(nf, args.core_side, core_origin)
    mode = model.mode
    plan, geometry = build_targeted_two_scale_patch(
        coarse_side=pair.coarse.shape[-1], refinement=pair.refinement,
        owned_fine=owned, fine_pre_layers=args.depth,
        fine_post_layers=args.depth, coarse_static_layers=args.depth,
        coarse_dynamic_layers=args.depth if mode == "dynamic" else 0,
        periodic_length=pair.periodic_length,
        max_fine_read=args.max_fine_read,
        max_coarse_read=args.max_coarse_read,
    )
    geometry = _geometry_device(geometry, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    state = xt[plan.fine_read].to(device=device, dtype=torch.float32)
    base = baseline[plan.fine_read].to(device=device, dtype=torch.float32)
    condition = coarse[plan.coarse_read].to(device=device, dtype=torch.float32)
    time_tensor = torch.tensor([fm_time], dtype=torch.float32, device=device)
    true_velocity = target[owned].to(device=device, dtype=torch.float32)
    predicted = model(
        state, baseline=base, coarse_state=condition,
        time=time_tensor, **_forward_args(geometry),
    )
    observed = predicted[geometry.owned_fine_local]
    loss = (observed - true_velocity).square().mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite FM training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise RuntimeError("nonfinite or zero gradient")
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "gradient_norm_preclip": float(norm),
        "fine_nodes_read": len(plan.fine_read),
        "coarse_nodes_read": len(plan.coarse_read),
        "owned_fine_nodes": len(owned),
        "projected_source_abs_defect": defect,
    }


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (args.max_sources < 1 or args.steps < 1 or args.core_side < 1
            or args.width < 1 or args.heads < 1 or args.width % args.heads
            or args.depth < 1 or args.learning_rate <= 0
            or args.grad_clip <= 0 or args.seed < 0):
        raise ValueError("invalid M48.4 training options")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    records, selection_sha = select_role_records(
        args.ms_sr_root, role="teacher", filter_id=args.filter_id,
        case_prefix=args.case_prefix, max_sources=args.max_sources,
        manifest_sha256=args.manifest_sha256,
    )
    # Never use validation or locked data to fit the statistics.
    normalization = fit_teacher_normalization(
        args.ms_sr_root, records, selection_sha,
        manifest_sha256=args.manifest_sha256,
    )
    scales = checked_normalization(
        normalization, records, selection_sha, manifest_sha256=args.manifest_sha256,
    )
    started = time.perf_counter()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    models = {
        mode: TwoScaleDiNAT(
            mode=mode, channels=5, fine_width=args.width,
            coarse_width=args.width, num_heads=args.heads,
            fine_pre_layers=args.depth, fine_post_layers=args.depth,
            coarse_static_layers=args.depth, coarse_dynamic_layers=args.depth,
        ).to(device)
        for mode in ("static", "dynamic")
    }
    optimizers = {
        mode: torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        for mode, model in models.items()
    }
    # Prevent a silent single-snapshot run: at least two independent sources.
    if len(records) < 2:
        raise ValueError("M48.4 qualification needs >=2 teacher snapshots")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_dir / "normalization.json", normalization)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    receipt = {
        "schema": "graph_attention.m48_multisource_train.v1",
        "git_commit": revision,
        "manifest_sha256": args.manifest_sha256,
        "selection_sha256": selection_sha,
        "records": [record_identity(r) for r in records],
        "steps": args.steps, "seed": args.seed,
        "mode": "qualification_only",
        "filter_id": args.filter_id, "case_prefix": args.case_prefix,
        "depth": args.depth, "width": args.width,
        "heads": args.heads, "core_side": args.core_side,
        "learning_rate": args.learning_rate, "device": args.device,
        "results": [],
    }
    for step in range(args.steps):
        record = records[step % len(records)]
        # Randomness is independent of network mode to enable matched samples.
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 100_000 + step)
        fm_time = float(0.001 + 0.998 * torch.rand((), generator=generator))
        # The periodic offset can cross a true global seam.
        seed_int = args.seed + 1_000_000 + step
        pair, cedar, coarse, baseline, x1 = _prepare_pair(args, record, scales)
        nf = pair.fine.shape[-1]
        origin = tuple(
            int(torch.randint(nf, (1,), generator=generator).item())
            for _ in range(3)
        )
        modes = {}
        for mode in ("static", "dynamic"):
            modes[mode] = _one_step(
                models[mode], optimizers[mode], args,
                pair=pair, cedar=cedar, coarse=coarse,
                baseline=baseline, x1=x1,
                source_seed=seed_int, core_origin=origin,
                fm_time=fm_time, device=device,
            )
        entry = {
            "step": step + 1, "case": record["case"],
            "snapshot_id": record["snapshot_id"],
            "fm_time": fm_time, "core_origin": origin,
            "modes": modes,
        }
        receipt["results"].append(entry)
        print(json.dumps(entry), flush=True)
    receipt["wall_seconds"] = time.perf_counter() - started
    for mode, model in models.items():
        save_path = args.output_dir / f"{mode}.pt"
        torch.save({
            "state_dict": model.state_dict(), "optimizer": optimizers[mode].state_dict(),
            "manifest_sha256": args.manifest_sha256,
            "selection_sha256": selection_sha,
            "normalization": normalization,
            "model_config": {
                "mode": mode, "channels": 5, "fine_width": args.width,
                "coarse_width": args.width, "num_heads": args.heads,
                "fine_pre_layers": args.depth, "fine_post_layers": args.depth,
                "coarse_static_layers": args.depth, "coarse_dynamic_layers": args.depth,
            },
            "seed": args.seed, "steps": args.steps,
        }, save_path)
        receipt.setdefault("checkpoints", {})[mode] = {
            "path": save_path.name,
            "sha256": hashlib.sha256(save_path.read_bytes()).hexdigest(),
        }
    _write_json(args.output_dir / "summary.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms-sr-root", type=Path, default=Path("/scratch/ai/MS_SR"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    parser.add_argument("--filter-id", choices=("w1p25_l1", "w1p5_l1"), default="w1p25_l1")
    parser.add_argument("--case-prefix", default="ReT060/")
    parser.add_argument("--max-sources", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--core-side", type=int, default=2)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-fine-read", type=int, default=100_000)
    parser.add_argument("--max-coarse-read", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=484)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
