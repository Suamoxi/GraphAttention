"""M45.8.1: train static/dynamic DiNAT on local coarse-conditioned HIT-like residuals.

Independent seeds are used for resolved-data selection, target residual draws,
model initialization, optimization schedule, validation and generation.
All model training is full-domain; M45.7 and the M42 DNS/LES pipeline are unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from graph_attention.data.conditional_hit_synthetic import (
    LocalHITConfig, generate_coarse_velocity, make_local_pair,
)
from graph_attention.data.two_scale_synthetic import SyntheticQB, fit_synthetic_scales
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from scripts.m45_7_oracle_supervision import _make_model
from scripts.m45_checkpoint_diagnostics import _write_csv
from scripts.m45_synthetic_pilot import _path, _state, _to_device, _validate, _velocity


def data_seed(base: int, index: int) -> int:
    if base < 0 or index < 0:
        raise ValueError("seed and index must be nonnegative")
    return int(base) + 3001 * index


def residual_seed(base: int, index: int) -> int:
    if base < 0 or index < 0:
        raise ValueError("seed and index must be nonnegative")
    return int(base) + 1_000_003 * index


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing directory: {args.output_dir}")
    cfg = LocalHITConfig(
        coarse_rms=args.coarse_rms,
        residual_fraction=args.residual_fraction,
        context_hops=args.context_hops,
        correlation_cutoff=args.correlation_cutoff,
        anisotropy=args.anisotropy,
    )
    cfg.validate()
    if args.mode not in ("static", "dynamic"):
        raise ValueError("mode must be static or dynamic")
    if args.residual_policy not in ("resample", "fixed"):
        raise ValueError("residual_policy must be resample or fixed")
    if (min(args.coarse_side, args.refinement, args.train_fields, args.val_fields,
            args.normalization_fields, args.steps, args.depth, args.width, args.heads) < 1
            or args.coarse_side < 3 or args.refinement < 2
            or args.width % args.heads
            or args.normalization_fields > args.train_fields
            or not 0 < args.learning_rate < 1):
        raise ValueError("invalid mesh, training or normalization settings")
    for name in ("dataset_seed", "residual_seed", "init_seed",
                 "schedule_seed", "validation_seed"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device(args.device)
    mesh = build_hit_two_scale_geometry(
        coarse_side=args.coarse_side, refinement=args.refinement,
    )
    mesh_device = _to_device(mesh, device)
    qb = SyntheticQB(mesh, "block_average")
    # Cache only distinct resolved coarse states, not every fine residual.
    coarse_fields = [
        generate_coarse_velocity(qb, cfg, seed=data_seed(args.dataset_seed, i))
        for i in range(args.train_fields)
    ]
    normalizer_pairs = [
        make_local_pair(
            qb, cfg, sample_id=i, coarse=coarse_fields[i],
            coarse_seed=data_seed(args.dataset_seed, i),
            residual_seed=residual_seed(args.residual_seed, i),
        )
        for i in range(args.normalization_fields)
    ]
    scales = fit_synthetic_scales(normalizer_pairs)
    val_pairs = [
        make_local_pair(
            qb, cfg, sample_id=100000 + i,
            coarse_seed=data_seed(args.validation_seed, i),
            residual_seed=residual_seed(args.validation_seed + 8_000_000, i),
        )
        for i in range(args.val_fields)
    ]
    model = _make_model(
        mode=args.mode, seed=args.init_seed, width=args.width,
        heads=args.heads, depth=args.depth, channels=3,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    choices = torch.randint(
        args.train_fields, (args.steps,),
        generator=torch.Generator(device="cpu").manual_seed(args.schedule_seed),
    )
    history = []
    started = time.perf_counter()
    for step, idx in enumerate(choices.tolist()):
        # The 'fixed' policy caches one realization implicitly by seed; both
        # policies see identical coarse snapshots and FM time/noise streams.
        noise_index = idx if args.residual_policy == "fixed" else step + args.train_fields
        pair = make_local_pair(
            qb, cfg, sample_id=idx, coarse=coarse_fields[idx],
            coarse_seed=data_seed(args.dataset_seed, idx),
            residual_seed=residual_seed(args.residual_seed, noise_index),
        )
        c, b, x1 = _state(pair, scales)
        xt, target, t = _path(
            x1, args.schedule_seed + 500_000 + 997 * step, qb,
        )
        model.train()
        pred = _velocity(
            model, xt.to(device), b.to(device), c.to(device),
            mesh_device, t.to(device),
        )
        loss = (pred - target.to(device)).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite FM loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        history.append({
            "step": step + 1,
            "training_fm_loss": float(loss.detach()),
            "gradient_norm_before_clip": grad_norm,
        })
        if (step + 1) % max(1, args.steps // 10) == 0:
            print(
                f"M45.8 {args.mode}/{args.residual_policy} "
                f"step={step + 1}/{args.steps} loss={float(loss):.6g}",
                flush=True,
            )

    duration = time.perf_counter() - started
    heldout_fm_mse = _validate(
        model, mesh_device, qb, val_pairs, scales, device,
        args.schedule_seed + 900_000, projected=True,
    )
    out = args.output_dir
    out.mkdir(parents=True)
    checkpoint_dir = out / "local_hit" / "block_average" / args.mode
    checkpoint_dir.mkdir(parents=True)
    _write_csv(checkpoint_dir / "train_history.csv", history, list(history[0]))
    checkpoint = checkpoint_dir / "checkpoint.pt"
    torch.save({
        "model": model.state_dict(),
        "scales": scales,
        "steps": args.steps,
        "model_config": {
            "mode": args.mode, "channels": 3,
            "fine_pre_layers": args.depth, "fine_post_layers": args.depth,
            "coarse_static_layers": args.depth,
            "coarse_dynamic_layers": args.depth,
        },
    }, checkpoint)
    settings = {
        "case": "local_hit", "operator": "block_average",
        "source_subspace": "nullspace",
        "channels": 3, "coarse_side": args.coarse_side,
        "refinement": args.refinement, "train_fields": args.train_fields,
        "val_fields": args.val_fields,
        "normalization_fields": args.normalization_fields,
        "steps": args.steps, "depth": args.depth, "width": args.width,
        "heads": args.heads, "learning_rate": args.learning_rate,
        "mode": args.mode, "residual_policy": args.residual_policy,
        "dataset_seed": args.dataset_seed,
        "residual_seed": args.residual_seed,
        "init_seed": args.init_seed, "schedule_seed": args.schedule_seed,
        "validation_seed": args.validation_seed, "device": args.device,
        "generator": asdict(cfg),
        "normalization": "fixed_training_coarse_and_residual_subset",
    }
    report = {
        "schema": "graph_attention.m45_8_local_train.v1",
        "scope": "synthetic local conditional HIT-like velocity, NOT a DNS solution",
        "settings": settings,
        "results": {
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "final_training_fm_loss": history[-1]["training_fm_loss"],
            "heldout_fm_loss": heldout_fm_mse,
            "train_seconds": duration,
            "mean_coarse": scales.mean_coarse.tolist(),
            "std_coarse": scales.std_coarse.tolist(),
            "rms_residual": scales.rms_residual.tolist(),
        },
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"M45.8 training complete: {out}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "dynamic"), required=True)
    parser.add_argument("--residual-policy", choices=("resample", "fixed"), default="resample")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--coarse-side", type=int, default=8)
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--train-fields", type=int, default=2048)
    parser.add_argument("--val-fields", type=int, default=16)
    parser.add_argument("--normalization-fields", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=71001)
    parser.add_argument("--residual-seed", type=int, default=72001)
    parser.add_argument("--init-seed", type=int, default=73001)
    parser.add_argument("--schedule-seed", type=int, default=74001)
    parser.add_argument("--validation-seed", type=int, default=75001)
    parser.add_argument("--coarse-rms", type=float, default=0.2)
    parser.add_argument("--residual-fraction", type=float, default=0.25)
    parser.add_argument("--context-hops", type=int, default=2)
    parser.add_argument("--correlation-cutoff", type=float, default=0.48)
    parser.add_argument("--anisotropy", type=float, default=0.25)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
