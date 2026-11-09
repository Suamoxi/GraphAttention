"""M45.7: matched FM/oracle supervision with fixed or resampled residuals.

This is a synthetic block-average Q/B qualification, NOT DNS/LES validation.
For each matched experiment only the supervision and/or conditional-residual
sampling policy changes. Coarse fields, fitted scales, FM noise/times,
initialization, model, optimizer, validation split, and evaluation are shared.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from graph_attention.data.two_scale_synthetic import (
    CASES, SyntheticPair, SyntheticQB, SyntheticScales,
    fit_synthetic_scales, make_synthetic_pair, sample_conditional_residual,
)
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from graph_attention.models.two_scale_dinat import TwoScaleDiNAT
from scripts.m45_checkpoint_diagnostics import BlockGaussianOracle, _write_csv
from scripts.m45_synthetic_pilot import _path, _state, _to_device, _validate, _velocity


def _make_model(*, mode: str, seed: int, width: int, heads: int, depth: int) -> TwoScaleDiNAT:
    """Match M45.6 initialization, including shared static/dynamic parameters."""
    if mode not in ("static", "dynamic"):
        raise ValueError("mode must be static or dynamic")
    params = dict(
        coarse_width=width, fine_width=width, num_heads=heads,
        fine_pre_layers=depth, fine_post_layers=depth,
        coarse_static_layers=depth, coarse_dynamic_layers=depth,
    )
    torch.manual_seed(seed + 77)
    static = TwoScaleDiNAT(mode="static", **params)
    if mode == "static":
        return static
    torch.manual_seed(seed + 77)
    dynamic = TwoScaleDiNAT(mode="dynamic", **params)
    static_state = static.state_dict()
    dynamic.load_state_dict(
        {name: value for name, value in static_state.items()
         if name in dynamic.state_dict()},
        strict=False,
    )
    return dynamic


def _supervision_target(
    supervision: str, xt: torch.Tensor, fm_target: torch.Tensor, t: torch.Tensor,
    oracle: BlockGaussianOracle | None,
) -> torch.Tensor:
    """Targets are normalized velocity fields in the same Q-nullspace."""
    if supervision == "fm":
        return fm_target
    if supervision == "oracle":
        if oracle is None:
            raise ValueError("oracle supervision requires BlockGaussianOracle")
        return oracle.velocity(xt, float(t.item()))
    raise ValueError("supervision must be fm or oracle")


def _training_residual(
    pair: SyntheticPair, qb: SyntheticQB, scales: SyntheticScales, *,
    policy: str, case: str, seed: int, step: int,
) -> torch.Tensor:
    """Choose an R_f paired with the given coarse field; return normalized R_f.

    Both policies use the same frozen train-set normalization. Resampling
    changes only the unresolved conditional realization, never U_c or B(U_c).
    This RNG stream is independent of pair selection and FM source/time draws.
    """
    if policy == "fixed":
        return pair.residual / scales.rms_residual
    if policy != "resample":
        raise ValueError(f"unknown residual policy: {policy}")
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + 81_000_000 + (step + 1) * 1_000_003,
    )
    residual, _ = sample_conditional_residual(
        qb, pair.coarse, case=case, generator=generator,
    )
    return residual / scales.rms_residual


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.output_dir}")
    if args.residual_policy not in ("fixed", "resample"):
        raise ValueError(f"unknown residual policy: {args.residual_policy}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if (min(args.coarse_side, args.refinement, args.train_samples, args.val_samples,
            args.steps, args.depth, args.width, args.heads) < 1
            or args.refinement < 2 or args.width % args.heads):
        raise ValueError("invalid mesh, training, or attention dimensions")
    if not 0 < args.learning_rate < 1 or args.seed < 0 or args.val_seed < 0:
        raise ValueError("invalid learning rate or seed")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    mesh = build_hit_two_scale_geometry(
        coarse_side=args.coarse_side, refinement=args.refinement,
    )
    mesh_gpu = _to_device(mesh, device)
    qb = SyntheticQB(mesh, "block_average")
    train = [
        make_synthetic_pair(qb, case=args.case, sample_id=i,
                            seed=args.seed + 3001 * i)
        for i in range(args.train_samples)
    ]
    validation = [
        make_synthetic_pair(qb, case=args.case, sample_id=i + 100000,
                            seed=args.val_seed + 3001 * i)
        for i in range(args.val_samples)
    ]
    # Keep these reference scales FIXED across both policies. This isolates
    # conditional residual diversity from changes to normalization statistics.
    scales = fit_synthetic_scales(train)
    model = _make_model(
        mode=args.mode, seed=args.seed, width=args.width,
        heads=args.heads, depth=args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # The M45.6 full_all training path uses this exact sample selection seed.
    sample_generator = torch.Generator(device="cpu").manual_seed(args.seed + 23000)
    sample_ids = torch.randint(
        len(train), (args.steps,), generator=sample_generator,
    )
    cached_oracles: dict[int, BlockGaussianOracle] = {}
    history = []
    started = time.perf_counter()
    for step, sample_id in enumerate(sample_ids.tolist()):
        pair = train[sample_id]
        c, b, _ = _state(pair, scales)
        x1 = _training_residual(
            pair, qb, scales, policy=args.residual_policy, case=args.case,
            seed=args.seed, step=step,
        )
        xt, fm_target, t = _path(
            x1, args.seed + 500_000 + step * 997, qb,
        )
        xt_device = xt.to(device)
        oracle = None
        if args.supervision == "oracle":
            if sample_id not in cached_oracles:
                cached_oracles[sample_id] = BlockGaussianOracle(
                    qb, pair.coarse, scales, args.case, device,
                )
            oracle = cached_oracles[sample_id]
        with torch.no_grad():
            target = _supervision_target(
                args.supervision, xt_device, fm_target.to(device), t, oracle,
            )

        model.train()
        predicted = _velocity(
            model, xt_device, b.to(device), c.to(device),
            mesh_gpu, t.to(device),
        )
        # Both objectives use raw model outputs; this retains the M45.6
        # FM loss and penalizes off-nullspace velocity in the same manner.
        loss = (predicted - target).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite training loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0,
        ).item())
        optimizer.step()
        history.append(dict(
            step=step + 1, training_loss=float(loss.item()),
            grad_norm_before_clip=grad_norm,
        ))
        if (step + 1) % max(1, args.steps // 4) == 0:
            print(
                f"M45.7 case={args.case} {args.mode}/{args.supervision} "
                f"residual={args.residual_policy} "
                f"step={step + 1}/{args.steps} loss={loss.item():.6g}",
                flush=True,
            )

    train_seconds = time.perf_counter() - started
    heldout_fm_loss = _validate(
        model, mesh_gpu, qb, validation, scales, device,
        args.seed + 90_000, projected=True,
    )
    output = args.output_dir
    output.mkdir(parents=True)
    checkpoint_dir = output / args.case / "block_average" / args.mode
    checkpoint_dir.mkdir(parents=True)
    _write_csv(
        checkpoint_dir / "train_history.csv", history,
        ["step", "training_loss", "grad_norm_before_clip"],
    )
    torch.save({
        "model": model.state_dict(),
        "scales": scales,
        "steps": args.steps,
        "model_config": {
            "mode": model.mode,
            "fine_pre_layers": model.fine_pre_layers,
            "fine_post_layers": model.fine_post_layers,
            "coarse_static_layers": model.coarse_static_layers,
            "coarse_dynamic_layers": model.coarse_dynamic_layers,
        },
    }, checkpoint_dir / "checkpoint.pt")
    metrics = {
        "case": args.case,
        "mode": args.mode,
        "supervision": args.supervision,
        "residual_policy": args.residual_policy,
        "training_scope": "full_all",
        "training_steps": args.steps,
        "final_training_loss": history[-1]["training_loss"],
        "heldout_fm_loss": heldout_fm_loss,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": train_seconds,
    }
    (checkpoint_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
    )
    settings = {
        "operator": "block_average",
        "source_subspace": "nullspace",
        "cases": [args.case],
        "coarse_side": args.coarse_side,
        "refinement": args.refinement,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "steps": args.steps,
        "width": args.width,
        "heads": args.heads,
        "fine_pre_layers": args.depth,
        "fine_post_layers": args.depth,
        "coarse_static_layers": args.depth,
        "coarse_dynamic_layers": args.depth,
        "seed": args.seed,
        "val_seed": args.val_seed,
        "learning_rate": args.learning_rate,
        "training_scope": "full_all",
        "supervision": args.supervision,
        "residual_policy": args.residual_policy,
        "normalization": "fitted_once_on_initial_fixed_training_pairs",
        "residual_resampling_seed_rule": (
            "seed + 81000000 + (step + 1) * 1000003"
            if args.residual_policy == "resample" else None
        ),
        "device": args.device,
        "output_dir": str(output),
    }
    report = {
        "schema": "graph_attention.m45_7_oracle_supervision.v2",
        "scope": "synthetic block-average residuals, NOT physical DNS/LES",
        "settings": settings,
        "results": [metrics],
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
    )
    environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device)
                        if device.type == "cuda" else "cpu"),
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n",
    )
    print(f"M45.7 completed: {output / 'summary.json'}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "dynamic"), required=True)
    parser.add_argument("--supervision", choices=("fm", "oracle"), required=True)
    parser.add_argument(
        "--residual-policy", choices=("fixed", "resample"), default="fixed",
        help="Reuse one residual per coarse field, or sample a fresh R_f|U_c "
             "at every training update. Scales remain fitted on fixed pairs.",
    )
    parser.add_argument("--case", choices=CASES, default="shared_mode")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--coarse-side", type=int, default=4)
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-seed", type=int, default=7002026)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
