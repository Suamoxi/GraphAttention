"""M45 controlled synthetic FM experiment; NOT a physical DNS/LES validation.

Compare static/dynamic TwoScaleDiNAT on the *same* synthetic Q/B operators,
coarse/fine pairs, train/validation split, times and random sources. The fine
residual is synthesized in ker(Q); generated samples are unconstrained and
coarse consistency is measured rather than asserted.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
import json
from pathlib import Path
import time

import torch

from graph_attention.data.two_scale_synthetic import (
    CASES, OPERATORS, SyntheticQB, SyntheticPair, SyntheticScales,
    fit_synthetic_scales, make_synthetic_pair,
)
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from graph_attention.geometry.two_scale_partition import (
    extract_two_scale_patch_geometry, two_scale_patch_dependencies,
)
from graph_attention.models.two_scale_dinat import TwoScaleDiNAT


def _to_device(record, device: torch.device):
    return replace(record, **{
        f.name: getattr(record, f.name).to(device)
        for f in fields(record) if isinstance(getattr(record, f.name), torch.Tensor)
    })


def _forward_kwargs(mesh) -> dict:
    return {
        "fine_coords": mesh.fine_coords,
        "coarse_coords": mesh.coarse_coords,
        "fine_edges": mesh.fine_edges,
        "fine_dilated": mesh.fine_dilated,
        "coarse_edges": mesh.coarse_edges,
        "coarse_dilated": mesh.coarse_dilated,
        "coarse_to_fine": mesh.coarse_to_fine,
        "fine_to_coarse": mesh.fine_to_coarse,
        "cross_weights": mesh.cross_weights,
        "periodic_length": mesh.periodic_length,
        "coarse_spacing": mesh.coarse_spacing,
        "refinement": mesh.refinement,
    }


def _make_owned_cores(mesh, core_side: int) -> list[torch.Tensor]:
    nf = mesh.fine_side
    if core_side < 1 or nf % core_side != 0:
        raise ValueError("core-side must be positive and divide fine_side")
    cores = []
    for x0 in range(0, nf, core_side):
        for y0 in range(0, nf, core_side):
            for z0 in range(0, nf, core_side):
                axes = [torch.arange(start, start + core_side) for start in (x0, y0, z0)]
                xyz = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
                cores.append((xyz[:, 0] * nf * nf + xyz[:, 1] * nf + xyz[:, 2]).long())
    return cores


def _patches(mesh, model: TwoScaleDiNAT, core_side: int, device: torch.device):
    result = []
    for owned in _make_owned_cores(mesh, core_side):
        plan = two_scale_patch_dependencies(
            owned, fine_local=mesh.fine_edges, fine_dilated=mesh.fine_dilated,
            coarse_local=mesh.coarse_edges, coarse_dilated=mesh.coarse_dilated,
            coarse_to_fine=mesh.coarse_to_fine, fine_to_coarse=mesh.fine_to_coarse,
            fine_pre_layers=model.fine_pre_layers,
            fine_post_layers=model.fine_post_layers,
            coarse_static_layers=model.coarse_static_layers,
            coarse_dynamic_layers=model.coarse_dynamic_layers,
        )
        cpu = extract_two_scale_patch_geometry(mesh, plan)
        result.append((cpu, _to_device(cpu, device)))
    return result


def _state(pair: SyntheticPair, scales: SyntheticScales) -> tuple[torch.Tensor, ...]:
    return (
        (pair.coarse - scales.mean_coarse) / scales.std_coarse,
        (pair.baseline - scales.mean_coarse) / scales.std_coarse,
        pair.residual / scales.rms_residual,
    )


def _path(x1: torch.Tensor, seed: int, qb: SyntheticQB | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    t = torch.rand(1, generator=gen, dtype=x1.dtype)
    eps = torch.randn(x1.shape, generator=gen, dtype=x1.dtype)
    if qb is not None:
        eps = qb.nullspace(eps)
    return (1 - t) * eps + t * x1, x1 - eps, t


def _velocity(model, x: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
              geometry, t: torch.Tensor, cached_coarse=None):
    return model(
        x, baseline=b, coarse_state=c,
        time=t, cached_coarse=cached_coarse, **_forward_kwargs(geometry)
    )


def _validate(model, mesh_gpu, qb_cpu, pairs, scales, device: torch.device, seed: int, projected: bool) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for pair in pairs:
            c, b, x1 = _state(pair, scales)
            xt, velocity, t = _path(x1, seed + pair.sample_id * 101, qb_cpu if projected else None)
            y = _velocity(model, xt.to(device), b.to(device), c.to(device),
                          mesh_gpu, t.to(device))
            losses.append((y - velocity.to(device)).square().mean().item())
    return float(sum(losses) / len(losses))


@torch.no_grad()
def _generate(model, mesh_gpu, qb_gpu, pair, scales, *, device, seed, heun_steps, projected):
    model.eval()
    c, b, _ = _state(pair, scales)
    c, b = c.to(device), b.to(device)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(pair.residual.shape, generator=gen).to(device)
    if projected:
        x = qb_gpu.nullspace(x)
    cached = model.encode_coarse(
        c, coarse_coords=mesh_gpu.coarse_coords,
        coarse_edges=mesh_gpu.coarse_edges, coarse_dilated=mesh_gpu.coarse_dilated,
        refinement=mesh_gpu.refinement, time=x.new_zeros(1),
        periodic_length=mesh_gpu.periodic_length,
    )
    dt = 1.0 / heun_steps
    for step in range(heun_steps):
        t0 = x.new_tensor([step * dt])
        v0 = _velocity(model, x, b, c, mesh_gpu, t0, cached)
        if projected:
            v0 = qb_gpu.nullspace(v0)
        xp = x + dt * v0
        v1 = _velocity(model, xp, b, c, mesh_gpu, x.new_tensor([(step + 1) * dt]), cached)
        if projected:
            v1 = qb_gpu.nullspace(v1)
        x = x + 0.5 * dt * (v0 + v1)
    return (x.cpu() * scales.rms_residual).contiguous()


def _sample_diagnostics(model, mesh_gpu, qb, qb_gpu, pairs, scales, *, device, seed, heun_steps, projected):
    oracle_mode, generated_mode, coarse_defects, sample_rms = [], [], [], []
    for pair in pairs:
        residual = _generate(model, mesh_gpu, qb_gpu, pair, scales, device=device,
                             seed=seed + pair.sample_id * 103, heun_steps=heun_steps,
                             projected=projected)
        # The signed amplitude of the known shared spatial mode, *not* a paired
        # sample-matching error. Both ensembles use the same coarse snapshots.
        template = pair.mode[:, 0]
        denom = template.square().sum()
        oracle_mode.append(float(torch.dot(pair.residual[:, 1], template) / denom))
        generated_mode.append(float(torch.dot(residual[:, 1], template) / denom))
        coarse_defects.append(qb.restrict(residual).abs().max().item())
        sample_rms.append(residual.square().mean(dim=0).sqrt().tolist())
    result = {
        "oracle_mode_second_moment": sum(q*q for q in oracle_mode) / len(oracle_mode),
        "generated_mode_second_moment": sum(q*q for q in generated_mode) / len(generated_mode),
        "max_generated_Q_residual": max(coarse_defects),
        "generated_channel_rms": [sum(x[k] for x in sample_rms) / len(sample_rms) for k in range(5)],
        "oracle_channel_rms": torch.stack([p.residual.square().mean(0) for p in pairs]).mean(0).sqrt().tolist(),
        "generated_samples": len(pairs),
    }
    return result


def _train_one(
    model, mesh_cpu, mesh_gpu, qb, qb_gpu, train_pairs, val_pairs, scales, *,
    projected: bool, device: torch.device, seed: int, steps: int, lr: float, core_side: int,
    heun_steps: int, sample_count: int, out_dir: Path,
    training_scope: str = "patch_owned",
) -> dict:
    if training_scope not in ("patch_owned", "full_owned", "full_all"):
        raise ValueError(f"invalid training scope: {training_scope}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    patches = _patches(mesh_cpu, model, core_side, device)
    rng = torch.Generator().manual_seed(seed + 23000)
    # Same preselected sample/core sequence for both architectures, even when
    # their dependency closures contain different numbers of halo nodes.
    sample_ids = torch.randint(len(train_pairs), (steps,), generator=rng)
    core_ids = torch.randint(len(patches), (steps,), generator=rng)
    history = []
    start = time.perf_counter()
    for step in range(steps):
        model.train()
        pair = train_pairs[int(sample_ids[step])]
        patch_cpu, patch_gpu = patches[int(core_ids[step])]
        c, b, x1 = _state(pair, scales)
        xt, target_v, t = _path(x1, seed + 500_000 + step * 997, qb if projected else None)
        if training_scope == "patch_owned":
            predicted = _velocity(
                model, xt[patch_cpu.fine_global].to(device),
                b[patch_cpu.fine_global].to(device),
                c[patch_cpu.coarse_global].to(device), patch_gpu, t.to(device),
            )
            observed = target_v[patch_cpu.fine_global].to(device)
            indices = patch_gpu.owned_fine_local
            fine_read, coarse_read = len(patch_cpu.fine_global), len(patch_cpu.coarse_global)
        else:
            # Compare identical owned-node gradients on the complete graph,
            # or train on all fine nodes using the same sampled FM path.
            predicted = _velocity(model, xt.to(device), b.to(device), c.to(device),
                                  mesh_gpu, t.to(device))
            observed = target_v.to(device)
            indices = (patch_cpu.fine_global[patch_cpu.owned_fine_local].to(device)
                       if training_scope == "full_owned" else slice(None))
            fine_read, coarse_read = mesh_cpu.fine_side**3, mesh_cpu.coarse_side**3
        loss = (predicted[indices] - observed[indices]).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite M45 synthetic training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        record = {
            "step": step + 1, "training_loss": float(loss.item()),
            "gradient_norm": grad_norm,
            "fine_owned_nodes": (mesh_cpu.fine_side**3 if training_scope == "full_all"
                                 else len(patch_cpu.owned_fine_local)),
            "fine_read_nodes": fine_read,
            "coarse_read_nodes": coarse_read,
        }
        history.append(record)
        if (step + 1) % max(1, steps // 4) == 0:
            print(f"{model.mode:7} step {step+1:4}/{steps}: "
                  f"loss={record['training_loss']:.5g}; "
                  f"scope={training_scope} fine read/owned="
                  f"{record['fine_read_nodes']}/{record['fine_owned_nodes']}", flush=True)
    elapsed = time.perf_counter() - start
    val_loss = _validate(model, mesh_gpu, qb, val_pairs, scales, device, seed + 90000, projected)
    sample_metrics = _sample_diagnostics(
        model, mesh_gpu, qb, qb_gpu, val_pairs[:sample_count], scales,
        device=device, seed=seed + 110_000, heun_steps=heun_steps, projected=projected,
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    with (out_dir / "train_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    torch.save({
        "model": model.state_dict(), "scales": scales, "steps": steps,
        "model_config": {
            "mode": model.mode, "fine_pre_layers": model.fine_pre_layers,
            "fine_post_layers": model.fine_post_layers,
            "coarse_static_layers": model.coarse_static_layers,
            "coarse_dynamic_layers": model.coarse_dynamic_layers,
        },
    }, out_dir / "checkpoint.pt")
    result = {
        "mode": model.mode,
        "training_scope": training_scope,
        "train_samples": len(train_pairs),
        "training_steps": steps,
        "final_training_loss": history[-1]["training_loss"],
        "heldout_fm_loss": val_loss,
        "parameters": sum(p.numel() for p in model.parameters()),
        "time_train_s": elapsed,
        "mean_fine_read_to_owned": sum(h["fine_read_nodes"] / h["fine_owned_nodes"] for h in history) / steps,
        "max_fine_read_nodes": max(h["fine_read_nodes"] for h in history),
        "max_coarse_read_nodes": max(h["coarse_read_nodes"] for h in history),
        **sample_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coarse-side", type=int, default=4)
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--operator", choices=OPERATORS, default="block_average")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--heun-steps", type=int, default=8)
    parser.add_argument("--generated-samples", type=int, default=4)
    parser.add_argument("--core-side", type=int, default=4)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--fine-pre-layers", type=int, default=1)
    parser.add_argument("--fine-post-layers", type=int, default=1)
    parser.add_argument("--coarse-static-layers", type=int, default=1)
    parser.add_argument("--coarse-dynamic-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-seed", type=int, default=None,
                        help="Fixed validation seed across training seeds; default retains M45.2 split")
    parser.add_argument("--training-scope", choices=("patch_owned", "full_owned", "full_all"),
                        default="patch_owned")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--source-subspace", choices=("nullspace", "full"), default="nullspace")
    args = parser.parse_args()
    if (min(args.train_samples, args.val_samples, args.steps, args.heun_steps,
            args.generated_samples, args.core_side, args.width, args.heads,
            args.fine_pre_layers, args.fine_post_layers,
            args.coarse_static_layers, args.coarse_dynamic_layers) < 1
            or args.generated_samples > args.val_samples
            or args.width % args.heads):
        parser.error("positive counts, generated-samples <= val-samples, width divisible by heads required")
    if args.seed < 0 or (args.val_seed is not None and args.val_seed < 0):
        parser.error("random seeds must be nonnegative")
    if not (0 < args.learning_rate < 1):
        parser.error("learning-rate must be in (0,1)")
    if args.output_dir.exists():
        parser.error("refusing to overwrite existing output-dir")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    mesh = build_hit_two_scale_geometry(
        coarse_side=args.coarse_side, refinement=args.refinement
    )
    mesh_gpu = _to_device(mesh, device)
    qb = SyntheticQB(mesh, args.operator)
    qb_gpu = SyntheticQB(mesh_gpu, args.operator)
    projected = args.source_subspace == "nullspace"
    probe = torch.randn((mesh.coarse_side**3, 5), generator=torch.Generator().manual_seed(args.seed))
    operator_error = (qb.restrict(qb.reconstruct(probe)) - probe).abs().max().item()
    if operator_error > 1e-6:
        raise RuntimeError(f"Q B != I: {operator_error}")
    args.output_dir.mkdir(parents=True)
    all_results = []
    for case in args.cases:
        print(f"=== M45 case={case} operator={args.operator} r={args.refinement} ===", flush=True)
        train = [make_synthetic_pair(qb, case=case, sample_id=i, seed=args.seed + i * 3001)
                 for i in range(args.train_samples)]
        validation_seed = args.seed + 7_000_000 if args.val_seed is None else args.val_seed
        val = [make_synthetic_pair(qb, case=case, sample_id=i + 100000,
                                   seed=validation_seed + i * 3001)
               for i in range(args.val_samples)]
        scales = fit_synthetic_scales(train)
        params = dict(coarse_width=args.width, fine_width=args.width, num_heads=args.heads,
                      fine_pre_layers=args.fine_pre_layers,
                      fine_post_layers=args.fine_post_layers,
                      coarse_static_layers=args.coarse_static_layers,
                      coarse_dynamic_layers=args.coarse_dynamic_layers)
        torch.manual_seed(args.seed + 77)
        static = TwoScaleDiNAT(mode="static", **params)
        torch.manual_seed(args.seed + 77)
        dynamic = TwoScaleDiNAT(mode="dynamic", **params)
        # Exact shared initial parameters (dynamic-only modules remain random).
        static_weights = static.state_dict()
        dynamic.load_state_dict(
            {k: v for k, v in static_weights.items() if k in dynamic.state_dict()},
            strict=False,
        )
        for model in (static, dynamic):
            out = args.output_dir / case / args.operator / model.mode
            result = _train_one(
                model, mesh, mesh_gpu, qb, qb_gpu, train, val, scales,
                projected=projected, device=device,
                seed=args.seed, steps=args.steps, lr=args.learning_rate,
                core_side=args.core_side, heun_steps=args.heun_steps,
                sample_count=args.generated_samples, out_dir=out,
                training_scope=args.training_scope,
            )
            result.update({"case": case, "operator": args.operator, "refinement": args.refinement})
            all_results.append(result)
    report = {
        "schema": "graph_attention.m45_synthetic.v1",
        "scope": "synthetic operator-consistent diagnostic, NOT filtered LES or DNS validation",
        "settings": {**vars(args), "output_dir": str(args.output_dir)},
        "operator_QB_max_error": operator_error,
        "results": all_results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "metrics.csv").open("w", newline="") as f:
        keys = list(all_results[0])
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    print("M45 synthetic results:", args.output_dir / "summary.json", flush=True)


if __name__ == "__main__":
    main()
