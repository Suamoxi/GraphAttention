"""M48 real MS_SR -> CEDAR -> static/dynamic TwoScaleDiNAT smoke.

One *teacher* full periodic q8 -> q4 sample. CEDAR full-domain algebra and
projected FM noise are qualified in float64 on CPU; the model trains one
core-masked FM optimizer step in float32 on a true periodic, globally indexed
halo patch. This is a software/numerical gate, NOT a convergence or generative
quality result. No source dataset or CEDAR implementation is changed.
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

from graph_attention.data.ms_sr_adapter import (
    MANIFEST_SHA256, flat_channels, load_ms_sr_pair, component_max_relative,
)
from graph_attention.geometry.two_scale_targeted import build_targeted_two_scale_patch
from graph_attention.models.two_scale_dinat import TwoScaleDiNAT


def _device_geometry(geometry, device: torch.device):
    return replace(geometry, **{
        field.name: getattr(geometry, field.name).to(device)
        for field in fields(geometry)
        if isinstance(getattr(geometry, field.name), torch.Tensor)
    })


def _forward_kwargs(geometry):
    return dict(
        fine_coords=geometry.fine_coords, coarse_coords=geometry.coarse_coords,
        fine_edges=geometry.fine_edges, fine_dilated=geometry.fine_dilated,
        coarse_edges=geometry.coarse_edges, coarse_dilated=geometry.coarse_dilated,
        coarse_to_fine=geometry.coarse_to_fine,
        fine_to_coarse=geometry.fine_to_coarse,
        cross_weights=geometry.cross_weights,
        refinement=geometry.refinement,
        periodic_length=geometry.periodic_length,
        coarse_spacing=geometry.coarse_spacing,
    )


def _wrapped_core(nf: int, side: int) -> torch.Tensor:
    """Touch a true periodic seam without imposing periodic patch boundaries."""
    if not 1 <= side <= nf:
        raise ValueError("core side must be in [1, fine_side]")
    axes = [
        (torch.arange(side, dtype=torch.long) + offset) % nf
        for offset in (nf - 1, nf // 2, nf // 2)
    ]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    return torch.unique(
        (coords[:, 0] * nf + coords[:, 1]) * nf + coords[:, 2],
        sorted=True,
    )


def _normalize_one_teacher(pair):
    """One-snapshot *smoke* scales; never claim these are train-population stats."""
    c = flat_channels(pair.coarse)
    baseline = flat_channels(pair.baseline)
    residual = flat_channels(pair.residual)
    mean = c.mean(dim=0)
    std = c.std(dim=0, unbiased=False)
    rms = residual.square().mean(dim=0).sqrt()
    if torch.any(std < 1e-12) or torch.any(rms < 1e-12):
        raise ValueError("degenerate single-teacher normalization")
    return (c-mean)/std, (baseline-mean)/std, residual/rms, {
        "mean_coarse": mean.tolist(), "std_coarse": std.tolist(),
        "rms_residual": rms.tolist(),
        "scope": "one teacher snapshot only; NOT valid scientific training normalization",
    }, rms


def _one_mode(mode, *, pair, c, baseline, xt, target, args, device):
    nc = pair.coarse.shape[-1]
    nf = pair.fine.shape[-1]
    owned = _wrapped_core(nf, args.core_side)
    started = time.perf_counter()
    plan, geom_cpu = build_targeted_two_scale_patch(
        coarse_side=nc, refinement=pair.refinement,
        owned_fine=owned, fine_pre_layers=args.depth,
        fine_post_layers=args.depth, coarse_static_layers=args.depth,
        coarse_dynamic_layers=args.depth if mode == "dynamic" else 0,
        periodic_length=pair.periodic_length,
        max_fine_read=args.max_fine_read,
        max_coarse_read=args.max_coarse_read,
    )
    geometry_seconds = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.model_seed)
    model = TwoScaleDiNAT(
        mode=mode, channels=5, fine_width=args.width, coarse_width=args.width,
        num_heads=args.heads, fine_pre_layers=args.depth,
        fine_post_layers=args.depth, coarse_static_layers=args.depth,
        coarse_dynamic_layers=args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    geom_gpu = _device_geometry(geom_cpu, device)
    fine_ids = plan.fine_read
    coarse_ids = plan.coarse_read
    assert torch.equal(fine_ids[geom_cpu.owned_fine_local], owned)
    t = torch.tensor([args.fm_time], dtype=torch.float32, device=device)
    inputs = xt[fine_ids].to(device=device, dtype=torch.float32)
    b = baseline[fine_ids].to(device=device, dtype=torch.float32)
    coarse = c[coarse_ids].to(device=device, dtype=torch.float32)
    expected = target[owned].to(device=device, dtype=torch.float32)
    started = time.perf_counter()
    prediction = model(
        inputs, baseline=b, coarse_state=coarse,
        time=t, **_forward_kwargs(geom_gpu),
    )
    observed = prediction[geom_cpu.owned_fine_local.to(device)]
    if not torch.isfinite(observed).all():
        raise RuntimeError(f"{mode} generated non-finite FM velocity")
    loss = (observed - expected).square().mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f"{mode} FM loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise RuntimeError(f"{mode} gradients are nonfinite or zero")
    old = model.fine_decoder.final_layer.projection.weight.detach().clone()
    optimizer.step()
    if torch.equal(old, model.fine_decoder.final_layer.projection.weight.detach()):
        raise RuntimeError(f"{mode} one-step optimizer did not update the velocity head")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    return {
        "mode": mode,
        "parameters": sum(p.numel() for p in model.parameters()),
        "owned_fine_nodes": int(plan.owned_fine.numel()),
        "fine_nodes_read": int(plan.fine_read.numel()),
        "coarse_nodes_read": int(plan.coarse_read.numel()),
        "coarse_context_nodes": int(plan.coarse_up_keys.numel()),
        "fine_dynamic_feedback_nodes": int(plan.fine_down_keys.numel()),
        "geometry_seconds": geometry_seconds,
        "fm_loss_before_step": float(loss.detach()),
        "gradient_norm_preclip": float(norm),
        "optimizer_step_verified": True,
        "step_seconds": train_seconds,
        "peak_reserved_cuda_gib": (
            float(torch.cuda.max_memory_reserved(device) / (1024**3))
            if device.type == "cuda" else None
        ),
    }


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output: {args.output_dir}")
    if (args.depth < 1 or args.core_side < 1 or args.width < 1
            or args.heads < 1 or args.width % args.heads or
            args.model_seed < 0 or args.source_seed < 0
            or not 0 < args.fm_time < 1 or not 0 < args.learning_rate < 1):
        raise ValueError("invalid smoke test parameters")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("no CUDA device in allocated SLURM job")
    started = time.perf_counter()
    pair, cedar, operator_checks = load_ms_sr_pair(
        args.ms_sr_root, role="teacher",
        filter_id=args.filter_id,
        case_prefix=args.case_prefix,
        fine_factor=4, coarse_factor=8,
        expected_manifest_sha256=args.manifest_sha256,
    )
    data_seconds = time.perf_counter() - started
    c, baseline, x1, normalization, residual_scales = _normalize_one_teacher(pair)
    # Generate FM source on the entire periodic domain. The source must lie in
    # C's nullspace BEFORE extracting any network patch.
    source_generator = torch.Generator(device="cpu").manual_seed(args.source_seed)
    eps = torch.randn(pair.fine.shape, generator=source_generator, dtype=torch.float64)
    x0 = cedar.project(eps)
    source_defect = component_max_relative(
        cedar.restrict(x0), pair.coarse.new_ones(pair.coarse.shape),
    )
    if source_defect["max_component_scaled_linf"] > 1e-9:
        raise RuntimeError("FM source was not globally projected into ker(C)")
    # Per-component constant scaling commutes with C. X1=R/scale is in ker(C).
    n0 = flat_channels(x0)
    xt = (1.0 - args.fm_time) * n0 + args.fm_time * x1
    target = x1 - n0
    if not torch.isfinite(xt).all() or not torch.isfinite(target).all():
        raise RuntimeError("nonfinite normalized FM state")
    results = [
        _one_mode(mode, pair=pair, c=c, baseline=baseline, xt=xt,
                  target=target, args=args, device=device)
        for mode in ("static", "dynamic")
    ]
    ms_sr_root = args.ms_sr_root.resolve()
    def source_digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    report = {
        "schema": "graph_attention.m48_ms_sr_smoke.v1",
        "provenance": {
            "graph_attention_commit": revision,
            "ms_sr_campaign_data_sha256": source_digest(ms_sr_root / "code" / "campaign_data.py"),
            "ms_sr_torchcedar_sha256": source_digest(ms_sr_root / "code" / "models.py"),
            "ms_sr_cedar_revision": "ae96eece81204b7eb55365eade4838ae1e337b38",
        },
        "scope": "real MS_SR teacher snapshot, operator/geometry/backprop smoke ONLY",
        "checkpoint_training": "none; one optimizer update each mode",
        "scientific_evaluation": "none",
        "ms_sr": {
            "manifest_sha256": pair.manifest_sha256,
            "operator_sha256": pair.operator_sha256,
            "role": pair.metadata["role"],
            "case": pair.metadata["case"],
            "snapshot_id": pair.metadata["snapshot_id"],
            "physical_time_s": pair.metadata["physical_time_s"],
            "filter_id": pair.metadata["filter_id"],
            "fine_factor": pair.metadata["fine_factor"],
            "coarse_factor": pair.metadata["coarse_factor"],
            "input_shape": list(pair.coarse.shape),
            "target_shape": list(pair.fine.shape),
            "periodic_box_length": pair.periodic_length,
            "source_sha256": pair.metadata["source_sha256"],
            "mesh_sha256": pair.metadata["mesh_sha256"],
        },
        "operator_checks_float64": operator_checks,
        "source_projection_float64": source_defect,
        "single_teacher_normalization": normalization,
        "model": {
            "depth": args.depth, "width": args.width, "heads": args.heads,
            "model_seed": args.model_seed, "source_seed": args.source_seed,
            "fm_time": args.fm_time, "core_side": args.core_side,
            "learning_rate": args.learning_rate,
            "device": args.device, "torch_version": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "results": results,
        "data_and_cedar_qualification_seconds": data_seconds,
    }
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms-sr-root", type=Path, default=Path("/scratch/ai/MS_SR"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    parser.add_argument("--case-prefix", default="ReT060/")
    parser.add_argument("--filter-id", choices=sorted(("w1p25_l1", "w1p5_l1")), default="w1p25_l1")
    parser.add_argument("--core-side", type=int, default=2)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fm-time", type=float, default=0.37)
    parser.add_argument("--model-seed", type=int, default=481)
    parser.add_argument("--source-seed", type=int, default=482)
    parser.add_argument("--max-fine-read", type=int, default=100_000)
    parser.add_argument("--max-coarse-read", type=int, default=50_000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
