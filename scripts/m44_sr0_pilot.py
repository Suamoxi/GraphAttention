"""M44 SR-0 one-DNS-snapshot numerical integration pilot (NOT scientific LES).

Loads one real AVBP DNS snapshot, qualifies a Cartesian subvolume, creates a
point-sampled coarse condition, performs a few M37 conditional FM updates, and
saves a diagnostic checkpoint. Never reports held-out reconstruction quality.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import torch
from omegaconf import OmegaConf

from graph_attention.data import AVBPHDF5Dataset, AVBPSampleSpec
from graph_attention.data.sr0_pilot import build_sr0_pilot_pair
from graph_attention.tasks.sr0 import SR0FlowMatching, fit_sr0_scales
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m43_b1_memory_gate import choose_snapshot


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.steps < 1 or args.sample_steps < 1:
        raise ValueError("steps and sample-steps must be positive")
    if args.hidden_dim < 1 or args.layers < 1 or args.heads < 1 or args.hidden_dim % args.heads:
        raise ValueError("invalid DiNAT width, depth or head count")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA pilot but no CUDA device is available")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SR-0 pilot output: {output}")
    case_dir = (Path(args.root) / args.case).resolve(strict=True)
    mesh_file = (case_dir / "MESH" / "mesh.mesh.h5").resolve(strict=True)
    if args.snapshot is None:
        snapshot, _ = choose_snapshot(case_dir)
    else:
        snapshot = Path(args.snapshot).expanduser().resolve(strict=True)
    refs = (Path(args.reference_dir) / (args.case.replace("/", "_") + ".yaml")).resolve(strict=True)
    sample_id = f"{args.case}:{snapshot.stem}:pilot_patch_{'_'.join(map(str, args.origin))}"
    dataset = AVBPHDF5Dataset(
        samples=[AVBPSampleSpec(
            sample_id=sample_id,
            snapshot_file=snapshot,
            mesh_id=f"{args.case}/native_dns_mesh",
            mesh_file=mesh_file,
            case_id=args.case,
        )],
        case_files={args.case: refs},
    )
    dns = dataset[0]
    pair = build_sr0_pilot_pair(
        dns,
        fine_side=args.side,
        patch_cells=args.patch_cells,
        refinement=args.refinement,
        origin=tuple(args.origin),
    )
    # One-sample scaler is diagnostic only: NOT train/validation/test statistics.
    scales = fit_sr0_scales([pair], train_sample_ids=(sample_id,))
    task = SR0FlowMatching(scales, time_embedding_scale=1000.0)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pair = pair.to(device=device, dtype=torch.float32)
    rng = torch.Generator(device=device).manual_seed(args.seed + 1000)
    probe = task.make_training_problem(pair, generator=rng)
    model_cfg = OmegaConf.load(
        str(Path(__file__).resolve().parents[1] / "configs" / "model" / "dinat_dit.yaml")
    )
    model_cfg.hidden_dim = args.hidden_dim
    model_cfg.num_layers = args.layers
    model_cfg.num_heads = args.heads
    model_cfg.sparse_attention_backend = "scatter"
    model_cfg.activation_checkpointing = bool(args.checkpointing)
    model, initialization = instantiate_controlled_model(
        model_cfg, task.model_probe(probe), seed=args.seed,
    )
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    losses = []
    model.train()
    for _ in range(args.steps):
        problem = task.make_training_problem(pair, generator=rng)
        optimizer.zero_grad(set_to_none=True)
        aggregate = task.training_loss(model, problem)
        loss = aggregate.mean
        if not bool(torch.isfinite(loss)):
            raise ValueError("pilot Flow Matching loss is non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.inference_mode():
        generated = task.generate(
            model,
            replace(pair, fine_target=None),
            steps=args.sample_steps,
            solver="heun",
            sampling_seed=args.seed + 2000,
        )
    if not bool(torch.isfinite(generated).all()):
        raise ValueError("SR-0 pilot sampled field contains NaN/Inf")

    # Restriction Q_sample = point selection, NOT the dnstoles physical Q.
    counts = pair.transfer.crow_indices[1:] - pair.transfer.crow_indices[:-1]
    owned_fine = torch.nonzero(counts == 1).flatten()
    coarse_col = pair.transfer.col_indices[pair.transfer.crow_indices[owned_fine]]
    exact_training_match = (
        pair.fine_target[owned_fine] - pair.coarse_state[coarse_col]
    ).abs().amax().item()
    sampled_condition_error = (
        generated[owned_fine] - pair.coarse_state[coarse_col]
    ).abs().amax().item()
    if exact_training_match > 1e-5:
        raise RuntimeError("selected coarse nodes do not match their DNS targets")

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    ).stdout.strip() or None
    report = {
        "status": "integration_smoke_passed",
        "scientific_validity": "NOT LES; no physical filtering or independent validation",
        "coarse_observation": "point sampling on the original DNS Cartesian nodes",
        "baseline": "trilinear interpolation; no physical Q/B consistency claim",
        "source_snapshot": str(snapshot),
        "source_mesh": str(mesh_file),
        "case_references": str(refs),
        "sample_id": sample_id,
        "git_revision": revision,
        "case": args.case,
        "fine_original_nodes": args.side**3,
        "fine_patch_nodes": pair.fine.coords.shape[0],
        "coarse_nodes": pair.coarse_state.shape[0],
        "refinement": args.refinement,
        "patch_cells": args.patch_cells,
        "origin": list(args.origin),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_configuration": OmegaConf.to_container(model_cfg, resolve=True),
        "initialization": initialization,
        "steps": args.steps,
        "train_loss_trace": losses,
        "sample_heun_steps": args.sample_steps,
        "training_point_sample_match_max_abs": exact_training_match,
        "generated_point_sample_mismatch_max_abs": sampled_condition_error,
        "device": args.device,
        "seed": args.seed,
        "caution": "Single snapshot used for fitting and updating; losses are not test metrics",
    }
    output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "pilot_only": True,
        "model_state_dict": model.cpu().state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": OmegaConf.to_container(model_cfg, resolve=True),
        "scales": scales,
        "sample_id": sample_id,
        "git_revision": revision,
    }, output / "pilot_checkpoint.pt")
    torch.save({
        "generated_dimensionless_state": generated.cpu(),
        "fine_dimensionless_target": pair.fine_target.cpu(),
        "coarse_dimensionless_state": pair.coarse_state.cpu(),
        "fine_dimensionless_coords": pair.fine.coords.cpu(),
    }, output / "pilot_sample.pt")
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", default="Re54/Ma0.3")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--side", type=int, default=65)
    parser.add_argument("--origin", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--patch-cells", type=int, default=8)
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
