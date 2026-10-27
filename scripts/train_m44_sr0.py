"""M44 SR-0.1: small-patch, multi-snapshot conditional Flow Matching training.

This is a RESEARCH BASELINE using point-sampled DNS as a proxy coarse state.
It is NOT a physical LES restriction and cannot establish SGS validity.

The run enforces snapshot-disjoint chronological train/validation/test splits,
train-only scaling, held-out validation, best-checkpoint selection, resumable
optimizer/RNG state and deterministic conditional generation diagnostics.
It reuses M37, the existing FlowMatchingTask, and the M44 SR-0 pair contract.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import torch
from omegaconf import OmegaConf

from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, HITCaseSpec
from graph_attention.data.sr0_pilot import build_sr0_pilot_pair
from graph_attention.data.sr0_training import (
    choose_patch_origins,
    select_chronological_indices,
    split_chronological_indices,
)
from graph_attention.geometry.sr0_cartesian import build_sr0_cartesian_patch
from graph_attention.tasks.sr0 import SR0FlowMatching, SR0PairBatch, fit_sr0_scales
from graph_attention.training.model_factory import instantiate_controlled_model

_FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def _revision() -> str | None:
    run = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return run.stdout.strip() or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    info = path.stat()
    return {"path": str(path.resolve(strict=True)), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def _save_torch(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _scales_dict(scales: Any) -> dict[str, object]:
    return {
        "coarse_mean": scales.coarse_mean.cpu(),
        "coarse_std": scales.coarse_std.cpu(),
        "residual_rms": scales.residual_rms.cpu(),
        "train_sample_ids": scales.train_sample_ids,
    }


def _check_scales(scales: Any, saved: dict[str, Any]) -> None:
    current = _scales_dict(scales)
    for key in ("coarse_mean", "coarse_std", "residual_rms"):
        if not torch.equal(current[key], saved[key]):
            raise ValueError(f"resume train-only statistics mismatch: {key}")
    if current["train_sample_ids"] != tuple(saved["train_sample_ids"]):
        raise ValueError("resume train sample IDs differ")


def _manifest(dataset: AVBPMultiCaseHITDataset, args: argparse.Namespace) -> dict[str, Any]:
    selection = select_chronological_indices(len(dataset), args.max_snapshots)
    split = split_chronological_indices(
        selection,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        embargo_snapshots=args.embargo_snapshots,
    )
    origins = choose_patch_origins(
        side=args.side,
        patch_cells=args.patch_cells,
        refinement=args.refinement,
        patches_per_snapshot=args.patches_per_snapshot,
        seed=args.seed,
        boundary_margin=args.boundary_margin,
    )
    records = {}
    for name in ("train", "validation", "test", "embargo"):
        records[name] = [
            {"sample_id": dataset.sample_ids[i], **_file_record(Path(dataset.sample_specs[i].snapshot_file))}
            for i in getattr(split, name)
        ]
    reference = dataset.case_definitions[args.case].source_path
    mesh = Path(dataset.sample_specs[0].mesh_file)
    return {
        "protocol": "M44_SR0.1_POINT_SAMPLED_DNS_NOT_LES",
        "case": args.case,
        "root": str(Path(args.root).resolve()),
        "side": args.side,
        "patch_cells": args.patch_cells,
        "refinement": args.refinement,
        "patch_origins": [list(o) for o in origins],
        "boundary_margin": args.boundary_margin,
        "snapshot_indices": {name: list(getattr(split, name)) for name in records},
        "snapshots": records,
        "reference_file": {**_file_record(Path(reference)), "sha256": _sha256(Path(reference))},
        "mesh_file": _file_record(mesh),
        "spatial_condition_mode": args.condition_mode,
        "absolute_coordinate_embedding": args.absolute_coordinate_embedding,
        "time_embedding_scale": args.time_embedding_scale,
        "model": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.layers,
            "num_heads": args.heads,
            "sparse_attention_backend": args.sparse_attention_backend,
            "activation_checkpointing": args.checkpointing,
        },
        "optimizer": {"name": "AdamW", "lr": args.lr},
        "seed": args.seed,
        "temporal_policy": (
            "Chronological filename counters, fixed patch origins, selected-snapshot embargo; "
            "no validated physical eddy-turnover-time separation"
        ),
        "split_unit": "entire source DNS snapshot; all spatial patches share the split",
        "coarse_observation": "point-sampled DNS nodes; no physical filtering",
    }


def _build_pairs(
    dataset: AVBPMultiCaseHITDataset,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, list[SR0PairBatch]]:
    origins = [tuple(o) for o in manifest["patch_origins"]]
    all_indices = set()
    for name in ("train", "validation", "test"):
        all_indices.update(manifest["snapshot_indices"][name])
    if not all_indices:
        raise ValueError("empty selected snapshot set")

    # All snapshots of one case use its immutable native mesh. Validate each
    # distinct patch geometry only once, then reuse it across source snapshots.
    first_dns = dataset[min(all_indices)]
    native = first_dns.mesh
    geometry = {}
    for origin in origins:
        geometry[origin] = build_sr0_cartesian_patch(
            native.coords,
            native.cell_connectivity,
            side=args.side,
            patch_cells=args.patch_cells,
            refinement=args.refinement,
            origin=origin,
        )

    pairs = {name: [] for name in ("train", "validation", "test")}
    for name in pairs:
        for index in manifest["snapshot_indices"][name]:
            dns = first_dns if index == min(all_indices) else dataset[index]
            for origin in origins:
                suffix = "_".join(str(k) for k in origin)
                source = replace(dns, sample_id=f"{dns.sample_id}/sr0_patch_{suffix}")
                pair = build_sr0_pilot_pair(
                    source,
                    fine_side=args.side,
                    patch_cells=args.patch_cells,
                    refinement=args.refinement,
                    origin=origin,
                    prepared_geometry=geometry[origin],
                )
                pairs[name].append(pair)
    return pairs


def _evaluate(
    model: torch.nn.Module,
    task: SR0FlowMatching,
    pairs: list[SR0PairBatch],
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for pair in pairs:
            prepared = task.make_validation_problem(pair.to(device=device, dtype=torch.float32))
            result = task.training_loss(model, prepared)
            total += float(result.mean.cpu())
    return total / len(pairs)


def _forward_condition(
    model: torch.nn.Module,
    problem: Any,
    spatial: torch.Tensor,
) -> torch.Tensor:
    flow = problem.flow
    return model(
        torch.cat((flow.inputs, spatial), dim=1),
        edge_index=flow.edge_index,
        coords=flow.coords,
        batch_index=flow.batch_index,
        conditioning=flow.conditioning,
        attention_edge_indices=flow.attention_edge_indices,
    )


def _conditional_sensitivity(
    model: torch.nn.Module,
    task: SR0FlowMatching,
    pair: SR0PairBatch,
) -> dict[str, float]:
    """Fixed z_t and t, deliberately spatially permuted coarse condition.

    A nonzero response is NECESSARY, not sufficient, for conditional learning.
    The permutation is deliberately nonphysical and not a quality metric.
    """
    problem = task.make_validation_problem(pair)
    model.eval()
    with torch.inference_mode():
        current = _forward_condition(model, problem, problem.spatial_condition)
        permutation = torch.arange(problem.spatial_condition.shape[0] - 1, -1, -1,
                                   device=problem.spatial_condition.device)
        perturbed = _forward_condition(model, problem, problem.spatial_condition[permutation])
    return {
        "mean_abs_velocity_change": float((current - perturbed).abs().mean().cpu()),
        "velocity_rms": float(current.square().mean().sqrt().cpu()),
        "interpretation": "nonphysical condition permutation; sensitivity only",
    }


def _sample_diagnostic(
    model: torch.nn.Module,
    task: SR0FlowMatching,
    pair: SR0PairBatch,
    *,
    steps: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    model.eval()
    with torch.inference_mode():
        sampled = task.generate(
            model, replace(pair, fine_target=None),
            steps=steps, solver="heun", sampling_seed=seed,
        )
    # Point-selection Q_sample; NOT the dnstoles filtered LES restriction.
    counts = pair.transfer.crow_indices[1:] - pair.transfer.crow_indices[:-1]
    selected = torch.nonzero(counts == 1).flatten()
    coarse_col = pair.transfer.col_indices[pair.transfer.crow_indices[selected]]
    if selected.numel() != pair.coarse_state.shape[0]:
        raise RuntimeError("not every coarse node was retained exactly once in the fine patch")
    error = sampled[selected] - pair.coarse_state[coarse_col]
    baseline_error = pair.fine_baseline[selected] - pair.coarse_state[coarse_col]
    diag = {
        "num_coarse_nodes": int(selected.numel()),
        "baseline_point_match_max_abs": float(baseline_error.abs().amax().cpu()),
        "generated_point_sample_max_abs_by_field": {
            field: float(error[:, i].abs().amax().cpu()) for i, field in enumerate(_FIELDS)
        },
        "generated_point_sample_rms_by_field": {
            field: float(error[:, i].square().mean().sqrt().cpu())
            for i, field in enumerate(_FIELDS)
        },
        "generated_residual_rms_by_field": {
            field: float((sampled[:, i] - pair.fine_baseline[:, i]).square().mean().sqrt().cpu())
            for i, field in enumerate(_FIELDS)
        },
        "reference_residual_rms_by_field": {
            field: float((pair.fine_target[:, i] - pair.fine_baseline[:, i]).square().mean().sqrt().cpu())
            for i, field in enumerate(_FIELDS)
        },
        "warning": "one stochastic draw, not a paired prediction or distributional validation",
    }
    artifact = {
        "generated_dimensionless_state": sampled.cpu(),
        "fine_dimensionless_target": pair.fine_target.cpu(),
        "coarse_dimensionless_state": pair.coarse_state.cpu(),
        "baseline_dimensionless_state": pair.fine_baseline.cpu(),
        "fine_dimensionless_coords": pair.fine.coords.cpu(),
        "transfer_crow_indices": pair.transfer.crow_indices.cpu(),
        "transfer_col_indices": pair.transfer.col_indices.cpu(),
        "transfer_weights": pair.transfer.weights.cpu(),
        "sample_id": pair.fine.source.sample_ids[0],
    }
    return diag, artifact


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1 or args.sample_steps < 1 or args.max_snapshots < 3:
        raise ValueError("epochs, sample-steps and max-snapshots must be positive")
    if args.hidden_dim < 1 or args.layers < 1 or args.heads < 1 or args.hidden_dim % args.heads:
        raise ValueError("invalid DiNAT width, depth or heads")
    if args.lr <= 0:
        raise ValueError("learning rate must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required by selected --device")
    output = Path(args.output).expanduser().resolve()
    if args.resume and not output.is_dir():
        raise FileNotFoundError("--resume requires an existing run directory")
    if not args.resume and output.exists():
        raise FileExistsError(f"refusing to overwrite SR-0 run directory: {output}")

    dataset = AVBPMultiCaseHITDataset(
        root=args.root,
        cases=(HITCaseSpec(args.case, args.side),),
        reference_dir=args.reference_dir,
        require_references=True,
        expected_snapshots_per_case=None,
    )
    manifest = _manifest(dataset, args)
    if args.resume:
        prior = json.loads((output / "manifest.json").read_text())
        if manifest != prior:
            raise ValueError("SR-0 run signature changed on resume; use a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _save_json(output / "manifest.json", manifest)

    started = time.perf_counter()
    print("[M44 SR-0.1] Loading fixed, snapshot-disjoint small-patch dataset", flush=True)
    pairs = _build_pairs(dataset, manifest, args)
    train_ids = tuple(pair.fine.source.sample_ids[0] for pair in pairs["train"])
    scales = fit_sr0_scales(pairs["train"], train_sample_ids=train_ids)
    task = SR0FlowMatching(
        scales,
        validation_seed=args.seed + 2000,
        time_embedding_scale=args.time_embedding_scale,
        spatial_condition_mode=args.condition_mode,
    )
    print(f"[M44 SR-0.1] {len(pairs['train'])} train, "
          f"{len(pairs['validation'])} validation, {len(pairs['test'])} test patches", flush=True)
    if args.resume:
        saved_scaler = torch.load(output / "scales.pt", map_location="cpu", weights_only=True)
        _check_scales(scales, saved_scaler)
    else:
        _save_torch(output / "scales.pt", _scales_dict(scales))

    seed = args.seed
    torch.manual_seed(seed)
    cfg = OmegaConf.load(str(Path(__file__).resolve().parents[1] / "configs/model/dinat_dit.yaml"))
    cfg.hidden_dim = args.hidden_dim
    cfg.num_layers = args.layers
    cfg.num_heads = args.heads
    cfg.sparse_attention_backend = args.sparse_attention_backend
    cfg.activation_checkpointing = bool(args.checkpointing)
    cfg.use_absolute_coordinate_embedding = bool(args.absolute_coordinate_embedding)

    rng = torch.Generator(device=device).manual_seed(seed + 1000)
    probe_pair = pairs["train"][0].to(device=device, dtype=torch.float32)
    probe_problem = task.make_training_problem(probe_pair, generator=rng)
    model, initialization = instantiate_controlled_model(cfg, task.model_probe(probe_problem), seed=seed)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    effective_cfg = OmegaConf.to_container(cfg, resolve=True)
    effective_cfg.update({
        "in_channels": model.in_channels, "out_channels": model.out_channels,
        "spatial_dim": model.spatial_dim, "conditioning_channels": model.conditioning_channels,
    })
    print(f"[M44 SR-0.1] M37 input channels={model.in_channels} "
          f"absolute coordinates={model.use_absolute_coordinate_embedding}", flush=True)

    history: list[dict[str, object]] = []
    best_val = float("inf")
    best_epoch = -1
    first_epoch = 0
    if args.resume:
        saved = torch.load(output / "last.pt", map_location=device, weights_only=True)
        if saved["manifest_sha256"] != _sha256(output / "manifest.json"):
            raise ValueError("SR-0 checkpoint manifest SHA mismatch")
        if saved["model_config"] != effective_cfg:
            raise ValueError("SR-0 checkpoint model config mismatch")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        rng.set_state(saved["training_generator_state"].to(device="cpu"))
        history = list(saved["history"])
        best_val = float(saved["best_validation_loss"])
        best_epoch = int(saved["best_epoch"])
        first_epoch = int(saved["epoch"]) + 1
        if first_epoch >= args.epochs:
            print("[M44 SR-0.1] No more epochs requested; evaluating saved best", flush=True)

    for epoch in range(first_epoch, args.epochs):
        order = list(range(len(pairs["train"])))
        random.Random(seed + epoch).shuffle(order)
        model.train()
        train_sum = 0.0
        start = time.perf_counter()
        for index in order:
            pair = pairs["train"][index].to(device=device, dtype=torch.float32)
            problem = task.make_training_problem(pair, generator=rng)
            optimizer.zero_grad(set_to_none=True)
            aggregate = task.training_loss(model, problem)
            if not torch.isfinite(aggregate.mean):
                raise ValueError(f"nonfinite training objective at epoch {epoch}")
            aggregate.mean.backward()
            optimizer.step()
            train_sum += float(aggregate.mean.detach().cpu())
        train_loss = train_sum / len(order)
        val_loss = _evaluate(model, task, pairs["validation"], device)
        if not (float("-inf") < val_loss < float("inf")):
            raise ValueError("nonfinite validation loss")
        history.append({
            "epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
            "seconds": time.perf_counter() - start,
        })
        better = val_loss < best_val
        if better:
            best_val, best_epoch = val_loss, epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_generator_state": rng.get_state(),
            "history": history,
            "best_validation_loss": best_val,
            "best_epoch": best_epoch,
            "model_config": effective_cfg,
            "initialization": initialization,
            "train_scales": _scales_dict(scales),
            "manifest_sha256": _sha256(output / "manifest.json"),
            "git_revision": _revision(),
        }
        _save_torch(output / "last.pt", checkpoint)
        if better:
            _save_torch(output / "best.pt", checkpoint)
        _save_json(output / "history.json", {"history": history})
        print(f"epoch={epoch:03d} train_fm_mse={train_loss:.8f} "
              f"val_fm_mse={val_loss:.8f} seconds={history[-1]['seconds']:.2f}", flush=True)

    best = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"], strict=True)
    test_loss = _evaluate(model, task, pairs["test"], device)
    val_pair = pairs["validation"][0].to(device=device, dtype=torch.float32)
    sensitivity = _conditional_sensitivity(model, task, val_pair)
    diagnostic, sampled = _sample_diagnostic(
        model, task, val_pair, steps=args.sample_steps, seed=seed + 3000,
    )
    _save_torch(output / "validation_sample.pt", sampled)
    summary = {
        "status": "sr0_multisnapshot_train_complete",
        "scientific_validity": "POINT-SAMPLED DNS PROXY, NOT LES; snapshot-held-out numeric validation only",
        "git_revision": _revision(),
        "case": args.case,
        "snapshot_counts": {key: len(manifest["snapshot_indices"][key]) for key in
                            ("train", "validation", "test", "embargo")},
        "patch_counts": {name: len(pairs[name]) for name in pairs},
        "fine_nodes_per_patch": args.patch_cells + 1,
        "fine_total_nodes_per_patch": (args.patch_cells + 1)**3,
        "coarse_total_nodes_per_patch": (args.patch_cells // args.refinement + 1)**3,
        "refinement": args.refinement,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_configuration": effective_cfg,
        "spatial_condition_mode": args.condition_mode,
        "best_epoch": best_epoch,
        "best_validation_flow_velocity_mse": best_val,
        "test_flow_velocity_mse_at_best_validation": test_loss,
        "conditional_sensitivity": sensitivity,
        "validation_generation_diagnostic": diagnostic,
        "sampling_heun_steps": args.sample_steps,
        "epochs_completed": len(history),
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "notes": [
            "Selection is by source snapshot; not by spatial patch",
            "Validation/test losses use deterministic FM noises/times and are not physical SR quality metrics",
            "No independent coarse LES, physical filter, conservation projection or positivity guarantee",
            "Fixed patch origins; temporal embargo measured in selected snapshots, not physical eddy times",
        ],
    }
    _save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--case", default="Re54/Ma0.3")
    parser.add_argument("--side", type=int, default=65)
    parser.add_argument("--patch-cells", type=int, default=8)
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--patches-per-snapshot", type=int, default=2)
    parser.add_argument("--boundary-margin", type=int, default=1)
    parser.add_argument("--max-snapshots", type=int, default=24)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--embargo-snapshots", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-embedding-scale", type=float, default=1000.0)
    parser.add_argument("--condition-mode", choices=("interpolated_only", "interpolated_plus_baseline"),
                        default="interpolated_only")
    parser.add_argument("--absolute-coordinate-embedding", action="store_true")
    parser.add_argument("--sparse-attention-backend", choices=("scatter", "torch_sparse", "dgl"),
                        default="scatter")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
