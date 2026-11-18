"""M48.4b: frozen 44-teacher normalization, matched static/dynamic FM training.

Run one architecture per process, with the same source order, patch locations,
Flow Matching times, global projected-noise seeds and optimizer schedule.
This reuses the numerically qualified M48.4a _prepare_pair and _one_step.
No validation/test sources and no MS_SR writes. Does not implement sampling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import torch

from graph_attention.data.ms_sr_adapter import FIELDS, MANIFEST_SHA256
from graph_attention.data.ms_sr_training import (
    _digest, checked_normalization, record_identity, select_role_records,
)
from graph_attention.models.two_scale_dinat import TwoScaleDiNAT
from scripts.m48_ms_sr_train import _one_step, _prepare_pair


DEFAULT_NORMALIZATION = (
    Path(__file__).resolve().parents[1]
    / "configs/m48/teacher_44_w1p25_l1_q4_q8.json"
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: object) -> str:
    return _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _read_frozen(
    path: Path, records: list[dict], selection_sha256: str,
    *, manifest_sha256: str = MANIFEST_SHA256,
) -> tuple[dict, tuple[torch.Tensor, ...], str]:
    """Fail closed on changed source set, filter, channels or frozen statistics.

    The stored source selection is authenticated by the SHA-256 produced
    from all 44 original manifest source/operator identities. Injecting the
    authenticated manifest records permits reuse of checked_normalization.
    """
    identities = [record_identity(r) for r in records]
    if (any(r["role"] != "teacher" or r["source_role"] != "teacher"
            for r in records)
            or _digest(identities) != selection_sha256):
        raise ValueError("frozen M48.4b teacher normalization provenance mismatch")
    raw = path.read_bytes()
    frozen = json.loads(raw)
    if (frozen.get("schema") != "graph_attention.m48_teacher_normalization.v1"
            or frozen.get("role") != "teacher"
            or frozen.get("fields") != list(FIELDS)
            or frozen.get("physical_nondimensionalization")
            != "none; original MS_SR conservative fields"
            or frozen.get("residual_centering")
            != "none; only per-channel RMS scaling"
            or frozen.get("case_prefix") != ""
            or frozen.get("source_count") != 44
            or len(records) != 44
            or frozen.get("case_count") != len({r["case"] for r in records})
            or frozen.get("selection_sha256") != selection_sha256
            or frozen.get("manifest_sha256") != manifest_sha256
            or frozen.get("audit_sha256") is None):
        raise ValueError("frozen M48.4b teacher normalization provenance mismatch")
    payload = dict(frozen)
    payload["sources"] = identities
    scales = checked_normalization(
        payload, records, selection_sha256, manifest_sha256=manifest_sha256,
    )
    return payload, scales, _sha256(raw)


def _epoch_order(source_count: int, seed: int, epoch: int) -> list[int]:
    order = list(range(source_count))
    random.Random(seed + 104729 * epoch).shuffle(order)
    return order


def _schedule_step(
    records: list[dict], index: int, patches_per_source: int, seed: int,
) -> tuple[int, dict, int]:
    """Each source occurs exactly patches_per_source times per epoch."""
    if index < 0 or patches_per_source < 1 or not records:
        raise ValueError("invalid training schedule arguments")
    epoch_size = len(records) * patches_per_source
    epoch, within = divmod(index, epoch_size)
    source_position, patch_number = divmod(within, patches_per_source)
    source_index = _epoch_order(len(records), seed, epoch)[source_position]
    return epoch, records[source_index], patch_number


def _step_draw(seed: int, index: int, fine_side: int) -> tuple[float, tuple[int, int, int], int]:
    """Seed step from global index; independent of model mode and resume."""
    if seed < 0 or index < 0 or fine_side < 1:
        raise ValueError("invalid deterministic FM step")
    rng = torch.Generator(device="cpu").manual_seed(seed + 100_000 + index)
    fm_time = float(0.001 + 0.998 * torch.rand((), generator=rng).item())
    origin = tuple(
        int(torch.randint(fine_side, (1,), generator=rng).item())
        for _ in range(3)
    )
    return fm_time, origin, seed + 1_000_000 + index


def _write_json_atomic(path: Path, payload: dict) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".",
        suffix=".tmp", encoding="utf-8", delete=False,
    ) as out:
        temp = Path(out.name)
        out.write(data)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _save_checkpoint(
    path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    config_sha256: str, completed_steps: int, history: list[dict], device: torch.device,
) -> None:
    state = {
        "schema": "graph_attention.m48_4b_checkpoint.v1",
        "config_sha256": config_sha256,
        "completed_steps": completed_steps,
        "history": history,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        ),
    }
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=path.name + ".",
        suffix=".tmp", delete=False,
    ) as out:
        temp = Path(out.name)
        torch.save(state, out)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _load_checkpoint(
    path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    config_sha256: str, device: torch.device,
) -> tuple[int, list[dict]]:
    # This operation is intended only for self-generated trusted checkpoints.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state.get("schema") != "graph_attention.m48_4b_checkpoint.v1"
            or state.get("config_sha256") != config_sha256):
        raise ValueError("M48.4b checkpoint configuration mismatch")
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda":
        if state["cuda_rng_state"] is None:
            raise ValueError("checkpoint is missing CUDA generator state")
        torch.cuda.set_rng_state(state["cuda_rng_state"], device)
    completed = state["completed_steps"]
    history = state["history"]
    if (not isinstance(completed, int) or completed < 0
            or not isinstance(history, list) or len(history) != completed):
        raise ValueError("inconsistent M48.4b checkpoint step/history")
    return completed, history


def _summary(config: dict, history: list[dict], total: int) -> dict:
    groups: dict[str, list[float]] = {}
    for entry in history:
        groups.setdefault(entry["case"], []).append(entry["loss"])
    return {
        "schema": "graph_attention.m48_4b_train_summary.v1",
        "config_sha256": _canonical_hash(config),
        "mode": config["mode"],
        "completed_steps": len(history),
        "total_planned_steps": total,
        "training_complete": len(history) == total,
        "loss_by_teacher_case": {
            case: {"steps": len(values), "mean": sum(values) / len(values)}
            for case, values in sorted(groups.items())
        },
        "last_step": history[-1] if history else None,
        "total_forward_backward_seconds": sum(
            item["optimizer_step_seconds"] for item in history
        ),
        "history": history,
        "evaluation": "teacher-only training; validation and global generation deferred",
    }


def run(args: argparse.Namespace) -> dict:
    if (args.mode not in ("static", "dynamic")
            or args.epochs < 1 or args.patches_per_source < 1
            or args.core_side < 1 or args.width < 1 or args.heads < 1
            or args.width % args.heads or args.depth < 1
            or args.learning_rate <= 0 or args.grad_clip <= 0
            or args.max_steps < 0 or args.checkpoint_every < 1
            or args.seed < 0):
        raise ValueError("invalid M48.4b training configuration")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    records, selection_sha = select_role_records(
        args.ms_sr_root, role="teacher", filter_id="w1p25_l1",
        case_prefix="", max_sources=0, manifest_sha256=MANIFEST_SHA256,
    )
    normalization, scales, frozen_sha = _read_frozen(
        args.normalization_file, records, selection_sha,
        manifest_sha256=MANIFEST_SHA256,
    )
    # Reuse the original M48.4a full-field loader with its pinned manifest.
    args.manifest_sha256 = MANIFEST_SHA256
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    if dirty:
        raise RuntimeError("M48.4b scientific training requires a clean Git checkout")
    config = {
        "schema": "graph_attention.m48_4b_training_config.v1",
        "git_commit": revision,
        "mode": args.mode,
        "manifest_sha256": MANIFEST_SHA256,
        "selection_sha256": selection_sha,
        "source_count": len(records),
        "source_identities": [record_identity(r) for r in records],
        "normalization_sha256": frozen_sha,
        "normalization_audit_sha256": normalization["audit_sha256"],
        "seed": args.seed,
        "epochs": args.epochs,
        "patches_per_source": args.patches_per_source,
        "core_side": args.core_side,
        "width": args.width,
        "heads": args.heads,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "grad_clip": args.grad_clip,
        "max_fine_read": args.max_fine_read,
        "max_coarse_read": args.max_coarse_read,
        "device": args.device,
        "loss": "owned_core_mean_MSE_equal_source_snapshot_weight",
        "flow": "global_CEDAR_projected_gaussian_linear_FM",
    }
    config_hash = _canonical_hash(config)
    total_steps = len(records) * args.patches_per_source * args.epochs

    if args.resume:
        if not args.output_dir.is_dir():
            raise FileNotFoundError("resume output directory does not exist")
        previous = json.loads((args.output_dir / "config.json").read_text())
        if _canonical_hash(previous) != config_hash:
            raise ValueError("resume configuration differs from original run")
        if json.loads((args.output_dir / "normalization.json").read_text()) != normalization:
            raise ValueError("resume normalization has changed")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write_json_atomic(args.output_dir / "config.json", config)
        _write_json_atomic(args.output_dir / "normalization.json", normalization)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = TwoScaleDiNAT(
        mode=args.mode, channels=5, fine_width=args.width,
        coarse_width=args.width, num_heads=args.heads,
        fine_pre_layers=args.depth, fine_post_layers=args.depth,
        coarse_static_layers=args.depth, coarse_dynamic_layers=args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    checkpoint_path = args.output_dir / "checkpoint_latest.pt"
    if args.resume:
        completed, history = _load_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer,
            config_sha256=config_hash, device=device,
        )
    else:
        completed, history = 0, []

    if completed > total_steps:
        raise ValueError("checkpoint exceeds configured training length")
    last = (
        total_steps if args.max_steps == 0
        else min(total_steps, completed + args.max_steps)
    )
    previous_key = None
    prepared = None
    for step in range(completed, last):
        epoch, record, patch_number = _schedule_step(
            records, step, args.patches_per_source, args.seed,
        )
        key = (record["case"], str(record["snapshot_id"]))
        if key != previous_key:
            prepared = _prepare_pair(args, record, scales)
            previous_key = key
        pair, cedar, coarse, baseline, x1 = prepared
        fm_time, origin, source_seed = _step_draw(args.seed, step, pair.fine.shape[-1])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result = _one_step(
            model, optimizer, args, pair=pair, cedar=cedar,
            coarse=coarse, baseline=baseline, x1=x1,
            source_seed=source_seed, core_origin=origin,
            fm_time=fm_time, device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["optimizer_step_seconds"] = time.perf_counter() - start
        if device.type == "cuda":
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        entry = {
            "step": step + 1, "epoch": epoch + 1,
            "case": record["case"],
            "snapshot_id": str(record["snapshot_id"]),
            "patch_number": patch_number,
            "fm_time": fm_time, "core_origin": origin,
            **result,
        }
        history.append(entry)
        print(json.dumps(entry, allow_nan=False), flush=True)
        if (step + 1) % args.checkpoint_every == 0:
            _save_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer,
                config_sha256=config_hash,
                completed_steps=step + 1, history=history, device=device,
            )
            _write_json_atomic(
                args.output_dir / "summary.json", _summary(config, history, total_steps)
            )

    _save_checkpoint(
        checkpoint_path, model=model, optimizer=optimizer,
        config_sha256=config_hash, completed_steps=last,
        history=history, device=device,
    )
    summary = _summary(config, history, total_steps)
    _write_json_atomic(args.output_dir / "summary.json", summary)
    print(
        f"M48.4b {args.mode} completed {last}/{total_steps} updates; "
        f"frozen source seal {selection_sha}", flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms-sr-root", type=Path, default=Path("/scratch/ai/MS_SR"))
    parser.add_argument("--normalization-file", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("static", "dynamic"), required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patches-per-source", type=int, default=4)
    parser.add_argument("--core-side", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-fine-read", type=int, default=100_000)
    parser.add_argument("--max-coarse-read", type=int, default=50_000)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=4,
                        help="Limit optimizer steps in this invocation; 0 means all remaining")
    parser.add_argument("--seed", type=int, default=484)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
