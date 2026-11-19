"""M48.4f: opt-in three-way SR patch execution, with an unchanged FM objective.

Uses M48.4c exact two-scale patch geometry and CEDAR-prepared CPU patches.
Modes differ ONLY in GPU execution and DiT conditioning broadcast.
Each optimizer update averages owned-core losses over the same B patches.
This module does not change M48 scientific training defaults.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence

import torch

from scripts.m48_4c_patch_batch import pack_patches
from scripts.m48_ms_sr_train import _forward_args, _geometry_device

MODES = ("sequential_single", "packed_index", "packed_segmented")


def tensor_sha256(tensors: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().contiguous().cpu()
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def sample_sha256(patches: list[dict]) -> str:
    """Digest exact CPU states, targets, geometry, times and physical context."""
    tensors = []
    for patch in patches:
        geom = patch["geometry"]
        for name in ("state", "baseline", "coarse_state", "target"):
            tensors.append(patch[name])
        tensors.append(torch.tensor([patch["fm_time"]], dtype=torch.float64))
        for name in (
            "fine_global", "coarse_global", "owned_fine_local",
            "fine_coords", "coarse_coords", "fine_edges", "fine_dilated",
            "coarse_edges", "coarse_dilated", "coarse_to_fine",
            "fine_to_coarse", "cross_weights",
        ):
            tensors.append(getattr(geom, name))
    return tensor_sha256(tensors)


def make_gpu_batch(patches: list[dict], device: torch.device, mode: str) -> dict:
    """No artificial graph-index/packing overhead for the sequential reference."""
    if mode not in MODES or not patches:
        raise ValueError("invalid SR patch execution mode or empty microbatch")
    if mode == "sequential_single":
        if len(patches) != 1:
            raise ValueError("sequential_single requires one patch per GPU invocation")
        patch = patches[0]
        return {
            "state": patch["state"].to(device),
            "baseline": patch["baseline"].to(device),
            "coarse_state": patch["coarse_state"].to(device),
            "target": patch["target"].to(device),
            "time": torch.tensor([patch["fm_time"]], dtype=torch.float32, device=device),
            "geometry": _geometry_device(patch["geometry"], device),
            "fine_batch_index": None,
            "coarse_batch_index": None,
            "fine_graph_sizes": None,
            "coarse_graph_sizes": None,
            "owned_sizes": (patch["target"].shape[0],),
        }
    result = pack_patches(patches, device)
    result["fine_graph_sizes"] = (
        tuple(int(p["geometry"].fine_coords.shape[0]) for p in patches)
        if mode == "packed_segmented" else None
    )
    result["coarse_graph_sizes"] = (
        tuple(int(p["geometry"].coarse_coords.shape[0]) for p in patches)
        if mode == "packed_segmented" else None
    )
    return result


def owned_losses(model, batch: dict) -> torch.Tensor:
    """Per-physical-patch owned-core mean squared FM velocity error."""
    geom = batch["geometry"]
    pred = model(
        batch["state"],
        baseline=batch["baseline"],
        coarse_state=batch["coarse_state"],
        time=batch["time"],
        fine_batch_index=batch["fine_batch_index"],
        coarse_batch_index=batch["coarse_batch_index"],
        fine_graph_sizes=batch["fine_graph_sizes"],
        coarse_graph_sizes=batch["coarse_graph_sizes"],
        **_forward_args(geom),
    )
    sq = (pred[geom.owned_fine_local] - batch["target"]).square()
    parts = torch.stack([part.mean() for part in sq.split(batch["owned_sizes"])])
    if not bool(torch.isfinite(parts).all()):
        raise RuntimeError("nonfinite M48.4f owned-core loss")
    return pred[geom.owned_fine_local], parts


def execute_patch_update(
    model, optimizer, patches: list[dict], *,
    mode: str, device: torch.device, grad_clip: float,
    capture_audit: bool = False,
) -> tuple[dict, dict | None]:
    """One equal-patch optimizer step; only packing/launch structure differs."""
    if mode not in MODES or not patches or grad_clip <= 0:
        raise ValueError("invalid M48.4f optimizer step arguments")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    transfer_seconds = 0.0
    forward_backward_seconds = 0.0
    predictions = []
    patch_loss_values = []
    groups = ([p] for p in patches) if mode == "sequential_single" else (patches,)
    for group in groups:
        mark = time.perf_counter()
        batch = make_gpu_batch(group, device, mode)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        transfer_seconds += time.perf_counter() - mark
        mark = time.perf_counter()
        pred_owned, per_patch = owned_losses(model, batch)
        (per_patch.sum() / len(patches)).backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_backward_seconds += time.perf_counter() - mark
        patch_loss_values.extend(float(x) for x in per_patch.detach().cpu())
        if capture_audit:
            predictions.append(pred_owned.detach().cpu())
        del batch, pred_owned, per_patch
    audit = None
    if capture_audit:
        audit = {
            "predictions": torch.cat(predictions),
            "gradients": {
                name: p.grad.detach().cpu().clone() if p.grad is not None else None
                for name, p in model.named_parameters()
            },
        }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    optimizer_mark = time.perf_counter()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not bool(torch.isfinite(grad_norm)) or float(grad_norm) <= 0:
        raise RuntimeError("nonfinite or zero M48.4f accumulated gradient")
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    optimizer_seconds = time.perf_counter() - optimizer_mark
    elapsed_seconds = time.perf_counter() - started
    if capture_audit:
        audit["post_update_parameters"] = {
            name: p.detach().cpu().clone() for name, p in model.named_parameters()
        }
    metrics = {
        "loss": sum(patch_loss_values) / len(patches),
        "patch_losses": patch_loss_values,
        "grad_norm_preclip": float(grad_norm),
        "gpu_microbatch": 1 if mode == "sequential_single" else len(patches),
        "owned_fine_nodes": sum(p["target"].shape[0] for p in patches),
        "fine_read_nodes": sum(p["state"].shape[0] for p in patches),
        "coarse_read_nodes": sum(p["coarse_state"].shape[0] for p in patches),
        "transfer_seconds": transfer_seconds,
        "forward_backward_seconds": forward_backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "update_seconds": elapsed_seconds,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda" else None
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 2**30
            if device.type == "cuda" else None
        ),
    }
    return metrics, audit
