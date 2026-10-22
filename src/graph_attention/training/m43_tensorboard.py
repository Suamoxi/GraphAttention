"""TensorBoard event logging for M43-B without changing the scientific objective.

All inputs are detached scalar summaries or current optimizer/model state.
The logger does not sample random numbers, update optimizer state, write a
checkpoint, or alter the numerical train/validation losses. No events are
required to resume training: legacy epoch history can seed epoch-level curves.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def completed_optimizer_steps(history: list[Mapping[str, Any]]) -> int:
    """Recover absolute optimizer-step offset from historical M43-B checkpoints."""
    total = 0
    for expected_epoch, record in enumerate(history):
        if int(record["epoch"]) != expected_epoch:
            raise ValueError("M43-B saved history contains noncontiguous epochs")
        count = int(record["train"]["optimizer_steps"])
        if count < 1:
            raise ValueError("each completed M43-B epoch needs optimizer steps")
        total += count
    return total


def log_completed_epoch(
    writer: Any, record: Mapping[str, Any], global_step: int,
) -> None:
    """Emit epoch-level summaries at the *absolute optimizer-step* axis."""
    epoch = int(record["epoch"])
    train = record["train"]
    validation = record["validation"]
    writer.add_scalar("loss/train_epoch_velocity_mse", train["mean"], global_step)
    writer.add_scalar("loss/validation_epoch_velocity_mse", validation["mean"], global_step)
    writer.add_scalar("epoch/index_zero_based", epoch, global_step)
    writer.add_scalar("epoch/wall_seconds", record["epoch_wall_seconds"], global_step)
    writer.add_scalar("epoch/train_samples", train["samples"], global_step)
    writer.add_scalar("epoch/validation_samples", validation["samples"], global_step)
    writer.add_scalar("throughput/physical_samples_per_second",
                      train["physical_samples_per_second"], global_step)
    writer.add_scalar("timing/mean_optimizer_step_seconds",
                      train["mean_optimizer_step_seconds"], global_step)
    for resolution, result in validation["by_resolution"].items():
        writer.add_scalar(f"loss/validation_resolution_{resolution}_mse",
                          result["mse"], global_step)
    for case, result in validation["by_case"].items():
        writer.add_scalar(
            f"loss/validation_case_{case.replace('/', '_')}_mse",
            result["mse"], global_step,
        )
    for case, result in train["by_case"].items():
        writer.add_scalar(
            f"loss/train_epoch_case_{case.replace('/', '_')}_mse",
            result["mse"], global_step,
        )
    for resolution, result in train["by_resolution"].items():
        writer.add_scalar(f"loss/train_epoch_resolution_{resolution}_mse",
                          result["mse"], global_step)
    for name in ("forward_by_resolution", "backward_by_resolution"):
        for resolution, result in train[name].items():
            writer.add_scalar(
                f"timing/{name.replace('_by_resolution', '')}_"
                f"resolution_{resolution}_mean_seconds",
                result["mean_seconds"], global_step,
            )


def gradient_global_l2(model: torch.nn.Module) -> float:
    """Read-only gradient norm, sampled *after* optimizer.step (before zero_grad)."""
    squared = None
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is None:
                continue
            # No in-place operation; gradients and optimizer are unchanged.
            part = param.grad.detach().float().square().sum(dtype=torch.float64)
            squared = part if squared is None else squared + part
        if squared is None:
            return 0.0
        return float(torch.sqrt(squared).item())


def log_completed_optimizer_step(
    writer: Any,
    *,
    epoch: int,
    global_step: int,
    step_mean_velocity_mse: float,
    running_epoch_velocity_mse: float,
    samples_per_step: int,
    elapsed_seconds: float,
    per_case_loss: Mapping[str, float],
    per_resolution_loss: Mapping[str, float],
    forward_per_resolution_seconds: Mapping[str, float],
    backward_per_resolution_seconds: Mapping[str, float],
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    """Log one *completed* optimizer step; all model information is read-only."""
    if global_step < 1:
        raise ValueError("global_step must be positive")
    writer.add_scalar("loss/train_step_velocity_mse", step_mean_velocity_mse, global_step)
    writer.add_scalar("loss/train_running_epoch_velocity_mse",
                      running_epoch_velocity_mse, global_step)
    writer.add_scalar("train/epoch_index_zero_based", epoch, global_step)
    writer.add_scalar("train/samples_per_optimizer_step", samples_per_step, global_step)
    writer.add_scalar("timing/optimizer_step_wall_seconds", elapsed_seconds, global_step)
    for case, loss in per_case_loss.items():
        writer.add_scalar(
            f"loss/train_step_case_{case.replace('/', '_')}_mse",
            loss, global_step,
        )
    for resolution, loss in per_resolution_loss.items():
        writer.add_scalar(f"loss/train_step_resolution_{resolution}_mse",
                          loss, global_step)
    for resolution, seconds in forward_per_resolution_seconds.items():
        writer.add_scalar(f"timing/forward_resolution_{resolution}_step_seconds",
                          seconds, global_step)
    for resolution, seconds in backward_per_resolution_seconds.items():
        writer.add_scalar(f"timing/backward_resolution_{resolution}_step_seconds",
                          seconds, global_step)
    for group_index, group in enumerate(optimizer.param_groups):
        writer.add_scalar(f"optimizer/group_{group_index}_learning_rate",
                          group["lr"], global_step)
    writer.add_scalar("optimizer/gradient_global_l2",
                      gradient_global_l2(model), global_step)
    if device.type == "cuda":
        writer.add_scalar(
            "gpu/allocated_gib", torch.cuda.memory_allocated(device) / 1024**3,
            global_step,
        )
        writer.add_scalar(
            "gpu/reserved_gib", torch.cuda.memory_reserved(device) / 1024**3,
            global_step,
        )
        writer.add_scalar(
            "gpu/peak_allocated_gib",
            torch.cuda.max_memory_allocated(device) / 1024**3,
            global_step,
        )


class M43TensorBoardMonitor:
    """No TensorBoard metadata is placed in scientific checkpoints or signatures."""

    def __init__(
        self,
        run_dir: Path,
        *,
        resume: bool,
        history: list[Mapping[str, Any]],
    ) -> None:
        # Dependency is already used by the legacy train_generative.py runner.
        from torch.utils.tensorboard import SummaryWriter

        self.global_step_offset = completed_optimizer_steps(history)
        # If an interrupted epoch wrote steps beyond last.pt, hide those
        # stale steps when reopening TensorBoard after a resume.
        purge_step = self.global_step_offset + 1 if resume else None
        self.writer = SummaryWriter(
            log_dir=str(run_dir / "tensorboard"),
            purge_step=purge_step,
        )
        if history:
            # Older M43-B checkpoints contain only epoch-level history.
            # We can recover those epoch scalars exactly, but must NEVER
            # fabricate historic per-optimizer-step measurements.
            position = 0
            for record in history:
                position += int(record["train"]["optimizer_steps"])
                log_completed_epoch(self.writer, record, position)
            self.writer.flush()

    def log_step(self, **kwargs: Any) -> None:
        log_completed_optimizer_step(self.writer, **kwargs)

    def log_epoch(self, record: Mapping[str, Any], global_step: int) -> None:
        log_completed_epoch(self.writer, record, global_step)
        self.writer.flush()

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    def __enter__(self) -> "M43TensorBoardMonitor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
