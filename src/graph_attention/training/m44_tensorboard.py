"""TensorBoard scalar logging for standalone M44 SR-0 training.

No Lightning Trainer is used by this runner. The writer is opt-in so legacy
training execution and checkpoint numerical behavior remain unchanged.
Logging must run through Singularity on grace for cluster experiments.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def validate_history(rows: list[dict[str, Any]]) -> None:
    """Require consecutive, finite, zero-based epochs before event import."""
    if not isinstance(rows, list):
        raise TypeError("M44 history must be a list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("epoch")) is not int:
            raise ValueError(f"history row {index} lacks an integer epoch")
        if row["epoch"] != index:
            raise ValueError(f"history epochs must be contiguous from zero (index {index})")
        for key in ("train_loss", "validation_loss", "seconds"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"history epoch {index}: missing numeric {key}")
            if not math.isfinite(float(value)):
                raise ValueError(f"history epoch {index}: nonfinite {key}")


def write_epoch(
    writer: Any,
    row: dict[str, Any],
    *,
    best_validation: float,
    learning_rate: float,
    updates_per_epoch: int,
) -> None:
    """Log epoch-level M44 scalar metrics without changing model state."""
    for key, value in (
        ("best_validation", best_validation),
        ("learning_rate", learning_rate),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"nonfinite {key} in TensorBoard metric")
    if updates_per_epoch < 1:
        raise ValueError("optimizer updates per epoch must be positive")
    step = int(row["epoch"])
    writer.add_scalar("loss/train_fm_velocity_mse", float(row["train_loss"]), step)
    writer.add_scalar("loss/validation_fm_velocity_mse", float(row["validation_loss"]), step)
    writer.add_scalar("loss/best_validation_fm_velocity_mse", float(best_validation), step)
    writer.add_scalar("optimizer/learning_rate", float(learning_rate), step)
    writer.add_scalar("optimizer/updates_per_epoch", updates_per_epoch, step)
    writer.add_scalar("time/epoch_seconds", float(row["seconds"]), step)


def open_writer(
    log_dir: Path,
    *,
    previous_history: list[dict[str, Any]],
    resume_step: int | None,
    learning_rate: float,
    updates_per_epoch: int,
    refresh_from_history: bool = False,
) -> Any:
    """Start live logging or import a validated history into one run directory.

    Existing event files are never overwritten for a fresh run or backfill.
    On resume, PyTorch's purge_step hides steps from a partially interrupted
    session at or after the restored checkpoint epoch. If this is a legacy
    run without events, its saved checkpoint history is imported first.
    Explicit history refresh uses a step-0 TensorBoard purge event and replays
    the full saved history, replacing stale curves without touching checkpoints.
    Do not refresh an actively logging native TensorBoard process.
    """
    validate_history(previous_history)
    if resume_step is not None and resume_step != len(previous_history):
        raise ValueError("TensorBoard resume_step differs from checkpoint history length")
    if refresh_from_history and resume_step is not None:
        raise ValueError("TensorBoard history refresh cannot run simultaneously with checkpoint resume")
    existing = log_dir.exists() and any(log_dir.glob("events.out.tfevents.*"))
    if existing and resume_step is None and not refresh_from_history:
        raise FileExistsError(f"refusing to duplicate TensorBoard event files: {log_dir}")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "The Singularity Python environment needs the 'tensorboard' package "
            "(already declared in pyproject.toml)."
        ) from exc
    writer = SummaryWriter(
        log_dir=str(log_dir),
        purge_step=(0 if refresh_from_history else resume_step) if existing else None,
        flush_secs=15,
    )
    if not existing or refresh_from_history:
        best = float("inf")
        for row in previous_history:
            best = min(best, float(row["validation_loss"]))
            write_epoch(
                writer,
                row,
                best_validation=best,
                learning_rate=learning_rate,
                updates_per_epoch=updates_per_epoch,
            )
        writer.flush()
    return writer
