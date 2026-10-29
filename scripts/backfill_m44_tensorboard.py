"""Create TensorBoard event files from an existing M44 history.json.

Run ONLY via scripts/slurm/m44_tensorboard_backfill.slurm on grace using
Singularity. Never run this Python script directly on a cluster login node.
This command does not retrain or modify the M44 checkpoint/summary files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graph_attention.training.m44_tensorboard import open_writer, validate_history


def run(run_dir: Path, *, refresh: bool = False) -> dict[str, object]:
    source = run_dir.expanduser().resolve(strict=True)
    manifest_path = source / "manifest.json"
    history_path = source / "history.json"
    if not manifest_path.is_file() or not history_path.is_file():
        raise FileNotFoundError("M44 run requires both manifest.json and history.json")
    manifest = json.loads(manifest_path.read_text())
    payload = json.loads(history_path.read_text())
    if manifest.get("protocol") != "M44_SR0.1_POINT_SAMPLED_DNS_NOT_LES":
        raise ValueError("refusing TensorBoard backfill for an unsupported experiment protocol")
    rows = payload["history"]
    validate_history(rows)
    if not rows:
        raise ValueError("history.json has no completed epoch to import")
    lr = float(manifest["optimizer"]["lr"])
    budget = int(
        manifest.get("train_patch_schedule", {}).get(
            "patches_per_snapshot_per_epoch", len(manifest["patch_origins"])
        )
    )
    updates = len(manifest["snapshot_indices"]["train"]) * budget
    log_dir = source / "tensorboard"
    with open_writer(
        log_dir, previous_history=rows, resume_step=None,
        learning_rate=lr, updates_per_epoch=updates,
        refresh_from_history=refresh,
    ):
        pass
    output: dict[str, object] = {
        "status": "m44_tensorboard_history_backfilled",
        "run_dir": str(source),
        "tensorboard_dir": str(log_dir),
        "epochs_imported": len(rows),
        "first_epoch": int(rows[0]["epoch"]),
        "last_epoch": int(rows[-1]["epoch"]),
        "optimizer_updates_per_epoch": updates,
        "refreshed_existing_events": bool(refresh),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true",
                        help="replace historical curves using TensorBoard's purge-step semantics")
    args = parser.parse_args()
    result = run(args.run_dir, refresh=args.refresh)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
