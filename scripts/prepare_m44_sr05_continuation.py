"""Prepare an immutable-copy M44 SR-0.5 optimizer-state continuation.

Runs ONLY inside Singularity via grace Slurm; never execute on the login
node. Copies existing last.pt, best.pt, scales.pt, manifest.json and
history.json into a new run directory. The original training runs remain
unchanged. Training then resumes from the full AdamW/RNG state.

This is point-sampled DNS, NOT physically filtered LES.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from scripts.train_m44_sr0 import _sha256

_PROTOCOL = "M44_SR05_IMMUTABLE_CONTINUATION_V1"
_FILES = ("manifest.json", "last.pt", "best.pt", "scales.pt", "history.json")


def _load_source(
    source: Path, mode: str, initial_epochs: int, *, production_profile: bool = True
) -> dict[str, Any]:
    manifest_path = source / "manifest.json"
    for filename in _FILES:
        path = source / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing source training artifact: {path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != "M44_SR0.1_POINT_SAMPLED_DNS_NOT_LES":
        raise ValueError("source run is not the M44 point-sampling proxy")
    if manifest.get("point_sample_constraint") is not True:
        raise ValueError("source continuation must retain intrinsic point constraint")
    diversified = "train_patch_pool_origins" in manifest
    if diversified != (mode == "diverse"):
        raise ValueError("continuation mode disagrees with source training data policy")
    expected = {
        "case": "Re54/Ma0.3",
        "side": 65,
        "patch_cells": 8,
        "refinement": 2,
        "seed": 42,
        "spatial_condition_mode": "interpolated_only",
    }
    if production_profile:
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"unexpected source {key}: {manifest.get(key)!r}")
        if {key: len(manifest["snapshot_indices"][key])
                for key in ("train", "validation", "test", "embargo")} != {
                    "train": 15, "validation": 3, "test": 4, "embargo": 2
                }:
            raise ValueError("source chronological split has changed")
    if len(manifest["patch_origins"]) != 2:
        raise ValueError("expected two fixed validation/test origins")
    model = manifest["model"]
    if production_profile and (
        model.get("hidden_dim") != 64 or model.get("num_layers") != 4
        or model.get("num_heads") != 4
        or model.get("sparse_attention_backend") != "scatter"
        or model.get("activation_checkpointing") is not False
    ):
        raise ValueError("source model architecture differs from matched M44 baseline")
    if (
        manifest["optimizer"]["name"] != "AdamW"
        or float(manifest["optimizer"]["lr"]) != 1e-3
    ):
        raise ValueError("source optimizer is not the matched constant-rate AdamW")
    if diversified:
        schedule = manifest.get("train_patch_schedule")
        frozen = manifest.get("frozen_training_scales")
        if (
            not isinstance(schedule, dict)
            or schedule.get("patches_per_snapshot_per_epoch") != 2
            or (production_profile and schedule.get("pool_size") != 16)
            or (production_profile and len(manifest["train_patch_pool_origins"]) != 16)
            or not isinstance(frozen, dict)
        ):
            raise ValueError("source SR-0.4 patch schedule/frozen normalization mismatch")
        base = Path(frozen["source_run_dir"]).resolve(strict=True)
        if _sha256(base / "manifest.json") != frozen["source_manifest_sha256"]:
            raise ValueError("source SR-0.4 frozen baseline manifest changed")
        if _sha256(base / "scales.pt") != frozen["source_scales_sha256"]:
            raise ValueError("source SR-0.4 frozen baseline scales changed")

    last = torch.load(source / "last.pt", map_location="cpu", weights_only=True)
    best = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    original_history = json.loads((source / "history.json").read_text())["history"]
    if not (
        last.get("epoch") == initial_epochs - 1
        and len(last.get("history", [])) == initial_epochs
        and original_history == last["history"]
        and best.get("best_epoch", -1) < initial_epochs
    ):
        raise ValueError("initial checkpoint epoch and history are incompatible")
    if last["manifest_sha256"] != _sha256(manifest_path):
        raise ValueError("source last checkpoint manifest SHA mismatch")
    if best["manifest_sha256"] != last["manifest_sha256"]:
        raise ValueError("source best checkpoint manifest SHA mismatch")
    if "optimizer_state_dict" not in last or "training_generator_state" not in last:
        raise ValueError("full optimizer/RNG state required for exact continuation")
    if best["model_config"] != last["model_config"]:
        raise ValueError("source best and last model configurations differ")
    source_scales = torch.load(source / "scales.pt", map_location="cpu", weights_only=True)
    for key in ("coarse_mean", "coarse_std", "residual_rms"):
        if not torch.equal(last["train_scales"][key], source_scales[key]):
            raise ValueError(f"checkpoint and scales differ for {key}")
    if tuple(last["train_scales"]["train_sample_ids"]) != tuple(
        source_scales["train_sample_ids"]
    ):
        raise ValueError("checkpoint and scaler training IDs differ")

    return {
        "protocol": _PROTOCOL,
        "mode": mode,
        "source_run_dir": str(source),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_last_sha256": _sha256(source / "last.pt"),
        "source_best_sha256": _sha256(source / "best.pt"),
        "source_scales_sha256": _sha256(source / "scales.pt"),
        "source_history_sha256": _sha256(source / "history.json"),
        "initial_epochs": initial_epochs,
        "optimizer_updates_per_epoch": len(manifest["snapshot_indices"]["train"]) * 2,
        "initial_optimizer_updates": initial_epochs * len(manifest["snapshot_indices"]["train"]) * 2,
        "training_data_policy": (
            "16-origin cyclic pool, 2 per snapshot per epoch"
            if diversified else "2 fixed origins per snapshot"
        ),
        "source_frozen_baseline_run": (
            manifest["frozen_training_scales"]["source_run_dir"] if diversified else None
        ),
        "scientific_validity": "POINT_SAMPLED_DNS_PROXY_NOT_PHYSICALLY_FILTERED_LES",
    }


def prepare(
    source_run: Path,
    output: Path,
    *,
    mode: str,
    initial_epochs: int = 50,
    target_epochs: int = 150,
    resume: bool = False,
    production_profile: bool = True,
) -> dict[str, Any]:
    if mode not in ("baseline", "diverse"):
        raise ValueError("mode must be 'baseline' or 'diverse'")
    if initial_epochs < 1 or target_epochs <= initial_epochs:
        raise ValueError("target_epochs must exceed initial_epochs")
    source = source_run.expanduser().resolve(strict=True)
    target = output.expanduser().resolve()
    if source == target:
        raise ValueError("continuation must use a separate output directory")
    parent = _load_source(
        source, mode, initial_epochs, production_profile=production_profile
    )
    record = {
        **parent,
        "target_epochs": target_epochs,
        "target_optimizer_updates": target_epochs * parent["optimizer_updates_per_epoch"],
    }
    provenance_path = target / "continuation_provenance.json"
    if resume:
        if not provenance_path.is_file():
            raise FileNotFoundError("continuation resume requires existing provenance")
        if json.loads(provenance_path.read_text()) != record:
            raise ValueError("source checkpoint or continuation configuration changed")
        if not all((target / file).is_file() for file in _FILES):
            raise FileNotFoundError("continuation resume artifacts are incomplete")
        last = torch.load(target / "last.pt", map_location="cpu", weights_only=True)
        if last["manifest_sha256"] != parent["source_manifest_sha256"]:
            raise ValueError("resumed run manifest differs from original training run")
        if last["epoch"] < initial_epochs - 1 or last["epoch"] >= target_epochs:
            raise ValueError("resumed run epoch is outside approved continuation")
    else:
        if target.exists():
            raise FileExistsError(f"refusing to overwrite SR-0.5 output: {target}")
        target.mkdir(parents=True, exist_ok=False)
        for filename in _FILES:
            shutil.copy2(source / filename, target / filename)
            if _sha256(target / filename) != _sha256(source / filename):
                raise RuntimeError(f"failed to verify cloned artifact: {filename}")
        provenance_path.write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return {"output": str(target), "source": str(source), **record}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "diverse"), required=True)
    parser.add_argument("--initial-epochs", type=int, default=50)
    parser.add_argument("--target-epochs", type=int, default=150)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = prepare(
        args.source_run, args.output, mode=args.mode,
        initial_epochs=args.initial_epochs,
        target_epochs=args.target_epochs, resume=args.resume,
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
