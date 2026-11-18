"""M47.4 immutable two-stage continuation preparation and checkpoint snapshots.

Scientific training is executed from the exact M47.1 Git revision. This module
only copies/audits files; it never loads or rewrites an original checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

ORIGINAL_REVISION = "5d472796828a8abeb51747751b777dc8cc837f35"
ORIGINAL_SIGNATURES = {
    "none": "ae0ad99149ed65715ea58dbf928f41d6542814cb719bc84e0424ebde6c391534",
    "mach": "677cf9dfd048ce4d229a3a2535e0f73c233d185a56e2aed19031fd78ac67260d",
    "mach_re": "c407a5cb7fa9526515046320f75389b1deb6e9d51c70e5f1462735cc67968fa6",
}
TRACKED = ("last.pt", "best.pt", "standardizers.pt",
           "run_signature.json", "split.json")
PROVENANCE = "m47_4_source_provenance.json"
_HOST_SCRATCH = Path("/gpfs-calypso/scratch")
_CONTAINER_SCRATCH = Path("/scratch")


def _mount_independent_source_path(path: Path) -> Path:
    """Identify the same Calypso scratch files across host and Singularity mounts."""
    try:
        relative = path.relative_to(_HOST_SCRATCH)
    except ValueError:
        return path
    return _CONTAINER_SCRATCH / relative


def accessible_original_source(provenance: dict[str, Any]) -> Path:
    """Resolve only the known host-to-container scratch mount alias.

    The provenance remains immutable. Never silently search for a source under
    a different root or accept a replacement dataset.
    """
    stored = Path(provenance["source_path"])
    if not stored.is_absolute():
        raise ValueError("M47.4 source provenance path must be absolute")
    if stored.is_dir():
        return stored
    candidate = _mount_independent_source_path(stored)
    if candidate != stored and candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"original M47.1 source unavailable at {stored} "
        f"or its exact Singularity mount alias {candidate}"
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(2**20), b""):
            h.update(data)
    return h.hexdigest()


def source_record(source: Path, mode: str) -> dict[str, Any]:
    if mode not in ORIGINAL_SIGNATURES:
        raise ValueError(f"unsupported M47.4 mode: {mode}")
    signature = read_json(source / "run_signature.json")
    summary = read_json(source / "summary.json")
    split = read_json(source / "split.json")
    content = signature.get("content", {})
    phys = content.get("m47_physical_conditions", {})
    if (signature.get("sha256") != ORIGINAL_SIGNATURES[mode]
        or content.get("revision") != ORIGINAL_REVISION
        or content.get("seed") != 42
        or content.get("mode") != "train"
        or content.get("temporal_embargo") != 2
        or content.get("accumulation") != 6
        or phys.get("condition_mode") != mode):
        raise ValueError("source does not match the frozen M47.1 run signature")
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(serialized.encode()).hexdigest() != signature["sha256"]:
        raise ValueError("source signature checksum invalid")
    if (summary.get("signature_sha256") != signature["sha256"]
        or summary.get("conditioning_mode") != mode
        or summary.get("status") != "train_complete"
        or summary.get("best_epoch") != 1
        or len(summary.get("history", [])) != 2
        or summary["history"][-1]["epoch"] != 1
        or summary.get("m47_training_hyperparameters", {}).get("learning_rate") != 1e-4):
        raise ValueError("source is not the completed two-epoch M47.1 training")
    if any(len(split.get(k, [])) != n for k, n in
           (("train_ids", 240), ("validation_ids", 30), ("test_ids", 42),
            ("excluded_ids", 48))):
        raise ValueError("frozen M47.1 split counts changed")
    for field in ("train_ids", "validation_ids", "test_ids"):
        if split[field] != content[field]:
            raise ValueError("source split and signature differ")
    missing = [name for name in TRACKED if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing source files: {missing}")
    return {
        "protocol": "M47.4/pinned-original-source/v1",
        "mode": mode,
        "original_training_revision": ORIGINAL_REVISION,
        "original_run_signature_sha256": signature["sha256"],
        "source_path": str(source.resolve()),
        "original_hashes": {name: digest(source / name) for name in TRACKED},
    }


def verify_original(source: Path, provenance: dict[str, Any], mode: str) -> None:
    actual = source_record(source, mode)
    recorded_path = Path(provenance["source_path"])
    actual_path = Path(actual["source_path"])
    if (_mount_independent_source_path(actual_path)
            != _mount_independent_source_path(recorded_path)):
        raise ValueError("M47.1 original source path differs from recorded provenance")
    # Preserve the original host path string in the immutable audit record.
    # Only the strictly recognized /gpfs-calypso/scratch -> /scratch alias
    # may differ; all source file hashes and physical signatures must match.
    actual["source_path"] = provenance["source_path"]
    if actual != provenance:
        raise ValueError("M47.1 original source changed after M47.4 staging")


def check_stage_summary(run_dir: Path, mode: str, epoch_count: int,
                        provenance: dict[str, Any]) -> dict[str, Any]:
    summary = read_json(run_dir / "summary.json")
    signature = read_json(run_dir / "run_signature.json")
    split = read_json(run_dir / "split.json")
    if (summary.get("status") != "train_complete"
        or summary.get("conditioning_mode") != mode
        or summary.get("signature_sha256") != provenance["original_run_signature_sha256"]
        or signature.get("sha256") != provenance["original_run_signature_sha256"]
        or len(summary.get("history", [])) != epoch_count
        or summary["history"][-1].get("epoch") != epoch_count - 1
        or not 0 <= int(summary.get("best_epoch", -1)) < epoch_count
        or summary.get("m47_training_hyperparameters", {}).get("learning_rate") != 1e-4
        or tuple(split.get("train_ids", [])) != tuple(signature["content"]["train_ids"])
        or tuple(split.get("validation_ids", [])) != tuple(signature["content"]["validation_ids"])):
        raise ValueError(f"invalid {epoch_count}-epoch training summary or provenance")
    if not (run_dir / "last.pt").is_file() or not (run_dir / "best.pt").is_file():
        raise FileNotFoundError("training checkpoint missing")
    return summary


def _assert_safe_destination(source: Path, destination: Path) -> None:
    src = source.resolve()
    dst = destination.resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError("M47.4 destination must be independent of the M47.1 source")
    if destination.exists():
        raise FileExistsError(destination)


def prepare(source: Path, workspace: Path, mode: str) -> None:
    record = source_record(source, mode)
    dest = workspace / "train_s42"
    _assert_safe_destination(source, dest)
    if workspace.exists():
        raise FileExistsError(f"workspace already exists: {workspace}")
    workspace.mkdir(parents=True)
    try:
        shutil.copytree(source, dest)
        (workspace / PROVENANCE).write_text(json.dumps(record, indent=2) + "\n")
        verify_original(source, record, mode)
        if any(digest(dest / name) != val for name, val in record["original_hashes"].items()):
            raise ValueError("staged files differ from original M47.1 files")
    except Exception:
        shutil.rmtree(workspace)
        raise
    print(f"[M47.4] independently staged {mode} source to {dest}", flush=True)


def gate_stage10(source: Path, workspace: Path, mode: str) -> None:
    provenance = read_json(workspace / PROVENANCE)
    verify_original(source, provenance, mode)
    active = workspace / "train_s42"
    prior = workspace / "epoch05"
    check_stage_summary(active, mode, 5, provenance)
    check_stage_summary(prior, mode, 5, provenance)
    for name in (*TRACKED, "history.json"):
        if digest(active / name) != digest(prior / name):
            raise ValueError(f"stage 5 checkpoint diverged from immutable snapshot: {name}")
    if (workspace / "epoch10").exists():
        raise FileExistsError("stage 10 snapshot already exists")
    print(f"[M47.4] approved exact stage-5 optimizer restart: {mode}", flush=True)


def freeze(source: Path, workspace: Path, mode: str, epochs: int) -> None:
    if epochs not in (5, 10):
        raise ValueError("M47.4 checkpoints must be at 5 or 10 epochs")
    provenance = read_json(workspace / PROVENANCE)
    verify_original(source, provenance, mode)
    current = workspace / "train_s42"
    check_stage_summary(current, mode, epochs, provenance)
    destination = workspace / f"epoch{epochs:02d}"
    if destination.exists():
        raise FileExistsError(destination)
    temporary = workspace / f".epoch{epochs:02d}.{os.getpid()}.tmp"
    try:
        shutil.copytree(current, temporary)
        check_stage_summary(temporary, mode, epochs, provenance)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"[M47.4] immutable epoch {epochs} snapshot: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "gate10", "freeze"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(ORIGINAL_SIGNATURES), required=True)
    parser.add_argument("--epochs", type=int, choices=(5, 10))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source, args.workspace, args.mode)
    elif args.action == "gate10":
        gate_stage10(args.source, args.workspace, args.mode)
    else:
        if args.epochs is None:
            parser.error("--epochs is required for freeze")
        freeze(args.source, args.workspace, args.mode, args.epochs)


if __name__ == "__main__":
    main()
