"""Explicit migration of the original M49 Full DiT run; caller holds run lock."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
from omegaconf import OmegaConf

LEGACY = "d5af49b464815b01cb6f60d242f1ac5696b3e7cc"
QUALIFIED = "0a9f5b349573c2fa6c2a7ee83be240d0b690e173"
RUN_NAME = "m49_full_dit_fp32_h128_l10_h4_b8_a2_seed42_blocked50"
REASON = "M49 epoch-boundary persistent-worker shuffle RNG restoration"
MISSING_GIT = {"sha": None, "branch": None, "dirty": None}
# Reviewed copies from Drive: original 41-epoch checkpoint/config/manifest.
REVIEWED_SHA256 = {
    "last.pt": "3f520882a1c5c73a7194167af8b551f986cab2d4065c5b10428f0dd5e6a4a538",
    "dataset_split_manifest.json": "42413ebfbd6be08950b8d18f71b66b387c253081bf8bfb7ed44dadae61c250c0",
    "resolved_config.yaml": "5ff15ac2d69e41b087543826c78a43f86ea70e2cf820e224898507e3e3803331",
}


def verify_missing_git_evidence(run, cfg, repo, ckpt, *, accepted):
    if not accepted:
        raise RuntimeError("legacy Git provenance missing; explicitly authorize reviewed migration")
    legacy = Path(str(cfg.data.case_file)).resolve().parents[1]
    if legacy == repo or legacy.name != "GraphAttention_m49_full_blocked":
        raise RuntimeError("unexpected original Full DiT worktree")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=legacy, text=True).strip()
    if git("rev-parse", "HEAD") != LEGACY or git("status", "--porcelain"):
        raise RuntimeError("legacy worktree is not clean and pinned to the expected revision")
    if Path(str(cfg.generative.split_manifest_file)).resolve() != (
            legacy / "configs/splits/m49_hit_les_forced_blocked50.json"):
        raise RuntimeError("saved split does not point at the original worktree")
    source = run if ckpt["runtime_provenance"]["git"] == MISSING_GIT else run/"pre_rng_migration"
    for name, expected in REVIEWED_SHA256.items():
        with (source/name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected:
            raise RuntimeError(f"unreviewed legacy artifact: {name}; SHA256 {digest}")
    return {
        "kind": "reviewed_artifact_hashes_and_current_clean_legacy_worktree",
        "legacy_worktree": str(legacy),
        "legacy_worktree_sha": LEGACY,
        "artifact_sha256": REVIEWED_SHA256.copy(),
        "historical_execution_revision_verified": False,
    }


def validate_migration(manifest, checkpoint, runtime, run_name, *, legacy_evidence=None):
    original = manifest["runtime_provenance"]["git"]
    source = checkpoint["runtime_provenance"]["git"]
    target = runtime["git"]
    if run_name != RUN_NAME or checkpoint["model_class"] != "FullDiTGraphTransformer":
        raise RuntimeError("migration is restricted to the original M49 Full DiT run")
    original_missing = original == MISSING_GIT
    if original_missing:
        if (not isinstance(legacy_evidence, dict)
                or legacy_evidence.get("artifact_sha256") != REVIEWED_SHA256
                or legacy_evidence.get("legacy_worktree_sha") != LEGACY
                or legacy_evidence.get("historical_execution_revision_verified") is not False):
            raise RuntimeError("missing reviewed legacy provenance evidence")
    elif original["sha"] != LEGACY or original["dirty"] is not False:
        raise RuntimeError("unexpected source manifest revision")
    if not target["sha"] or target["dirty"] is not False:
        raise RuntimeError("migration requires a clean, identified target revision")
    source_is_legacy = (
        source == MISSING_GIT if original_missing else
        source["sha"] == LEGACY and source["dirty"] is False
    )
    if source_is_legacy and "revision_migration" not in checkpoint:
        record = {"from_sha": None if original_missing else LEGACY, "to_sha": target["sha"],
                  "completed_epochs_at_migration": int(checkpoint["epoch"]) + 1,
                  "reason": REASON}
        if original_missing:
            record.update(expected_legacy_sha=LEGACY, source_checkpoint_git=source,
                          source_manifest_git=original, legacy_revision_evidence=legacy_evidence)
        return record
    record = checkpoint.get("revision_migration")
    if (source["sha"] == target["sha"] and source["dirty"] is False and isinstance(record, dict)
            and record.get("from_sha") == (None if original_missing else LEGACY)
            and record.get("to_sha") == target["sha"]
            and record.get("reason") == REASON
            and (not original_missing or record.get("legacy_revision_evidence") == legacy_evidence)):
        return record
    raise RuntimeError("checkpoint revision is not the authorized migration source or target")


def verify_code(repo):
    # New orchestration may be added, but not changes to the validated
    # scientific implementation or configuration.
    subprocess.run(["git", "merge-base", "--is-ancestor", QUALIFIED, "HEAD"],
                   cwd=repo, check=True)
    subprocess.run(["git", "diff", "--quiet", QUALIFIED, "HEAD", "--",
                    "src", "cases", "configs/data", "configs/geometry",
                    "configs/model", "configs/task", "configs/optimizer",
                    "configs/generative", "configs/splits", "configs/config.yaml"],
                   cwd=repo, check=True)


def main():
    from scripts.train_generative import run_generative_training
    from graph_attention.utils.provenance import collect_runtime_provenance
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--accept-reviewed-missing-git", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    verify_code(repo)
    cfg = OmegaConf.load(run / "resolved_config.yaml")
    if Path(str(cfg.generative.output_root)).resolve() / str(cfg.run_name) != run:
        raise RuntimeError("saved config points at a different run")
    if not cfg.generative.strict_resume or cfg.generative.evaluate_test:
        raise RuntimeError("require strict resume and no test evaluation")
    manifest = json.loads((run / "dataset_split_manifest.json").read_text())
    ckpt = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    runtime = collect_runtime_provenance(repo)
    for key in ("runtime", "hardware"):
        if manifest["runtime_provenance"][key] != runtime[key] or ckpt["runtime_provenance"][key] != runtime[key]:
            raise RuntimeError(f"migration {key} differs from source")
    evidence = None
    if manifest["runtime_provenance"]["git"] == MISSING_GIT:
        evidence = verify_missing_git_evidence(
            run, cfg, repo, ckpt, accepted=args.accept_reviewed_missing_git,
        )
    record = validate_migration(manifest, ckpt, runtime, run.name, legacy_evidence=evidence)
    completed = int(ckpt["epoch"]) + 1
    if not completed < args.max_epochs <= 1000:
        raise ValueError(f"require {completed} < max_epochs <= 1000")
    if int(ckpt["global_step"]) != completed * 50 or len(ckpt["history"]) != completed:
        raise RuntimeError("checkpoint epoch/update/history inconsistency")
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if int(best["epoch"]) >= completed:
        raise RuntimeError("best.pt is ahead of last.pt")
    backup = run / "pre_rng_migration"
    if "revision_migration" not in ckpt:
        if not backup.exists():
            with tempfile.TemporaryDirectory(prefix=".migration_", dir=run) as temp:
                staged = Path(temp) / "backup"
                staged.mkdir()
                for name in ("last.pt", "best.pt", "history.csv", "resolved_config.yaml",
                             "dataset_split_manifest.json", "standardizers.pt"):
                    shutil.copy2(run / name, staged / name)
                (staged / "migration.json").write_text(json.dumps(record, indent=2) + "\n")
                staged.rename(backup)
        elif json.loads((backup / "migration.json").read_text()) != record:
            raise RuntimeError("existing migration backup describes a different migration")
    elif not backup.is_dir():
        raise RuntimeError("missing pre-migration recovery backup")
    del ckpt, best
    cfg.generative.resume = True
    cfg.generative.max_epochs = args.max_epochs
    print(json.dumps(record, indent=2), flush=True)
    run_generative_training(cfg, m49_resume_migration=True, m49_legacy_evidence=evidence)


if __name__ == "__main__":
    main()
