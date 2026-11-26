"""Explicit migration of the original M49 Full DiT run; caller holds run lock."""
from __future__ import annotations

import argparse
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


def validate_migration(manifest, checkpoint, runtime, run_name):
    original = manifest["runtime_provenance"]["git"]
    source = checkpoint["runtime_provenance"]["git"]
    target = runtime["git"]
    if run_name != RUN_NAME or checkpoint["model_class"] != "FullDiTGraphTransformer":
        raise RuntimeError("migration is restricted to the original M49 Full DiT run")
    if original["sha"] != LEGACY or original["dirty"] is not False:
        raise RuntimeError("unexpected source manifest revision")
    if not target["sha"] or target["dirty"] is not False or source["dirty"] is not False:
        raise RuntimeError("migration requires clean source and target revisions")
    if source["sha"] == LEGACY and "revision_migration" not in checkpoint:
        return {"from_sha": LEGACY, "to_sha": target["sha"],
                "completed_epochs_at_migration": int(checkpoint["epoch"]) + 1,
                "reason": REASON}
    record = checkpoint.get("revision_migration")
    if (source["sha"] == target["sha"] and isinstance(record, dict)
            and record.get("from_sha") == LEGACY and record.get("to_sha") == target["sha"]
            and record.get("reason") == REASON):
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
    record = validate_migration(manifest, ckpt, collect_runtime_provenance(repo), run.name)
    completed = int(ckpt["epoch"]) + 1
    if not completed < args.max_epochs <= 1000:
        raise ValueError(f"require {completed} < max_epochs <= 1000")
    if int(ckpt["global_step"]) != completed * 50 or len(ckpt["history"]) != completed:
        raise RuntimeError("checkpoint epoch/update/history inconsistency")
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if int(best["epoch"]) >= completed:
        raise RuntimeError("best.pt is ahead of last.pt")
    backup = run / "pre_rng_migration"
    if ckpt["runtime_provenance"]["git"]["sha"] == LEGACY:
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
    run_generative_training(cfg, m49_resume_migration=True)


if __name__ == "__main__":
    main()
