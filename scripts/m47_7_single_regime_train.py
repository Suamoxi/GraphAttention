"""M47.7B: controlled single-regime training using the unchanged M37 DiNAT-DiT.

Reuse the original M47.1 qualified stationarity train/validation membership for
Re54/Ma0.2. Fit one train-only scaler; optimize the unchanged Flow Matching
task and independent standard Gaussian source. Freeze epoch10, then resume
the *same* optimizer/RNG stream through epoch30. Do not load M47.4 weights.

Use separate jobs for 10 and 30 epochs. Neither test snapshots nor results
of the covariance audit influence training or checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from omegaconf import OmegaConf

from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, HITCaseSpec
from graph_attention.data.splits import SplitManifest
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training import fit_train_standardizers
from graph_attention.training.data_pipeline import GraphTaskCollator
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m47_4_evaluate import validate_provenance
from scripts.m47_5_regime_diagnostics import ANCHOR_CASE
from scripts.m47_generation_metrics import CHANNELS
from scripts.train_generative import (
    _atomic_torch_save, _load_standardizers, _validate_generative_standardizers,
)
from scripts.train_m43_b_multires import (
    FIELDS, build_recipe, evaluate, load_training_checkpoint, train_epoch,
)

PROTOCOL = "m47_7b_single_regime_extended_training_v1"


def _write(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def source_split(frozen_run_dir: Path) -> tuple[SplitManifest, dict]:
    """Never invent new temporal splits or sample IDs for this comparison."""
    if frozen_run_dir.name != "epoch10":
        raise ValueError("source must be immutable M47.4 epoch10 run")
    validate_provenance(frozen_run_dir, "none", 10)
    payload = json.loads((frozen_run_dir / "split.json").read_text())
    splits = {}
    for part in ("train", "validation", "test"):
        splits[part] = tuple(sid for sid in payload[part + "_ids"]
                              if sid.startswith(ANCHOR_CASE + ":"))
    manifest = SplitManifest(**{part + "_ids": splits[part]
                               for part in ("train", "validation", "test")})
    if len(manifest.train_ids) < 10 or len(manifest.validation_ids) < 2:
        raise ValueError("insufficient frozen qualified samples")
    return manifest, payload


def signature(args: argparse.Namespace, split: SplitManifest) -> dict:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if not revision:
        raise RuntimeError("missing source revision")
    recipe = OmegaConf.to_container(build_recipe(), resolve=True)
    ref_file = args.reference_dir / "Re54_Ma0.2.yaml"
    content = {
        "protocol": PROTOCOL,
        "case": ANCHOR_CASE,
        "source_frozen_run": str(args.frozen_run_dir),
        "source_split_sha256": hashlib.sha256(
            (args.frozen_run_dir / "split.json").read_bytes()
        ).hexdigest(),
        "source_signature_sha256": json.loads(
            (args.frozen_run_dir / "run_signature.json").read_text()
        )["sha256"],
        "reference_yaml_sha256": hashlib.sha256(ref_file.read_bytes()).hexdigest(),
        "reference_path": str(ref_file),
        "archive_root": str(args.archive_root),
        "train_ids": list(split.train_ids),
        "validation_ids": list(split.validation_ids),
        "test_ids_reserved": list(split.test_ids),
        "code_revision": revision,
        "recipe": recipe,
        "physical_nondimensionalization": True,
        "source_distribution": "independent_unit_Gaussian_standardized",
        "time_embedding_scale": 1000.0,
        "validation_seed": 1234,
        "model_initialization_seed": args.seed,
        "training_rng_seed": args.seed + 1000,
        "learning_rate": args.lr,
        "weight_decay": 0.0,
        "physical_samples_per_optimizer_step": 1,
    }
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return {"sha256": hashlib.sha256(encoded.encode()).hexdigest(), "content": content}


def _standardizer_payload(scalers) -> dict:
    return {
        "weighting": scalers.weighting,
        "physical_nondimensionalization": scalers.physical_nondimensionalization,
        "train_sample_ids": scalers.train_sample_ids,
        "inputs": {
            "channel_names": scalers.inputs.channel_names,
            "mean": scalers.inputs.mean,
            "scale": scalers.inputs.scale,
        },
        "targets": {
            "channel_names": scalers.targets.channel_names,
            "mean": scalers.targets.mean,
            "scale": scalers.targets.scale,
        },
    }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("M47.7B training requires CUDA")
    if args.max_epochs not in (10, 30):
        raise ValueError("controlled training stages are 10 or 30 epochs")
    if args.seed != 42 or args.lr != 1e-4:
        raise ValueError("M47.7B freezes model seed=42 and learning rate=1e-4")
    # Resuming an interrupted stage10 or stage30 must be possible on Slurm.
    if args.resume != args.output_dir.exists():
        raise ValueError("fresh run needs new output; resume requires existing output")

    split, original_split = source_split(args.frozen_run_dir)
    data = AVBPMultiCaseHITDataset(
        root=args.archive_root, cases=(HITCaseSpec(ANCHOR_CASE, 65),),
        reference_dir=args.reference_dir, require_references=True,
        expected_snapshots_per_case=100,
        selected_sample_ids=split.train_ids + split.validation_ids,
    )
    ids = {sid: i for i, sid in enumerate(data.sample_ids)}
    if set(ids) != set(split.train_ids + split.validation_ids):
        raise ValueError("source and selected dataset IDs differ")
    train_indices = tuple(ids[sid] for sid in split.train_ids)
    val_indices = tuple(ids[sid] for sid in split.validation_ids)
    signature_report = signature(args, split)

    task = FlowMatchingTask(
        state_fields=FIELDS, conditioning_parameters=(),
        physical_nondimensionalization=True,
        validation_seed=1234, time_embedding_scale=1000.0,
    )
    if args.resume:
        stored = json.loads((args.output_dir / "run_signature.json").read_text())
        if stored["sha256"] != signature_report["sha256"] or stored["content"] != signature_report["content"]:
            raise ValueError("refusing resume: training/source/code signature changed")
        scalers = _load_standardizers(args.output_dir / "standardizers.pt")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write(args.output_dir / "run_signature.json", signature_report)
        _write(args.output_dir / "split.json", {
            "case": ANCHOR_CASE, "train_ids": list(split.train_ids),
            "validation_ids": list(split.validation_ids),
            "test_ids_reserved_unread": list(split.test_ids),
            "original_policy": original_split.get("split_policy"),
            "source_run": str(args.frozen_run_dir),
        })
        print(f"[M47.7B] fitting scaler on {len(train_indices)} qualified train fields",
              flush=True)
        scalers = fit_train_standardizers(
            task, (data[i] for i in train_indices),
            data.field_catalog,
            SplitManifest(split.train_ids, split.validation_ids),
        )
        _validate_generative_standardizers(scalers)
        _atomic_torch_save(
            _standardizer_payload(scalers),
            args.output_dir / "standardizers.pt",
        )
    _validate_generative_standardizers(scalers)
    if (
        scalers.weighting != "sample_balanced" or
        not scalers.physical_nondimensionalization or
        tuple(scalers.train_sample_ids) != split.train_ids or
        tuple(name.split(".")[0] for name in scalers.inputs.channel_names) != CHANNELS
    ):
        raise ValueError("single-regime scaler provenance mismatch")

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    collator = GraphTaskCollator(
        task, data.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    probe = task.make_validation_problem(scalers.transform(
        collator([data[val_indices[0]]])
    ))
    model, initialization = instantiate_controlled_model(
        build_recipe(), probe, seed=args.seed,
    )
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scalers = scalers.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    history = []
    best = float("inf")
    start_epoch = 0
    if args.resume:
        saved = load_training_checkpoint(
            args.output_dir / "last.pt",
            signature_sha256=signature_report["sha256"],
            model=model, optimizer=optimizer, generator=generator,
        )
        start_epoch = int(saved["epoch"]) + 1
        history = list(saved["history"])
        best = float(saved["best_validation_velocity_mse"])
        if not 0 < start_epoch < args.max_epochs:
            raise ValueError("resume checkpoint is not an unfinished requested stage")
        if args.max_epochs == 30 and (
            start_epoch < 10 or not (args.output_dir / "epoch10.pt").is_file()
        ):
            raise ValueError("epoch30 stage requires a completed epoch10 snapshot")
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, args.max_epochs):
        started = time.perf_counter()
        train_report = train_epoch(
            data, {ANCHOR_CASE: train_indices},
            epoch=epoch, seed=args.seed, accumulation=1,
            model=model, optimizer=optimizer, task=task,
            standardizers=scalers, collator=collator, device=device,
            generator=generator, tensorboard=None,
        )
        validation = evaluate(
            data, val_indices,
            model=model, task=task, standardizers=scalers,
            collator=collator, device=device,
        )
        if validation["mean"] < best:
            best = validation["mean"]
        row = {
            "epoch": epoch + 1,  # human-readable 1-based epoch count
            "train": train_report, "validation": validation,
            "wall_seconds": time.perf_counter() - started,
        }
        history.append(row)
        _write(args.output_dir / "history.json", {"epochs": history})
        checkpoint = {
            "epoch": epoch, "model_class": type(model).__name__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_generator_state": generator.get_state(),
            "signature_sha256": signature_report["sha256"],
            "best_validation_velocity_mse": best,
            "history": history,
            "initialization": initialization,
            "train_only_scaler_file": "standardizers.pt",
        }
        _atomic_torch_save(checkpoint, args.output_dir / "last.pt")
        if epoch + 1 in (10, 30):
            _atomic_torch_save(checkpoint, args.output_dir / f"epoch{epoch+1:02d}.pt")
        print(
            f"[M47.7B] epoch={epoch+1} train={train_report['mean']:.7e} "
            f"val={validation['mean']:.7e} best={best:.7e}",
            flush=True,
        )

    if len(history) != args.max_epochs:
        raise ValueError("wrong completed epoch count")
    summary = {
        "protocol": PROTOCOL,
        "status": f"epoch{args.max_epochs:02d}_complete",
        "train_run": str(args.output_dir),
        "case": ANCHOR_CASE,
        "data_policy": "original M47.1 stationarity-qualified time split",
        "train_count": len(train_indices),
        "validation_count": len(val_indices),
        "reserved_unread_test_count": len(split.test_ids),
        "model": "unchanged M37 DiNAT-DiT",
        "source": "independent unit Gaussian, straight FM path",
        "reference": "audited M47 case-specific scaling",
        "normalization": "single-regime train-only sample-balanced",
        "optimizer_accumulation": 1,
        "source_multi_regime_checkpoint_not_loaded": True,
        "training_signature_sha256": signature_report["sha256"],
        "original_split_file": str(args.frozen_run_dir / "split.json"),
        "current_revision": signature_report["content"]["code_revision"],
        "last_epoch": args.max_epochs,
        "best_validation_mse": best,
        "last_validation_mse": history[-1]["validation"]["mean"],
        "epoch10_validation_mse": history[9]["validation"]["mean"],
        "gpu_peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
        ),
        "optimizer_steps_total": sum(row["train"]["optimizer_steps"] for row in history),
        "mean_epoch_wall_seconds": sum(row["wall_seconds"] for row in history) / len(history),
        "caveats": [
            "Epoch10 and epoch30 use the same training trajectory, not independent seeds.",
            "The original mixed-regime checkpoint has different training population and scaler.",
            "Samples are from one chronological DNS realization; embargo does not prove independence.",
            "No physical projection or correlated Gaussian prior has been introduced.",
            "The test split is excluded from all training and selection.",
        ],
    }
    _write(args.output_dir / "summary.json", summary)
    print(f"[M47.7B] COMPLETE {summary['status']}", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frozen-run-dir", type=Path, required=True)
    p.add_argument("--archive-root", type=Path, default=Path(
        "/archivenfs/phlusim/HIT_DNS_COMP"
    ))
    p.add_argument("--reference-dir", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references"
    ))
    p.add_argument("--max-epochs", type=int, choices=(10, 30), required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    run(p.parse_args())


if __name__ == "__main__":
    main()
