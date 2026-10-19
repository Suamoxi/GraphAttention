"""M43 Experiment B: one unchanged M37 model across six 65^3/129^3 HIT cases.

Modes:
  inspect = read-only archive/reference inventory, NO GPU/training
  smoke   = strict M3 references, train-only scaler on a SMALL train subset,
            one checkpointed M37 mixed-case AdamW update, validation by case
  train   = full train-only scaler, case-stratified chronological split,
            checkpointed M37 mixed-resolution Flow Matching with resume

This is UNCONDITIONAL Flow Matching. SR conditioning and SGS prediction
are separate subsequent experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from graph_attention.data.multicase_avbp import (
    AVBPMultiCaseHITDataset, DEFAULT_M43_CASES,
)
from graph_attention.data.multicase_split import (
    balanced_case_interleave, temporal_case_split,
)
from graph_attention.data.splits import SplitManifest
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training import fit_train_standardizers
from graph_attention.training.data_pipeline import (
    GraphTaskCollator, task_batch_to_device,
)
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.train_generative import (
    _atomic_torch_save,
    _load_standardizers,
    _validate_generative_standardizers,
)

FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")
M37_RECIPE = {"hidden_dim": 128, "num_heads": 4, "num_layers": 10}


def revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            sha.update(block)
    return sha.hexdigest()


def inspect_inventory(
    dataset: AVBPMultiCaseHITDataset,
    *,
    reference_dir: Path | None,
) -> dict[str, Any]:
    cases = []
    for entry in dataset.inventory:
        record = dict(entry)
        if reference_dir is not None:
            record["reference_dir"] = str(reference_dir)
        cases.append(record)
    return {
        "status": "inspection_complete",
        "root": str(dataset.root),
        "cases": cases,
        "samples": len(dataset),
        "reference_policy": (
            "Explicit M3.3 per-case YAML required for smoke/train, with "
            "physical values audited from source case_info.json and run.params. "
            "Directory labels and HDF5 snapshots are NOT reference-scale sources."
        ),
        "temporal_policy": (
            "Sorted solution filenames approximate chronological ordering. "
            "Their spacing in physical eddy turnover times is not yet audited."
        ),
    }


def _case_ids_by_case(dataset: AVBPMultiCaseHITDataset) -> dict[str, tuple[str, ...]]:
    return {
        case.case_id: tuple(dataset.sample_ids[i] for i in dataset.indices_by_case[case.case_id])
        for case in dataset.cases
    }


def _subsample(ids: tuple[str, ...], maximum: int) -> tuple[str, ...]:
    if maximum < 1:
        raise ValueError("smoke subset maximum must be positive")
    if len(ids) <= maximum:
        return ids
    # Select throughout each temporal interval, without sampling from
    # excluded gaps or validation/test periods.
    return tuple(ids[(i + 1) * len(ids) // (maximum + 1)] for i in range(maximum))


def _smoke_split(
    complete: SplitManifest,
    dataset: AVBPMultiCaseHITDataset,
    *,
    train_per_case: int = 1,
    validation_per_case: int = 1,
) -> SplitManifest:
    sections: dict[str, tuple[str, ...]] = {}
    for name, ids in (
        ("train", complete.train_ids),
        ("validation", complete.validation_ids),
        ("test", complete.test_ids),
    ):
        limit = train_per_case if name == "train" else validation_per_case
        selected = []
        for case in dataset.cases:
            items = tuple(s for s in ids if dataset.case_by_sample_id[s] == case.case_id)
            selected.extend(_subsample(items, limit))
        sections[name] = tuple(selected)
    return SplitManifest(
        train_ids=sections["train"],
        validation_ids=sections["validation"],
        test_ids=sections["test"],
    )


def _indices_by_case(
    sample_ids: tuple[str, ...],
    dataset: AVBPMultiCaseHITDataset,
) -> dict[str, tuple[int, ...]]:
    index = {sample_id: i for i, sample_id in enumerate(dataset.sample_ids)}
    partition = set(sample_ids)
    return {
        case.case_id: tuple(i for i in dataset.indices_by_case[case.case_id]
                            if dataset.sample_ids[i] in partition)
        for case in dataset.cases
    }


def _references_signature(dataset: AVBPMultiCaseHITDataset) -> dict[str, dict[str, Any]]:
    signature = {}
    schemes = set()
    for case in dataset.cases:
        definition = dataset.case_definitions[case.case_id]
        schemes.add(definition.reference_scales.scheme)
        references = definition.reference_scales
        for name in ("rho_ref", "U_ref", "L_ref"):
            scale = references[name]
            if (scale.value <= 0 or scale.inference_available is not True
                or scale.provenance is None or not scale.provenance.strip()):
                raise ValueError(
                    f"{case.case_id}: invalid or unavailable declared {name}"
                )
        signature[case.case_id] = {
            "file": str(definition.source_path),
            "sha256": file_sha256(definition.source_path),
            "scheme": references.scheme,
            "references": {
                name: {
                    "value": references[name].value,
                    "definition": references[name].definition,
                    "units": references[name].units,
                    "provenance": references[name].provenance,
                } for name in ("rho_ref", "U_ref", "L_ref")
            },
        }
    if len(schemes) != 1:
        raise ValueError(
            "M43-B requires one coherent physical reference scheme across all cases"
        )
    return signature


def build_recipe(*, checkpointing: bool = True) -> Any:
    cfg = OmegaConf.load(
        str(Path(__file__).resolve().parents[1] / "configs/model/dinat_dit.yaml")
    )
    for key, value in M37_RECIPE.items():
        cfg[key] = value
    cfg.mlp_ratio = 4
    cfg.condition_embed_dim = 128
    cfg.sparse_attention_backend = "torch_sparse"
    cfg.include_self_attention = False
    cfg.activation_checkpointing = bool(checkpointing)
    return cfg


def _one_batch(
    dataset: AVBPMultiCaseHITDataset,
    index: int,
    *,
    collator: GraphTaskCollator,
    standardizers: Any,
    device: torch.device,
):
    sample = dataset[index]
    host = collator([sample])
    cuda_batch = task_batch_to_device(host, device=device, dtype=torch.float32)
    prepared = standardizers.transform(cuda_batch)
    return prepared


def _forward(model: torch.nn.Module, problem) -> torch.Tensor:
    return model(
        problem.inputs,
        coords=problem.coords,
        edge_index=problem.edge_index,
        batch_index=problem.batch_index,
        conditioning=problem.conditioning,
        attention_edge_indices=problem.attention_edge_indices,
    )


def _append(collector: dict[str, list[float]], category: str, loss: float) -> None:
    collector.setdefault(category, []).append(loss)


def _summarize_losses(collector: Mapping[str, list[float]]) -> dict[str, Any]:
    result = {}
    for name, values in collector.items():
        if not values:
            continue
        result[name] = {
            "mse": sum(values) / len(values), "physical_samples": len(values)
        }
    return result


def evaluate(
    dataset: AVBPMultiCaseHITDataset,
    indices: tuple[int, ...],
    *,
    model: torch.nn.Module,
    task: FlowMatchingTask,
    standardizers: Any,
    collator: GraphTaskCollator,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    by_case: dict[str, list[float]] = {}
    by_resolution: dict[str, list[float]] = {}
    losses = []
    with torch.inference_mode():
        for index in indices:
            prepared = _one_batch(
                dataset, index, collator=collator,
                standardizers=standardizers, device=device,
            )
            problem = task.make_validation_problem(prepared)
            aggregate = task.training_loss(_forward(model, problem), problem)
            if aggregate.sample_count != 1:
                raise RuntimeError("evaluation microbatch must be one physical sample")
            scalar = float(aggregate.mean.detach().cpu())
            if not (0 <= scalar < float("inf")):
                raise FloatingPointError("non-finite validation velocity MSE")
            case = dataset.case_by_sample_id[dataset.sample_ids[index]]
            resolution = str(dataset.resolution_by_sample_id[dataset.sample_ids[index]])
            _append(by_case, case, scalar)
            _append(by_resolution, resolution, scalar)
            losses.append(scalar)
            del prepared, problem, aggregate
    if not losses:
        raise ValueError("evaluation split is empty")
    return {
        "mean": sum(losses) / len(losses),
        "samples": len(losses),
        "by_case": _summarize_losses(by_case),
        "by_resolution": _summarize_losses(by_resolution),
    }


def train_epoch(
    dataset: AVBPMultiCaseHITDataset,
    selected: dict[str, tuple[int, ...]],
    *,
    epoch: int, seed: int, accumulation: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    task: FlowMatchingTask,
    standardizers: Any,
    collator: GraphTaskCollator,
    device: torch.device,
    generator: torch.Generator,
    max_steps: int | None = None,
) -> dict[str, Any]:
    order = balanced_case_interleave(selected, seed=seed, epoch=epoch)
    model.train()
    losses: list[float] = []
    by_case: dict[str, list[float]] = {}
    by_resolution: dict[str, list[float]] = {}
    step_count = 0
    started = time.perf_counter()
    for start in range(0, len(order), accumulation):
        step_indices = order[start:start+accumulation]
        optimizer.zero_grad(set_to_none=True)
        try:
            for index in step_indices:
                prepared = _one_batch(
                    dataset, index, collator=collator,
                    standardizers=standardizers, device=device,
                )
                problem = task.make_training_problem(
                    prepared, generator=generator,
                )
                aggregate = task.training_loss(_forward(model, problem), problem)
                if aggregate.sample_count != 1:
                    raise RuntimeError("microbatch must contain one physical sample")
                scalar = float(aggregate.mean.detach().cpu())
                if not (0 <= scalar < float("inf")):
                    raise FloatingPointError("non-finite Flow Matching velocity MSE")
                # Exact M6 equal-physical-sample objective:
                # sum per-sample gradients then divide by number of samples
                # actually used in the optimizer step (including short last).
                aggregate.loss_sum.backward()
                losses.append(scalar)
                sample_id = dataset.sample_ids[index]
                _append(by_case, dataset.case_by_sample_id[sample_id], scalar)
                _append(
                    by_resolution,
                    str(dataset.resolution_by_sample_id[sample_id]), scalar,
                )
                del aggregate, problem, prepared

            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(1.0 / len(step_indices))
            optimizer.step()
        except Exception:
            optimizer.zero_grad(set_to_none=True)
            raise
        step_count += 1
        elapsed = time.perf_counter() - started
        print(
            f"[M43-B] epoch={epoch} optimizer_step={step_count} "
            f"samples={len(losses)} last_batch_size={len(step_indices)} "
            f"mean_train_velocity_mse={sum(losses)/len(losses):.8e} "
            f"epoch_wall_s={elapsed:.1f}",
            flush=True,
        )
        if max_steps is not None and step_count >= max_steps:
            break
    if not losses:
        raise ValueError("epoch trained on no physical samples")
    return {
        "mean": sum(losses) / len(losses),
        "samples": len(losses),
        "optimizer_steps": step_count,
        "wall_seconds": time.perf_counter() - started,
        "by_case": _summarize_losses(by_case),
        "by_resolution": _summarize_losses(by_resolution),
    }


def _manifest_signature(
    dataset: AVBPMultiCaseHITDataset,
    split: SplitManifest,
    references: dict,
    *,
    mode: str, seed: int, embargo: int, accumulation: int,
) -> dict[str, Any]:
    content = {
        "mode": mode, "revision": revision(),
        "root": str(dataset.root),
        "samples": [
            {
                "id": spec.sample_id,
                "path": str(spec.snapshot_file),
                "snapshot_size": Path(spec.snapshot_file).stat().st_size,
                "snapshot_mtime_ns": Path(spec.snapshot_file).stat().st_mtime_ns,
                "mesh": str(spec.mesh_file),
                "mesh_size": Path(spec.mesh_file).stat().st_size,
                "mesh_mtime_ns": Path(spec.mesh_file).stat().st_mtime_ns,
                "case_id": spec.case_id,
            } for spec in dataset.sample_specs
        ],
        "train_ids": list(split.train_ids),
        "validation_ids": list(split.validation_ids),
        "test_ids": list(split.test_ids),
        "references": references,
        "seed": seed, "temporal_embargo": embargo,
        "accumulation": accumulation, "model": M37_RECIPE,
        "checkpointing": True,
        "physical_nondimensionalization": True,
        "time_embedding_scale": 1000.0,
    }
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "content": content,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode not in {"inspect", "smoke", "train"}:
        raise ValueError("mode must be inspect, smoke or train")
    root = Path(args.root).expanduser()
    reference_dir = (
        None if args.reference_dir is None
        else Path(args.reference_dir).expanduser().resolve()
    )
    dataset = AVBPMultiCaseHITDataset(
        root=root,
        cases=DEFAULT_M43_CASES,
        reference_dir=reference_dir,
        require_references=args.mode != "inspect",
        expected_snapshots_per_case=100,
    )
    destination = Path(args.run_dir).expanduser().resolve()
    if args.mode == "inspect":
        # Archive inspection needs no physical scaling or GPU and is safe
        # before cases have fully audited reference definitions.
        report = inspect_inventory(dataset, reference_dir=reference_dir)
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / "inventory.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return report

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("M43-B pilot is single-process; DDP unsupported")
    if not torch.cuda.is_available():
        raise RuntimeError("M43-B training/smoke requires a CUDA GPU")
    if args.mode == "smoke" and args.resume:
        raise ValueError("smoke mode cannot resume a scientific run")
    if args.mode == "train" and args.max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    if args.accumulation != len(dataset.cases):
        raise ValueError(
            "M43-B equal-case optimizer windows require accumulation=6 "
            "(one sample per case, alternating 65^3/129^3)"
        )
    if args.mode == "train" and args.smoke_steps is not None:
        raise ValueError("smoke_steps must not be passed to scientific train mode")
    refs = _references_signature(dataset)
    split_plan = temporal_case_split(
        _case_ids_by_case(dataset),
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        temporal_embargo=args.temporal_embargo,
    )
    split = (
        _smoke_split(split_plan.manifest, dataset)
        if args.mode == "smoke" else split_plan.manifest
    )
    signature = _manifest_signature(
        dataset, split, refs, mode=args.mode,
        seed=args.seed, embargo=args.temporal_embargo,
        accumulation=args.accumulation,
    )
    if destination.exists() and not args.resume:
        raise FileExistsError(
            f"output directory already exists: {destination}; "
            "choose a new run directory or use --resume for train"
        )
    if args.resume:
        stored = json.loads((destination / "run_signature.json").read_text())
        if stored != signature:
            raise RuntimeError("resume signature changed; refusing mismatched run")
    else:
        destination.mkdir(parents=True, exist_ok=False)
        atomic_json(destination / "run_signature.json", signature)
        atomic_json(destination / "inventory.json", inspect_inventory(
            dataset, reference_dir=reference_dir,
        ))
        atomic_json(destination / "split.json", {
            "train_ids": list(split.train_ids),
            "validation_ids": list(split.validation_ids),
            "test_ids": list(split.test_ids),
            "excluded_ids": list(split_plan.excluded_ids),
            "case_intervals": split_plan.by_case,
            "split_policy": "per-case chronological with index embargo",
            "temporal_embargo": args.temporal_embargo,
            "warning": (
                "File index gap does not establish physical temporal decorrelation; "
                "audit snapshot time versus large-eddy turnover time separately."
            ),
        })
        for case in dataset.cases:
            definition = dataset.case_definitions[case.case_id]
            shutil.copy2(
                definition.source_path,
                destination / ("ref_" + case.case_id.replace("/", "_") + ".yaml"),
            )

    index_by_id = {sid: i for i, sid in enumerate(dataset.sample_ids)}
    train_indices = tuple(index_by_id[sid] for sid in split.train_ids)
    validation_indices = tuple(index_by_id[sid] for sid in split.validation_ids)
    test_indices = tuple(index_by_id[sid] for sid in split.test_ids)
    task = FlowMatchingTask(
        state_fields=FIELDS,
        physical_nondimensionalization=True,
        validation_seed=1234,
        time_embedding_scale=1000.0,
    )
    standardizers_path = destination / "standardizers.pt"
    if args.resume:
        standardizers = _load_standardizers(standardizers_path)
        if tuple(standardizers.train_sample_ids) != split.train_ids:
            raise RuntimeError("saved scaler trained on wrong split")
        if not standardizers.physical_nondimensionalization:
            raise RuntimeError("saved scaler does not implement M3.3 preprocessing")
    else:
        print(f"[M43-B] Fitting train-only sample-balanced scaler "
              f"on {len(train_indices)} snapshots", flush=True)
        started = time.perf_counter()
        standardizers = fit_train_standardizers(
            task, (dataset[i] for i in train_indices),
            dataset.field_catalog, split,
        )
        _validate_generative_standardizers(standardizers)
        _atomic_torch_save({
            "weighting": standardizers.weighting,
            "physical_nondimensionalization": standardizers.physical_nondimensionalization,
            "train_sample_ids": standardizers.train_sample_ids,
            "inputs": {
                "channel_names": standardizers.inputs.channel_names,
                "mean": standardizers.inputs.mean,
                "scale": standardizers.inputs.scale,
            },
            "targets": {
                "channel_names": standardizers.targets.channel_names,
                "mean": standardizers.targets.mean,
                "scale": standardizers.targets.scale,
            },
        }, standardizers_path)
        print(f"[M43-B] Train-only scaling fit in "
              f"{time.perf_counter()-started:.2f} s", flush=True)
    _validate_generative_standardizers(standardizers)
    print("[M43-B] shared dimensionless channel mean: "
          + str(standardizers.inputs.mean.tolist()), flush=True)
    print("[M43-B] shared dimensionless channel std: "
          + str(standardizers.inputs.scale.tolist()), flush=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    collator = GraphTaskCollator(
        task, dataset.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    probe = collator([dataset[validation_indices[0]]])
    model_probe = task.make_validation_problem(standardizers.transform(probe))
    model, initialization = instantiate_controlled_model(
        build_recipe(), model_probe, seed=args.seed,
    )
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    device_standardizers = standardizers.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000)

    best_loss = float("inf")
    start_epoch = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = torch.load(
            destination / "last.pt", map_location="cpu", weights_only=True,
        )
        if checkpoint.get("signature_sha256") != signature["sha256"]:
            raise RuntimeError("checkpoint signature differs from manifest")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        generator.set_state(checkpoint["training_generator_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_velocity_mse"])
        history = list(checkpoint["history"])
        if not (destination / "best.pt").exists():
            raise FileNotFoundError("resume requires saved best.pt")
        print(f"[M43-B] Resuming from epoch {start_epoch}", flush=True)

    by_case = _indices_by_case(split.train_ids, dataset)
    torch.cuda.reset_peak_memory_stats(device)
    num_epochs = 1 if args.mode == "smoke" else args.max_epochs
    for epoch in range(start_epoch, num_epochs):
        print(f"[M43-B] start epoch {epoch}, mode={args.mode}, "
              f"resolution_order={[c.side for c in dataset.cases]}", flush=True)
        started = time.perf_counter()
        train_stats = train_epoch(
            dataset, by_case,
            epoch=epoch, seed=args.seed, accumulation=args.accumulation,
            model=model, optimizer=optimizer, task=task,
            standardizers=device_standardizers,
            collator=collator, device=device, generator=generator,
            max_steps=1 if args.mode == "smoke" else None,
        )
        validation = evaluate(
            dataset, validation_indices, model=model, task=task,
            standardizers=device_standardizers,
            collator=collator, device=device,
        )
        row = {
            "epoch": epoch,
            "train": train_stats,
            "validation": validation,
            "epoch_wall_seconds": time.perf_counter() - started,
        }
        history.append(row)
        atomic_json(destination / "history.json", {"epochs": history})
        with (destination / "history.csv").open("w", newline="") as stream:
            columns = [
                "epoch", "train_velocity_mse", "validation_velocity_mse",
                "train_samples", "optimizer_steps", "epoch_wall_seconds",
                "validation_65_mse", "validation_129_mse",
            ]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for recorded in history:
                groups = recorded["validation"]["by_resolution"]
                writer.writerow({
                    "epoch": recorded["epoch"],
                    "train_velocity_mse": recorded["train"]["mean"],
                    "validation_velocity_mse": recorded["validation"]["mean"],
                    "train_samples": recorded["train"]["samples"],
                    "optimizer_steps": recorded["train"]["optimizer_steps"],
                    "epoch_wall_seconds": recorded["epoch_wall_seconds"],
                    "validation_65_mse": groups.get("65", {}).get("mse"),
                    "validation_129_mse": groups.get("129", {}).get("mse"),
                })
        is_best = validation["mean"] < best_loss
        if is_best:
            best_loss = validation["mean"]
        checkpoint = {
            "epoch": epoch, "model_class": type(model).__name__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_generator_state": generator.get_state(),
            "initialization": initialization,
            "signature_sha256": signature["sha256"],
            "best_validation_velocity_mse": best_loss,
            "history": history,
            "train_only_scaler_file": "standardizers.pt",
            "activation_checkpointing": True,
        }
        _atomic_torch_save(checkpoint, destination / "last.pt")
        if is_best:
            _atomic_torch_save(checkpoint, destination / "best.pt")
        print(
            f"[M43-B] epoch {epoch} train={train_stats['mean']:.6e} "
            f"validation={validation['mean']:.6e} "
            f"per_resolution={validation['by_resolution']} "
            f"elapsed_s={row['epoch_wall_seconds']:.1f}", flush=True,
        )
    # Evaluate test ONLY at best validation selection, never to select model.
    best = torch.load(
        destination / "best.pt", map_location="cpu", weights_only=True,
    )
    model.load_state_dict(best["model_state_dict"])
    test_stats = evaluate(
        dataset, test_indices, model=model, task=task,
        standardizers=device_standardizers, collator=collator,
        device=device,
    )
    summary = {
        "status": "smoke_passed" if args.mode == "smoke" else "train_complete",
        "mode": args.mode,
        "host": socket.gethostname(),
        "code_revision": revision(),
        "signature_sha256": signature["sha256"],
        "num_physical_samples": len(dataset),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "test_samples": len(test_indices),
        "excluded_samples": len(split_plan.excluded_ids),
        "checkpointed_m37": True,
        "optimizer_accumulation": args.accumulation,
        "physical_nondimensionalization": True,
        "scaler": "train_only_sample_balanced_M6",
        "best_epoch": int(best["epoch"]),
        "best_validation_velocity_mse": float(
            best["best_validation_velocity_mse"]
        ),
        "test_at_best_validation": test_stats,
        "per_case_validation_at_best": history[int(best["epoch"])]["validation"]["by_case"],
        "per_resolution_validation_at_best": history[int(best["epoch"])]["validation"]["by_resolution"],
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "history": history,
        "scientific_caveats": [
            "Resolution is confounded with Reynolds number (Re54 vs Re94).",
            "Temporal index gap does not establish physical decorrelation.",
            "AVBP periodic SOLUTBOUND mapping not yet added to graph topology.",
            "One optimizer sees both resolutions; this is not conditional SR.",
            "Smoke mode uses limited training-split statistics and is diagnostic only.",
        ],
    }
    atomic_json(destination / "summary.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "run_dir": str(destination),
        "best_epoch": summary["best_epoch"],
        "best_validation": summary["best_validation_velocity_mse"],
        "test": summary["test_at_best_validation"],
        "gpu_peak_allocated_gib": summary["gpu_peak_allocated_gib"],
    }, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "smoke", "train"),
                        default="inspect")
    parser.add_argument("--root", type=Path,
                        default=Path("/archivenfs/phlusim/HIT_DNS_COMP"))
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--temporal-embargo", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "smoke" and args.smoke_steps not in (None, 1):
        raise ValueError("smoke runs exactly one optimizer step")
    run(args)


if __name__ == "__main__":
    main()
