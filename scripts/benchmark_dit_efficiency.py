"""Controlled CUDA efficiency benchmark for trained Flow-Matching DiT models.

This benchmark compares architecture cost without retraining. It reconstructs the
saved dataset/task/model contract from each run, loads the selected checkpoint,
and measures on the same physical samples:

- one model forward evaluation,
- one training optimizer step (forward + backward + AdamW step),
- a complete Flow-Matching generation with the requested Heun step count.

Dataset loading, graph construction, host-to-device transfer, and
standardization are intentionally outside the timed regions. Peak CUDA memory
includes the model and the already-prepared device batch. Training peak memory
also includes optimizer state, matching the cost of an actual training step.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from graph_attention.tasks import FlowMatchingTask, NodeRegressionBatch
from graph_attention.training import ChannelStandardizer, TaskStandardizers
from graph_attention.training.data_pipeline import (
    GraphTaskCollator,
    dataset_sample_ids,
    make_loader,
    task_batch_to_device,
)
from graph_attention.training.model_factory import instantiate_controlled_model
from graph_attention.utils.provenance import collect_runtime_provenance


_GIB = float(1024**3)


def main() -> None:
    args = _parse_args()
    run_specs = _parse_run_specs(args.run)
    backend_specs = _parse_run_specs(args.attention_backend or [])
    if len(run_specs) < 2:
        raise ValueError("benchmark requires at least two --run LABEL=RUN_DIR entries")
    unknown_backend_labels = sorted(set(backend_specs) - set(run_specs))
    if unknown_backend_labels:
        raise ValueError(
            "--attention-backend labels must also be present in --run: "
            f"{unknown_backend_labels}"
        )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("efficiency benchmark requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("efficiency benchmark must run in one process on one GPU")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output directory already exists: {output_dir}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    run_dirs = {label: Path(value).expanduser().resolve() for label, value in run_specs.items()}
    for label, run_dir in run_dirs.items():
        _validate_run_artifacts(run_dir, args.checkpoint, label)

    contract = _validate_comparison_contract(run_dirs)
    repo_root = Path(__file__).resolve().parents[1]
    provenance = collect_runtime_provenance(repo_root)

    results: list[dict[str, Any]] = []
    for label, run_dir in run_dirs.items():
        print()
        print("=" * 72)
        print(f"BENCHMARK {label}: {run_dir}")
        print("=" * 72)
        result = _benchmark_run(
            label=label,
            run_dir=run_dir,
            checkpoint_name=args.checkpoint,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            forward_warmup=args.forward_warmup,
            forward_steps=args.forward_steps,
            train_warmup=args.train_warmup,
            train_steps=args.train_steps,
            generation_warmup=args.generation_warmup,
            generation_repeats=args.generation_repeats,
            sampling_steps=args.sampling_steps,
            sampling_seed=args.sampling_seed,
            sparse_attention_backend=backend_specs.get(label),
        )
        results.append(result)
        print(json.dumps(result, indent=2))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    payload = {
        "benchmark": "dit_flow_efficiency_v1",
        "timing_scope": (
            "model_compute_only_excludes_data_loading_graph_construction_"
            "host_to_device_transfer_and_standardization"
        ),
        "training_measurement": "forward_backward_optimizer_step",
        "generation_measurement": "full_flow_ode_heun_sampling",
        "device": str(device),
        "runtime_provenance": provenance,
        "comparison_contract": contract,
        "settings": {
            "checkpoint": args.checkpoint,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "forward_warmup": args.forward_warmup,
            "forward_steps": args.forward_steps,
            "train_warmup": args.train_warmup,
            "train_steps": args.train_steps,
            "generation_warmup": args.generation_warmup,
            "generation_repeats": args.generation_repeats,
            "sampling_steps": args.sampling_steps,
            "sampling_solver": "heun",
            "sampling_method": "flow_ode",
            "sampling_seed": args.sampling_seed,
            "attention_backend_overrides": backend_specs,
        },
        "results": results,
        "relative_to_first": _relative_results(results),
    }
    (output_dir / "efficiency_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    _write_results_csv(output_dir / "efficiency_results.csv", results)
    _write_comparison_csv(
        output_dir / "efficiency_comparison.csv",
        results,
    )

    print()
    print("=" * 72)
    print("EFFICIENCY BENCHMARK COMPLETE")
    print("=" * 72)
    print(f"JSON: {output_dir / 'efficiency_summary.json'}")
    print(f"CSV:  {output_dir / 'efficiency_results.csv'}")
    print(f"CMP:  {output_dir / 'efficiency_comparison.csv'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run specification LABEL=RUN_DIR. Repeat once per model.",
    )
    parser.add_argument(
        "--attention-backend",
        action="append",
        default=None,
        help=(
            "Optional LABEL=BACKEND override for a run. "
            "Used to benchmark the same DiNAT checkpoint with scatter or dgl."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--forward-warmup", type=int, default=10)
    parser.add_argument("--forward-steps", type=int, default=50)
    parser.add_argument("--train-warmup", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--generation-warmup", type=int, default=1)
    parser.add_argument("--generation-repeats", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--sampling-seed", type=int, default=5678)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for name in (
        "batch_size",
        "forward_steps",
        "train_steps",
        "generation_repeats",
        "sampling_steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("num_workers", "forward_warmup", "train_warmup", "generation_warmup"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.sampling_seed < 0:
        raise ValueError("--sampling-seed must be non-negative")
    return args


def _parse_run_specs(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"run specification must be LABEL=RUN_DIR, got {value!r}")
        label, raw_path = value.split("=", maxsplit=1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label or not raw_path:
            raise ValueError(f"invalid run specification {value!r}")
        if label in result:
            raise ValueError(f"duplicate run label {label!r}")
        result[label] = raw_path
    return result


def _validate_run_artifacts(run_dir: Path, checkpoint_name: str, label: str) -> None:
    if not run_dir.is_dir():
        raise NotADirectoryError(f"{label} run directory does not exist: {run_dir}")
    for name in (
        "resolved_config.yaml",
        "summary.json",
        "dataset_split_manifest.json",
        "standardizers.pt",
        checkpoint_name,
    ):
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing required artifact: {path}")


def _validate_comparison_contract(run_dirs: dict[str, Path]) -> dict[str, Any]:
    configs = {
        label: OmegaConf.load(run_dir / "resolved_config.yaml")
        for label, run_dir in run_dirs.items()
    }
    manifests = {
        label: json.loads((run_dir / "dataset_split_manifest.json").read_text())
        for label, run_dir in run_dirs.items()
    }
    standardizers = {
        label: _load_standardizers(run_dir / "standardizers.pt")
        for label, run_dir in run_dirs.items()
    }

    labels = list(run_dirs)
    baseline = labels[0]
    baseline_manifest = manifests[baseline]
    for label in labels[1:]:
        for key in ("train_ids", "validation_ids", "test_ids"):
            if manifests[label].get(key) != baseline_manifest.get(key):
                raise ValueError(
                    f"{label} and {baseline} do not use the same dataset split for {key}"
                )

    shared_config_paths = (
        "seed",
        "data._target_",
        "task._target_",
        "task.time_embedding_scale",
        "model.hidden_dim",
        "model.num_heads",
        "model.num_layers",
        "model.mlp_ratio",
        "model.condition_embed_dim",
        "model.dropout",
        "model.qkv_bias",
        "model.out_proj_bias",
        "generative.batch_size",
    )
    shared_values: dict[str, Any] = {}
    for path in shared_config_paths:
        value = _resolved_select(configs[baseline], path)
        for label in labels[1:]:
            other = _resolved_select(configs[label], path)
            if other != value:
                raise ValueError(
                    f"comparison contract mismatch at {path}: "
                    f"{baseline}={value!r}, {label}={other!r}"
                )
        shared_values[path] = value

    baseline_standardizers = standardizers[baseline]
    for label in labels[1:]:
        current = standardizers[label]
        if current.inputs.channel_names != baseline_standardizers.inputs.channel_names:
            raise ValueError(f"{label} uses different input standardizer channels")
        if current.targets.channel_names != baseline_standardizers.targets.channel_names:
            raise ValueError(f"{label} uses different target standardizer channels")
        if not torch.equal(current.inputs.mean, baseline_standardizers.inputs.mean):
            raise ValueError(f"{label} uses different input standardizer means")
        if not torch.equal(current.inputs.scale, baseline_standardizers.inputs.scale):
            raise ValueError(f"{label} uses different input standardizer scales")

    return {
        "baseline_label": baseline,
        "labels": labels,
        "same_train_validation_test_split": True,
        "same_input_standardization": True,
        "shared_config": shared_values,
        "model_targets": {
            label: str(configs[label].model._target_) for label in labels
        },
        "geometry_configs": {
            label: OmegaConf.to_container(configs[label].geometry, resolve=True)
            for label in labels
        },
    }


def _resolved_select(cfg: DictConfig, path: str) -> Any:
    value = OmegaConf.select(cfg, path)
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _benchmark_run(
    *,
    label: str,
    run_dir: Path,
    checkpoint_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    forward_warmup: int,
    forward_steps: int,
    train_warmup: int,
    train_steps: int,
    generation_warmup: int,
    generation_repeats: int,
    sampling_steps: int,
    sampling_seed: int,
    sparse_attention_backend: str | None,
) -> dict[str, Any]:
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    if sparse_attention_backend is not None:
        OmegaConf.update(
            cfg,
            "model.sparse_attention_backend",
            sparse_attention_backend,
            merge=False,
            force_add=True,
        )
    manifest = json.loads((run_dir / "dataset_split_manifest.json").read_text())
    standardizers = _load_standardizers(run_dir / "standardizers.pt")
    checkpoint = torch.load(
        run_dir / checkpoint_name,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"invalid checkpoint in {run_dir}")

    dataset = instantiate(cfg.data)
    if not hasattr(dataset, "field_catalog"):
        raise TypeError("configured dataset must expose field_catalog")
    task = instantiate(cfg.task)
    if not isinstance(task, FlowMatchingTask):
        raise TypeError(
            f"efficiency benchmark currently requires FlowMatchingTask, got {type(task).__name__}"
        )

    sample_ids = dataset_sample_ids(dataset)
    index_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    train_indices = _indices_for_ids(
        manifest.get("train_ids"),
        index_by_id,
        batch_size,
        "train_ids",
    )
    test_indices = _indices_for_ids(
        manifest.get("test_ids"),
        index_by_id,
        batch_size,
        "test_ids",
    )

    collator = GraphTaskCollator(task, dataset.field_catalog, cfg.geometry)
    train_loader = make_loader(
        dataset,
        train_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=int(cfg.seed),
    )
    test_loader = make_loader(
        dataset,
        test_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=int(cfg.seed),
    )
    host_train = next(iter(train_loader))
    host_test = next(iter(test_loader))

    probe_scaled = standardizers.transform(host_train)
    probe_problem = task.make_validation_problem(probe_scaled)
    model_probe = task.make_model_probe(probe_problem)
    model, initialization = instantiate_controlled_model(
        cfg.model,
        model_probe,
        seed=int(cfg.seed),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device=device, dtype=torch.float32)
    device_standardizers = standardizers.to(device=device, dtype=torch.float32)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    train_batch = task_batch_to_device(
        host_train,
        device=device,
        dtype=torch.float32,
    )
    train_scaled = device_standardizers.transform(train_batch)
    training_generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + 1000)
    train_problem = task.make_training_problem(
        train_scaled,
        generator=training_generator,
    )
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)

    training = _benchmark_training(
        model,
        optimizer,
        task,
        train_problem,
        device=device,
        warmup=train_warmup,
        steps=train_steps,
    )

    del optimizer
    del train_problem
    del train_scaled
    del train_batch
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    test_batch = task_batch_to_device(
        host_test,
        device=device,
        dtype=torch.float32,
    )
    test_scaled = device_standardizers.transform(test_batch)
    forward_problem = task.make_validation_problem(test_scaled)

    forward = _benchmark_forward(
        model,
        forward_problem,
        device=device,
        warmup=forward_warmup,
        steps=forward_steps,
    )
    generation = _benchmark_generation(
        model,
        task,
        test_scaled,
        device=device,
        warmup=generation_warmup,
        repeats=generation_repeats,
        sampling_steps=sampling_steps,
        sampling_seed=sampling_seed,
    )

    node_counts = (
        test_scaled.ptr[1:] - test_scaled.ptr[:-1]
    ).detach().cpu().tolist()
    local_edges = int(test_scaled.edge_index.shape[1])
    attention_edges = {
        name: int(edge_index.shape[1])
        for name, edge_index in test_scaled.attention_edge_indices.items()
    }

    result = {
        "label": label,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "model": type(model).__name__,
        "model_target": str(cfg.model._target_),
        "model_parameters": parameter_count,
        "initialization": initialization,
        "sparse_attention_backend": getattr(
            model,
            "sparse_attention_backend",
            None,
        ),
        "batch_size": test_scaled.num_graphs,
        "nodes_per_sample": sorted({int(value) for value in node_counts}),
        "nodes_per_batch": int(test_scaled.inputs.shape[0]),
        "local_edges_per_batch": local_edges,
        "attention_edges_per_batch": attention_edges,
        "flow_time_embedding_scale": task.time_embedding_scale,
        "forward": forward,
        "training_step": training,
        "generation": generation,
    }

    del forward_problem
    del test_scaled
    del test_batch
    del model
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return result


def _indices_for_ids(
    raw_ids: Any,
    index_by_id: dict[str, int],
    count: int,
    label: str,
) -> list[int]:
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(f"manifest must contain non-empty {label}")
    chosen = raw_ids[:count]
    missing = [sample_id for sample_id in chosen if sample_id not in index_by_id]
    if missing:
        raise ValueError(f"{label} contains IDs absent from dataset: {missing[:5]}")
    return [index_by_id[sample_id] for sample_id in chosen]


def _benchmark_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    task: FlowMatchingTask,
    problem: NodeRegressionBatch,
    *,
    device: torch.device,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    model.train()
    for _ in range(warmup):
        _training_step(model, optimizer, task, problem)
    torch.cuda.synchronize(device)

    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.synchronize(device)
    baseline = _cuda_memory(device)
    torch.cuda.reset_peak_memory_stats(device)

    timings: list[float] = []
    for _ in range(steps):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        _training_step(model, optimizer, task, problem)
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)

    memory = _cuda_peak_memory(device, baseline)
    summary = _timing_summary(timings, problem.num_graphs)
    summary.update(memory)
    summary["warmup_steps"] = warmup
    summary["timed_steps"] = steps
    return summary


def _training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    task: FlowMatchingTask,
    problem: NodeRegressionBatch,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    prediction = _forward_model(model, problem)
    aggregate = task.training_loss(prediction, problem)
    aggregate.mean.backward()
    optimizer.step()


def _benchmark_forward(
    model: torch.nn.Module,
    problem: NodeRegressionBatch,
    *,
    device: torch.device,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            output = _forward_model(model, problem)
            del output
        torch.cuda.synchronize(device)

        gc.collect()
        torch.cuda.synchronize(device)
        baseline = _cuda_memory(device)
        torch.cuda.reset_peak_memory_stats(device)

        timings: list[float] = []
        for _ in range(steps):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = _forward_model(model, problem)
            torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
            del output

    memory = _cuda_peak_memory(device, baseline)
    summary = _timing_summary(timings, problem.num_graphs)
    summary.update(memory)
    summary["warmup_steps"] = warmup
    summary["timed_steps"] = steps
    return summary


def _benchmark_generation(
    model: torch.nn.Module,
    task: FlowMatchingTask,
    batch: NodeRegressionBatch,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    sampling_steps: int,
    sampling_seed: int,
) -> dict[str, Any]:
    model.eval()
    sampling_keys = tuple(f"efficiency_{index:06d}" for index in range(batch.num_graphs))

    for _ in range(warmup):
        generated = task.sample_standardized(
            model,
            batch,
            steps=sampling_steps,
            method="flow_ode",
            solver="heun",
            sampling_eps=0.0,
            final_denoise=False,
            sampling_seed=sampling_seed,
            sampling_keys=sampling_keys,
        )
        del generated
    torch.cuda.synchronize(device)

    gc.collect()
    torch.cuda.synchronize(device)
    baseline = _cuda_memory(device)
    torch.cuda.reset_peak_memory_stats(device)

    timings: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        generated = task.sample_standardized(
            model,
            batch,
            steps=sampling_steps,
            method="flow_ode",
            solver="heun",
            sampling_eps=0.0,
            final_denoise=False,
            sampling_seed=sampling_seed,
            sampling_keys=sampling_keys,
        )
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
        del generated

    memory = _cuda_peak_memory(device, baseline)
    summary = _timing_summary(timings, batch.num_graphs)
    summary.update(memory)
    summary["warmup_repeats"] = warmup
    summary["timed_repeats"] = repeats
    summary["sampling_steps"] = sampling_steps
    summary["solver"] = "heun"
    summary["method"] = "flow_ode"
    summary["model_evaluations_per_sample"] = 2 * sampling_steps
    summary["seconds_per_sample"] = summary["mean_seconds"] / batch.num_graphs
    return summary


def _forward_model(model: torch.nn.Module, batch: NodeRegressionBatch) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "edge_index": batch.edge_index,
        "coords": batch.coords,
        "batch_index": batch.batch_index,
        "conditioning": batch.conditioning,
    }
    if batch.attention_edge_indices:
        kwargs["attention_edge_indices"] = batch.attention_edge_indices
    return model(batch.inputs, **kwargs)


def _timing_summary(timings: list[float], samples_per_call: int) -> dict[str, Any]:
    if not timings:
        raise ValueError("timing list must not be empty")
    mean_seconds = statistics.fmean(timings)
    return {
        "mean_seconds": mean_seconds,
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "mean_ms": 1000.0 * mean_seconds,
        "median_ms": 1000.0 * statistics.median(timings),
        "samples_per_second": samples_per_call / mean_seconds,
        "raw_seconds": timings,
    }


def _cuda_memory(device: torch.device) -> dict[str, float | int]:
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "allocated_gib": allocated / _GIB,
        "reserved_gib": reserved / _GIB,
    }


def _cuda_peak_memory(
    device: torch.device,
    baseline: dict[str, float | int],
) -> dict[str, float | int]:
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    baseline_allocated = int(baseline["allocated_bytes"])
    baseline_reserved = int(baseline["reserved_bytes"])
    return {
        "baseline_allocated_gib": baseline_allocated / _GIB,
        "baseline_reserved_gib": baseline_reserved / _GIB,
        "peak_allocated_gib": peak_allocated / _GIB,
        "peak_reserved_gib": peak_reserved / _GIB,
        "incremental_peak_allocated_gib": (peak_allocated - baseline_allocated) / _GIB,
        "incremental_peak_reserved_gib": (peak_reserved - baseline_reserved) / _GIB,
    }


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _load_standardizers(path: Path) -> TaskStandardizers:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("standardizers.pt must contain a mapping")
    try:
        inputs = payload["inputs"]
        targets = payload["targets"]
        return TaskStandardizers(
            inputs=ChannelStandardizer(
                channel_names=tuple(inputs["channel_names"]),
                mean=inputs["mean"],
                scale=inputs["scale"],
            ),
            targets=ChannelStandardizer(
                channel_names=tuple(targets["channel_names"]),
                mean=targets["mean"],
                scale=targets["scale"],
            ),
            train_sample_ids=tuple(payload["train_sample_ids"]),
            physical_nondimensionalization=bool(
                payload["physical_nondimensionalization"]
            ),
            weighting=str(payload["weighting"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("standardizers.pt does not match the expected schema") from exc


def _relative_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = results[0]
    relative: dict[str, Any] = {}
    metric_paths = {
        "forward_mean_time": ("forward", "mean_seconds"),
        "forward_peak_memory": ("forward", "peak_allocated_gib"),
        "training_step_mean_time": ("training_step", "mean_seconds"),
        "training_peak_memory": ("training_step", "peak_allocated_gib"),
        "generation_time_per_sample": ("generation", "seconds_per_sample"),
        "generation_peak_memory": ("generation", "peak_allocated_gib"),
    }
    for result in results[1:]:
        ratios: dict[str, float] = {}
        for name, path in metric_paths.items():
            baseline_value = _nested_float(baseline, path)
            current_value = _nested_float(result, path)
            ratios[name + "_ratio"] = current_value / baseline_value
            ratios[name + "_change_percent"] = (
                current_value / baseline_value - 1.0
            ) * 100.0
        relative[result["label"]] = {
            "baseline": baseline["label"],
            **ratios,
        }
    return relative


def _nested_float(payload: dict[str, Any], path: tuple[str, str]) -> float:
    return float(payload[path[0]][path[1]])


def _write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        nodes_per_sample = result["nodes_per_sample"]
        rows.append(
            {
                "label": result["label"],
                "run_name": result["run_name"],
                "model": result["model"],
                "model_parameters": result["model_parameters"],
                "sparse_attention_backend": result["sparse_attention_backend"],
                "batch_size": result["batch_size"],
                "nodes_per_sample": (
                    nodes_per_sample[0] if len(nodes_per_sample) == 1 else str(nodes_per_sample)
                ),
                "nodes_per_batch": result["nodes_per_batch"],
                "local_edges_per_batch": result["local_edges_per_batch"],
                "dilated_edges_per_batch": result["attention_edges_per_batch"].get(
                    "dilated", 0
                ),
                "forward_mean_ms": result["forward"]["mean_ms"],
                "forward_samples_per_second": result["forward"]["samples_per_second"],
                "forward_peak_allocated_gib": result["forward"]["peak_allocated_gib"],
                "training_step_mean_ms": result["training_step"]["mean_ms"],
                "training_samples_per_second": result["training_step"][
                    "samples_per_second"
                ],
                "training_peak_allocated_gib": result["training_step"][
                    "peak_allocated_gib"
                ],
                "generation_mean_seconds": result["generation"]["mean_seconds"],
                "generation_seconds_per_sample": result["generation"][
                    "seconds_per_sample"
                ],
                "generation_samples_per_second": result["generation"][
                    "samples_per_second"
                ],
                "generation_peak_allocated_gib": result["generation"][
                    "peak_allocated_gib"
                ],
                "generation_model_evaluations_per_sample": result["generation"][
                    "model_evaluations_per_sample"
                ],
            }
        )
    _write_csv(path, rows)


def _write_comparison_csv(path: Path, results: list[dict[str, Any]]) -> None:
    baseline = results[0]
    metrics: dict[str, tuple[str, str]] = {
        "forward_mean_seconds": ("forward", "mean_seconds"),
        "forward_peak_allocated_gib": ("forward", "peak_allocated_gib"),
        "training_step_mean_seconds": ("training_step", "mean_seconds"),
        "training_peak_allocated_gib": ("training_step", "peak_allocated_gib"),
        "generation_seconds_per_sample": ("generation", "seconds_per_sample"),
        "generation_peak_allocated_gib": ("generation", "peak_allocated_gib"),
    }
    rows: list[dict[str, Any]] = []
    for contender in results[1:]:
        for metric, value_path in metrics.items():
            baseline_value = _nested_float(baseline, value_path)
            contender_value = _nested_float(contender, value_path)
            ratio = contender_value / baseline_value
            rows.append(
                {
                    "metric": metric,
                    "baseline_label": baseline["label"],
                    "baseline_value": baseline_value,
                    "contender_label": contender["label"],
                    "contender_value": contender_value,
                    "contender_over_baseline": ratio,
                    "change_percent": (ratio - 1.0) * 100.0,
                }
            )
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
