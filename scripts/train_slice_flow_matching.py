"""Train unconditional flow matching on precomputed HIT 2-D slices."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from scripts.train_slice_ablation import (
    _NodeRegressionCollator,
    _instantiate_model,
    _loader,
    _positive_int,
    _task_batch_to_device,
    _write_dataset_artifacts,
)

from graph_attention.data import PrecomputedSlicePTDataset, make_grouped_split_manifest
from graph_attention.geometry import cartesian_4_neighbor_edge_index
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training import (
    fit_train_standardizers,
    sample_reduced_mse,
    train_equal_sample_optimizer_step,
)
from graph_attention.utils.provenance import collect_runtime_provenance


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_slice_flow_matching(cfg)


def run_slice_flow_matching(cfg: DictConfig) -> dict[str, Any]:
    """Run one single-GPU HIT-slice flow-matching experiment."""

    if "generative" not in cfg:
        raise ValueError("add a generative config, e.g. +generative=hit_slice_flow_matching")
    settings = cfg.generative
    seed = _positive_int(cfg.seed, "seed", allow_zero=True)
    max_epochs = _positive_int(settings.max_epochs, "generative.max_epochs")
    batch_size = _positive_int(settings.batch_size, "generative.batch_size")
    num_workers = _positive_int(
        settings.num_workers,
        "generative.num_workers",
        allow_zero=True,
    )
    sample_steps = _positive_int(settings.sample_steps, "generative.sample_steps")
    sampling_seed = _positive_int(
        settings.sampling_seed,
        "generative.sampling_seed",
        allow_zero=True,
    )
    solver = str(settings.solver)
    if solver not in {"euler", "heun"}:
        raise ValueError("generative.solver must be 'euler' or 'heun'")

    device = torch.device(str(settings.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA flow matching requested but torch.cuda.is_available() is false")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("the first M11 flow-matching runner is intentionally single-process")

    run_name = cfg.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be set explicitly for a flow-matching run")
    output_dir = Path(str(settings.output_root)).expanduser().resolve() / run_name
    if output_dir.exists():
        raise FileExistsError(f"flow-matching output directory already exists: {output_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    runtime_provenance = collect_runtime_provenance(repo_root)

    dataset = instantiate(cfg.data)
    if not isinstance(dataset, PrecomputedSlicePTDataset):
        raise TypeError("M11 HIT flow matching requires data=hit_slice_pt")
    task = instantiate(cfg.task)
    if not isinstance(task, FlowMatchingTask):
        raise TypeError("M11 HIT flow matching requires task=hit_flow_matching")

    group_key = str(settings.group_metadata_key)
    group_ids = tuple(dataset.group_id(index, group_key) for index in range(len(dataset)))
    split = make_grouped_split_manifest(
        dataset.sample_ids,
        group_ids,
        seed=seed,
        train_ratio=float(settings.train_ratio),
        validation_ratio=float(settings.validation_ratio),
    )
    if not split.validation_ids:
        raise ValueError("grouped split produced an empty validation set")
    if not split.test_ids:
        raise ValueError("grouped split produced an empty test set")

    index_by_id = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}
    train_indices = [index_by_id[sample_id] for sample_id in split.train_ids]
    validation_indices = [index_by_id[sample_id] for sample_id in split.validation_ids]
    test_indices = [index_by_id[sample_id] for sample_id in split.test_ids]

    standardizers = fit_train_standardizers(
        task,
        (dataset[index] for index in train_indices),
        dataset.field_catalog,
        split,
    )
    _validate_flow_standardizers(standardizers)

    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
    _write_dataset_artifacts(
        output_dir,
        dataset=dataset,
        group_ids=group_ids,
        group_key=group_key,
        split=split,
        runtime_provenance=runtime_provenance,
        standardizers=standardizers,
    )

    edge_index = cartesian_4_neighbor_edge_index(dataset.grid_shape_2d)
    collator = _NodeRegressionCollator(task, dataset.field_catalog, edge_index)
    train_loader = _loader(
        dataset,
        train_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collator=collator,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _loader(
        dataset,
        validation_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=seed,
    )
    test_loader = _loader(
        dataset,
        test_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=seed,
    )

    probe = next(iter(validation_loader))
    probe_scaled = standardizers.transform(probe)
    probe_flow = task.make_validation_problem(probe_scaled)
    model, initialization = _instantiate_model(cfg.model, probe_flow, seed=seed)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    device_standardizers = standardizers.to(device=device, dtype=torch.float32)
    path_seed = seed + 1000
    path_generator = torch.Generator(device=device).manual_seed(path_seed)

    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_epoch = -1
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for host_batch in train_loader:
            batch = _task_batch_to_device(host_batch, device=device, dtype=torch.float32)
            scaled = device_standardizers.transform(batch)
            flow_batch = task.make_training_problem(scaled, generator=path_generator)
            result = train_equal_sample_optimizer_step(
                model,
                optimizer,
                [flow_batch],
                local_sample_count=flow_batch.num_graphs,
            )
            train_loss_sum += float(result.objective.cpu()) * result.local_sample_count
            train_samples += result.local_sample_count

        validation_flow_mse = _evaluate_flow_matching(
            model,
            validation_loader,
            task=task,
            standardizers=device_standardizers,
            device=device,
        )
        train_flow_mse = train_loss_sum / train_samples
        history.append(
            {
                "epoch": epoch,
                "train_flow_velocity_mse": train_flow_mse,
                "validation_flow_velocity_mse": validation_flow_mse,
            }
        )
        print(
            f"epoch={epoch:04d} train_flow_velocity_mse={train_flow_mse:.8e} "
            f"validation_flow_velocity_mse={validation_flow_mse:.8e}"
        )

        checkpoint = _checkpoint_payload(
            model,
            optimizer,
            epoch=epoch,
            validation_flow_mse=validation_flow_mse,
            runtime_provenance=runtime_provenance,
            initialization=initialization,
            path_seed=path_seed,
        )
        torch.save(checkpoint, last_path)
        if validation_flow_mse < best_validation:
            best_validation = validation_flow_mse
            best_epoch = epoch
            torch.save(checkpoint, best_path)

    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    test_flow_mse = _evaluate_flow_matching(
        model,
        test_loader,
        task=task,
        standardizers=device_standardizers,
        device=device,
    )
    generation_metrics = _generate_test_samples(
        model,
        test_loader,
        task=task,
        standardizers=device_standardizers,
        device=device,
        sample_steps=sample_steps,
        solver=solver,
        sampling_seed=sampling_seed,
        output_path=output_dir / "generated_test.pt",
    )
    _write_history(output_dir / "history.csv", history)

    group_by_sample = dict(zip(dataset.sample_ids, group_ids, strict=True))
    summary = {
        "run_name": run_name,
        "task": "linear_gaussian_flow_matching",
        "model": type(model).__name__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initialization": initialization,
        "state_channels": list(probe_scaled.input_channels),
        "time_conditioning": "raw_scalar_t_in_[0,1]",
        "path": "x_t=(1-t)*x_source+t*x_data",
        "target_velocity": "x_data-x_source",
        "best_epoch": best_epoch,
        "selection_metric": "validation_flow_velocity_mse",
        "best_validation_flow_velocity_mse": best_validation,
        "test_flow_velocity_mse_at_best_validation": test_flow_mse,
        "generation": generation_metrics,
        "num_samples": len(dataset),
        "num_train_samples": len(train_indices),
        "num_validation_samples": len(validation_indices),
        "num_test_samples": len(test_indices),
        "num_groups": len(set(group_ids)),
        "num_train_groups": len({group_by_sample[value] for value in split.train_ids}),
        "num_validation_groups": len({group_by_sample[value] for value in split.validation_ids}),
        "num_test_groups": len({group_by_sample[value] for value in split.test_ids}),
        "grid_shape_2d": list(dataset.grid_shape_2d),
        "nodes_per_slice": dataset.grid_shape_2d[0] * dataset.grid_shape_2d[1],
        "directed_edges_per_slice": int(edge_index.shape[1]),
        "split_group_metadata_key": group_key,
        "physical_nondimensionalization": task.physical_nondimensionalization,
        "statistical_scaling": standardizers.weighting,
        "path_seed": path_seed,
        "validation_seed": task.validation_seed,
        "sampling_seed": sampling_seed,
        "device": str(device),
        "dtype": "float32",
        "seed": seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def _evaluate_flow_matching(
    model: torch.nn.Module,
    loader: Any,
    *,
    task: FlowMatchingTask,
    standardizers: Any,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for host_batch in loader:
            batch = _task_batch_to_device(host_batch, device=device, dtype=torch.float32)
            scaled = standardizers.transform(batch)
            flow_batch = task.make_validation_problem(scaled)
            predictions = model(
                flow_batch.inputs,
                edge_index=flow_batch.edge_index,
                coords=flow_batch.coords,
                batch_index=flow_batch.batch_index,
                conditioning=flow_batch.conditioning,
            )
            aggregate = sample_reduced_mse(
                predictions,
                flow_batch.targets,
                flow_batch.ptr,
                node_weights=flow_batch.node_weights,
            )
            loss_sum += float(aggregate.loss_sum.cpu())
            sample_count += aggregate.sample_count
    if sample_count == 0:
        raise ValueError("flow-matching evaluation loader contains no samples")
    return loss_sum / sample_count


def _generate_test_samples(
    model: torch.nn.Module,
    loader: Any,
    *,
    task: FlowMatchingTask,
    standardizers: Any,
    device: torch.device,
    sample_steps: int,
    solver: str,
    sampling_seed: int,
    output_path: Path,
) -> dict[str, Any]:
    model.eval()
    generated_standardized_parts: list[torch.Tensor] = []
    generated_nondimensional_parts: list[torch.Tensor] = []
    target_nondimensional_parts: list[torch.Tensor] = []
    sample_ids: list[str] = []
    node_counts: list[int] = []
    channel_names: tuple[str, ...] | None = None

    with torch.inference_mode():
        for host_batch in loader:
            batch = _task_batch_to_device(host_batch, device=device, dtype=torch.float32)
            scaled = standardizers.transform(batch)
            generated_standardized = task.sample_standardized(
                model,
                scaled,
                steps=sample_steps,
                solver=solver,
                sampling_seed=sampling_seed,
            )
            generated_nondimensional = standardizers.inputs.inverse(
                generated_standardized,
                scaled.input_channels,
            )
            if channel_names is None:
                channel_names = scaled.input_channels
            elif channel_names != scaled.input_channels:
                raise ValueError("test batches produced inconsistent state-channel semantics")

            generated_standardized_parts.append(generated_standardized.cpu())
            generated_nondimensional_parts.append(generated_nondimensional.cpu())
            target_nondimensional_parts.append(batch.inputs.cpu())
            sample_ids.extend(batch.source.sample_ids)
            node_counts.extend(
                int(value)
                for value in (batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().tolist()
            )

    if channel_names is None:
        raise ValueError("test generation loader contains no samples")

    generated_standardized = torch.cat(generated_standardized_parts, dim=0)
    generated_nondimensional = torch.cat(generated_nondimensional_parts, dim=0)
    target_nondimensional = torch.cat(target_nondimensional_parts, dim=0)
    metrics = _marginal_generation_metrics(
        generated_nondimensional,
        target_nondimensional,
        channel_names,
    )
    torch.save(
        {
            "sample_ids": tuple(sample_ids),
            "node_counts": tuple(node_counts),
            "channel_names": channel_names,
            "generated_standardized": generated_standardized,
            "generated_nondimensional": generated_nondimensional,
            "target_nondimensional": target_nondimensional,
            "sample_steps": sample_steps,
            "solver": solver,
            "sampling_seed": sampling_seed,
            "marginal_metrics": metrics,
        },
        output_path,
    )
    return {
        "sample_steps": sample_steps,
        "solver": solver,
        "sampling_seed": sampling_seed,
        "artifact": output_path.name,
        **metrics,
    }


def _marginal_generation_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
    channel_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    if generated.shape != target.shape:
        raise ValueError("generated and target tensors must have identical shapes")
    if generated.ndim != 2 or generated.shape[1] != len(channel_names):
        raise ValueError("generation metric tensors must match declared channel semantics")

    result: dict[str, dict[str, float]] = {}
    generated64 = generated.to(torch.float64)
    target64 = target.to(torch.float64)
    for index, name in enumerate(channel_names):
        generated_channel = generated64[:, index]
        target_channel = target64[:, index]
        wasserstein_1 = torch.mean(
            torch.abs(
                torch.sort(generated_channel).values - torch.sort(target_channel).values
            )
        )
        result[name] = {
            "generated_mean": float(generated_channel.mean()),
            "generated_std": float(generated_channel.std(unbiased=False)),
            "target_mean": float(target_channel.mean()),
            "target_std": float(target_channel.std(unbiased=False)),
            "marginal_wasserstein_1": float(wasserstein_1),
        }
    return result


def _validate_flow_standardizers(standardizers: Any) -> None:
    if standardizers.inputs.channel_names != standardizers.targets.channel_names:
        raise RuntimeError("flow-matching input/target standardizers have different channels")
    if not torch.equal(standardizers.inputs.mean, standardizers.targets.mean):
        raise RuntimeError("flow-matching input/target means must be identical")
    if not torch.equal(standardizers.inputs.scale, standardizers.targets.scale):
        raise RuntimeError("flow-matching input/target scales must be identical")


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    validation_flow_mse: float,
    runtime_provenance: dict[str, Any],
    initialization: dict[str, Any],
    path_seed: int,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "validation_flow_velocity_mse": validation_flow_mse,
        "model_class": type(model).__name__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "runtime_provenance": runtime_provenance,
        "initialization": initialization,
        "path_seed": path_seed,
    }


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "epoch",
                "train_flow_velocity_mse",
                "validation_flow_velocity_mse",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
