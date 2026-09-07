"""Train the controlled M8-versus-M9 HIT 2-D slice regression ablation."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from graph_attention.data import (
    PackedBatch,
    PrecomputedSlicePTDataset,
    Sample,
    make_grouped_split_manifest,
)
from graph_attention.geometry import cartesian_4_neighbor_edge_index
from graph_attention.models import GeometricSparseGraphTransformer, SparseGraphTransformer
from graph_attention.tasks import NodeRegressionBatch, NodeRegressionTask
from graph_attention.training import (
    fit_train_standardizers,
    sample_reduced_mse,
    train_equal_sample_optimizer_step,
)
from graph_attention.utils.provenance import collect_runtime_provenance


class _NodeRegressionCollator:
    """Attach frozen Cartesian topology and prepare one packed task batch."""

    def __init__(
        self,
        task: NodeRegressionTask,
        catalog: Any,
        edge_index: torch.Tensor,
    ) -> None:
        self.task = task
        self.catalog = catalog
        self.edge_index = edge_index

    def __call__(self, samples: list[Sample]) -> NodeRegressionBatch:
        with_edges = [
            replace(sample, mesh=replace(sample.mesh, edge_index=self.edge_index))
            for sample in samples
        ]
        return self.task.pack_and_prepare(with_edges, self.catalog)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_slice_ablation(cfg)


def run_slice_ablation(cfg: DictConfig) -> dict[str, Any]:
    """Run one single-GPU deterministic-regression experiment and persist artifacts."""

    if "ablation" not in cfg:
        raise ValueError("add an ablation config, e.g. +ablation=hit_slice")
    settings = cfg.ablation
    seed = _positive_int(cfg.seed, "seed", allow_zero=True)
    max_epochs = _positive_int(settings.max_epochs, "ablation.max_epochs")
    batch_size = _positive_int(settings.batch_size, "ablation.batch_size")
    num_workers = _positive_int(settings.num_workers, "ablation.num_workers", allow_zero=True)
    device = torch.device(str(settings.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA ablation requested but torch.cuda.is_available() is false")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("the first HIT slice ablation runner is intentionally single-process")

    run_name = cfg.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be set explicitly for an ablation run")
    output_dir = Path(str(settings.output_root)).expanduser().resolve() / run_name
    if output_dir.exists():
        raise FileExistsError(f"ablation output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    repo_root = Path(__file__).resolve().parents[1]
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
    runtime_provenance = collect_runtime_provenance(repo_root)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = instantiate(cfg.data)
    if not isinstance(dataset, PrecomputedSlicePTDataset):
        raise TypeError("slice ablation requires data=hit_slice_pt")
    task = instantiate(cfg.task)
    if not isinstance(task, NodeRegressionTask):
        raise TypeError("slice ablation requires a NodeRegressionTask")

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
    _write_dataset_artifacts(
        output_dir,
        dataset=dataset,
        group_ids=group_ids,
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
        pin_memory=bool(settings.pin_memory),
    )
    validation_loader = _loader(
        dataset,
        validation_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=seed,
        pin_memory=bool(settings.pin_memory),
    )
    test_loader = _loader(
        dataset,
        test_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=seed,
        pin_memory=bool(settings.pin_memory),
    )

    probe = next(iter(validation_loader))
    model = _instantiate_model(cfg.model, probe).to(device=device, dtype=torch.float32)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    device_standardizers = standardizers.to(device=device, dtype=torch.float32)

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
            result = train_equal_sample_optimizer_step(
                model,
                optimizer,
                [batch],
                local_sample_count=batch.num_graphs,
                standardizers=device_standardizers,
            )
            train_loss_sum += float(result.objective.cpu()) * result.local_sample_count
            train_samples += result.local_sample_count

        validation_mse = _evaluate(
            model,
            validation_loader,
            standardizers=device_standardizers,
            device=device,
        )
        train_mse = train_loss_sum / train_samples
        history.append(
            {
                "epoch": epoch,
                "train_optimizer_mse": train_mse,
                "validation_mse": validation_mse,
            }
        )
        print(
            f"epoch={epoch:04d} train_optimizer_mse={train_mse:.8e} "
            f"validation_mse={validation_mse:.8e}"
        )

        checkpoint = _checkpoint_payload(
            model,
            optimizer,
            epoch=epoch,
            validation_mse=validation_mse,
            runtime_provenance=runtime_provenance,
        )
        torch.save(checkpoint, last_path)
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_epoch = epoch
            torch.save(checkpoint, best_path)

    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    test_mse = _evaluate(
        model,
        test_loader,
        standardizers=device_standardizers,
        device=device,
    )
    _write_history(output_dir / "history.csv", history)

    summary = {
        "run_name": run_name,
        "model": type(model).__name__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation,
        "test_mse_at_best_validation": test_mse,
        "num_samples": len(dataset),
        "num_train_samples": len(train_indices),
        "num_validation_samples": len(validation_indices),
        "num_test_samples": len(test_indices),
        "num_groups": len(set(group_ids)),
        "grid_shape_2d": list(dataset.grid_shape_2d),
        "nodes_per_slice": dataset.grid_shape_2d[0] * dataset.grid_shape_2d[1],
        "directed_edges_per_slice": int(edge_index.shape[1]),
        "split_group_metadata_key": group_key,
        "physical_nondimensionalization": task.physical_nondimensionalization,
        "periodic_cross_boundary_edges": "not_augmented",
        "device": str(device),
        "dtype": "float32",
        "seed": seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def _instantiate_model(model_cfg: DictConfig, probe: NodeRegressionBatch) -> torch.nn.Module:
    kwargs: dict[str, Any] = {
        "in_channels": probe.inputs.shape[1],
        "out_channels": probe.targets.shape[1],
        "conditioning_channels": probe.conditioning.shape[1],
    }
    if "spatial_dim" in model_cfg:
        kwargs["spatial_dim"] = probe.coords.shape[1]
    model = instantiate(model_cfg, **kwargs)
    if not isinstance(model, (SparseGraphTransformer, GeometricSparseGraphTransformer)):
        raise TypeError(
            "slice ablation compares SparseGraphTransformer and "
            "GeometricSparseGraphTransformer"
        )
    return model


def _loader(
    dataset: PrecomputedSlicePTDataset,
    indices: list[int],
    *,
    batch_size: int,
    num_workers: int,
    collator: _NodeRegressionCollator,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader[NodeRegressionBatch]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader[NodeRegressionBatch],
    *,
    standardizers: Any,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for host_batch in loader:
            batch = _task_batch_to_device(host_batch, device=device, dtype=torch.float32)
            prepared = standardizers.transform(batch)
            predictions = model(
                prepared.inputs,
                edge_index=prepared.edge_index,
                coords=prepared.coords,
                batch_index=prepared.batch_index,
                conditioning=prepared.conditioning,
            )
            aggregate = sample_reduced_mse(
                predictions,
                prepared.targets,
                prepared.ptr,
                node_weights=prepared.node_weights,
            )
            total_loss += float(aggregate.loss_sum.cpu())
            total_samples += aggregate.sample_count
    if total_samples == 0:
        raise ValueError("evaluation loader contains no samples")
    return total_loss / total_samples


def _task_batch_to_device(
    batch: NodeRegressionBatch,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> NodeRegressionBatch:
    source = _packed_structure_to_device(batch.source, device=device, dtype=dtype)
    return replace(
        batch,
        source=source,
        coords=batch.coords.to(device=device, dtype=dtype, non_blocking=True),
        inputs=batch.inputs.to(device=device, dtype=dtype, non_blocking=True),
        targets=batch.targets.to(device=device, dtype=dtype, non_blocking=True),
        conditioning=batch.conditioning.to(device=device, dtype=dtype, non_blocking=True),
    )


def _packed_structure_to_device(
    packed: PackedBatch,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PackedBatch:
    node_weights = packed.node_weights
    if node_weights is not None:
        node_weights = node_weights.to(device=device, dtype=dtype, non_blocking=True)
    return replace(
        packed,
        edge_index=packed.edge_index.to(device=device, non_blocking=True),
        batch_index=packed.batch_index.to(device=device, non_blocking=True),
        ptr=packed.ptr.to(device=device, non_blocking=True),
        node_weights=node_weights,
    )


def _write_dataset_artifacts(
    output_dir: Path,
    *,
    dataset: PrecomputedSlicePTDataset,
    group_ids: tuple[str, ...],
    split: Any,
    runtime_provenance: dict[str, Any],
    standardizers: Any,
) -> None:
    group_by_sample = dict(zip(dataset.sample_ids, group_ids, strict=True))
    payload = {
        "seed_grouping_source": "metadata.source_stem",
        "sample_files": [str(path) for path in dataset.files],
        "shared_mesh_file": str(dataset.mesh_file),
        "case_definition_file": str(dataset.case_file),
        "sample_ids": list(dataset.sample_ids),
        "group_ids": list(group_ids),
        "train_ids": list(split.train_ids),
        "validation_ids": list(split.validation_ids),
        "test_ids": list(split.test_ids),
        "train_groups": sorted({group_by_sample[value] for value in split.train_ids}),
        "validation_groups": sorted({group_by_sample[value] for value in split.validation_ids}),
        "test_groups": sorted({group_by_sample[value] for value in split.test_ids}),
        "runtime_provenance": runtime_provenance,
    }
    (output_dir / "dataset_split_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    torch.save(
        {
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
        },
        output_dir / "standardizers.pt",
    )


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    validation_mse: float,
    runtime_provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "validation_mse": validation_mse,
        "model_class": type(model).__name__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "runtime_provenance": runtime_provenance,
    }


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("epoch", "train_optimizer_mse", "validation_mse"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")
    return value


if __name__ == "__main__":
    main()
