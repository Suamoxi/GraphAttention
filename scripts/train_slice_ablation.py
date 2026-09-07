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

_M8_TARGET = "graph_attention.models.SparseGraphTransformer"
_M9_TARGET = "graph_attention.models.GeometricSparseGraphTransformer"


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

    repo_root = Path(__file__).resolve().parents[1]
    runtime_provenance = collect_runtime_provenance(repo_root)

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
    model, initialization = _instantiate_model(cfg.model, probe, seed=seed)
    model = model.to(device=device, dtype=torch.float32)
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

        validation_standardized, validation_nondimensional = _evaluate(
            model,
            validation_loader,
            standardizers=device_standardizers,
            device=device,
        )
        train_standardized = train_loss_sum / train_samples
        history.append(
            {
                "epoch": epoch,
                "train_standardized_mse": train_standardized,
                "validation_standardized_mse": validation_standardized,
                "validation_nondimensional_mse": validation_nondimensional,
            }
        )
        print(
            f"epoch={epoch:04d} train_standardized_mse={train_standardized:.8e} "
            f"validation_standardized_mse={validation_standardized:.8e} "
            f"validation_nondimensional_mse={validation_nondimensional:.8e}"
        )

        checkpoint = _checkpoint_payload(
            model,
            optimizer,
            epoch=epoch,
            validation_standardized_mse=validation_standardized,
            runtime_provenance=runtime_provenance,
            initialization=initialization,
        )
        torch.save(checkpoint, last_path)
        if validation_standardized < best_validation:
            best_validation = validation_standardized
            best_epoch = epoch
            torch.save(checkpoint, best_path)

    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    test_standardized, test_nondimensional = _evaluate(
        model,
        test_loader,
        standardizers=device_standardizers,
        device=device,
    )
    _write_history(output_dir / "history.csv", history)

    group_by_sample = dict(zip(dataset.sample_ids, group_ids, strict=True))
    summary = {
        "run_name": run_name,
        "model": type(model).__name__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initialization": initialization,
        "best_epoch": best_epoch,
        "selection_metric": "validation_standardized_mse",
        "best_validation_standardized_mse": best_validation,
        "test_standardized_mse_at_best_validation": test_standardized,
        "test_nondimensional_mse_at_best_validation": test_nondimensional,
        "num_samples": len(dataset),
        "num_train_samples": len(train_indices),
        "num_validation_samples": len(validation_indices),
        "num_test_samples": len(test_indices),
        "num_groups": len(set(group_ids)),
        "num_train_groups": len({group_by_sample[value] for value in split.train_ids}),
        "num_validation_groups": len(
            {group_by_sample[value] for value in split.validation_ids}
        ),
        "num_test_groups": len({group_by_sample[value] for value in split.test_ids}),
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


def _instantiate_model(
    model_cfg: DictConfig,
    probe: NodeRegressionBatch,
    *,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    target = str(model_cfg.get("_target_", ""))
    if target not in {_M8_TARGET, _M9_TARGET}:
        raise TypeError("slice ablation compares only the frozen M8 and M9 model classes")

    common = {
        "in_channels": probe.inputs.shape[1],
        "out_channels": probe.targets.shape[1],
        "hidden_dim": int(model_cfg.hidden_dim),
        "num_heads": int(model_cfg.num_heads),
        "num_layers": int(model_cfg.num_layers),
        "mlp_ratio": int(model_cfg.mlp_ratio),
        "conditioning_channels": probe.conditioning.shape[1],
    }

    torch.manual_seed(seed)
    reference = SparseGraphTransformer(**common)
    if target == _M8_TARGET:
        return reference, {
            "policy": "shared_m8_reference_initialization",
            "shared_parameter_seed": seed,
            "geometry_parameter_seed": None,
        }

    geometry_seed = seed + 1
    torch.manual_seed(geometry_seed)
    model = GeometricSparseGraphTransformer(
        **common,
        spatial_dim=probe.coords.shape[1],
    )
    incompatible = model.load_state_dict(reference.state_dict(), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected keys while matching M8/M9 initialization: {incompatible.unexpected_keys}"
        )
    expected_geometry_keys = sorted(
        name for name in model.state_dict() if ".geometry_mlp." in name
    )
    if sorted(incompatible.missing_keys) != expected_geometry_keys:
        raise RuntimeError(
            "M8/M9 shared-parameter initialization mismatch: "
            f"missing={sorted(incompatible.missing_keys)}, expected={expected_geometry_keys}"
        )
    return model, {
        "policy": "matched_m8_shared_parameters_plus_m9_geometry",
        "shared_parameter_seed": seed,
        "geometry_parameter_seed": geometry_seed,
        "geometry_parameter_names": expected_geometry_keys,
    }


def _loader(
    dataset: PrecomputedSlicePTDataset,
    indices: list[int],
    *,
    batch_size: int,
    num_workers: int,
    collator: _NodeRegressionCollator,
    shuffle: bool,
    seed: int,
) -> DataLoader[NodeRegressionBatch]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        generator=generator,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader[NodeRegressionBatch],
    *,
    standardizers: Any,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    standardized_loss_sum = 0.0
    nondimensional_loss_sum = 0.0
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
            standardized = sample_reduced_mse(
                predictions,
                prepared.targets,
                prepared.ptr,
                node_weights=prepared.node_weights,
            )
            nondimensional_predictions = standardizers.inverse_targets(
                predictions,
                prepared.target_channels,
            )
            nondimensional = sample_reduced_mse(
                nondimensional_predictions,
                batch.targets,
                batch.ptr,
                node_weights=batch.node_weights,
            )
            standardized_loss_sum += float(standardized.loss_sum.cpu())
            nondimensional_loss_sum += float(nondimensional.loss_sum.cpu())
            total_samples += standardized.sample_count
    if total_samples == 0:
        raise ValueError("evaluation loader contains no samples")
    return standardized_loss_sum / total_samples, nondimensional_loss_sum / total_samples


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
        coords=batch.coords.to(device=device, dtype=dtype),
        inputs=batch.inputs.to(device=device, dtype=dtype),
        targets=batch.targets.to(device=device, dtype=dtype),
        conditioning=batch.conditioning.to(device=device, dtype=dtype),
    )


def _packed_structure_to_device(
    packed: PackedBatch,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PackedBatch:
    node_weights = packed.node_weights
    if node_weights is not None:
        node_weights = node_weights.to(device=device, dtype=dtype)
    return replace(
        packed,
        edge_index=packed.edge_index.to(device=device),
        batch_index=packed.batch_index.to(device=device),
        ptr=packed.ptr.to(device=device),
        node_weights=node_weights,
    )


def _write_dataset_artifacts(
    output_dir: Path,
    *,
    dataset: PrecomputedSlicePTDataset,
    group_ids: tuple[str, ...],
    group_key: str,
    split: Any,
    runtime_provenance: dict[str, Any],
    standardizers: Any,
) -> None:
    group_by_sample = dict(zip(dataset.sample_ids, group_ids, strict=True))
    payload = {
        "group_metadata_key": group_key,
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
    validation_standardized_mse: float,
    runtime_provenance: dict[str, Any],
    initialization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "validation_standardized_mse": validation_standardized_mse,
        "model_class": type(model).__name__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "runtime_provenance": runtime_provenance,
        "initialization": initialization,
    }


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "epoch",
                "train_standardized_mse",
                "validation_standardized_mse",
                "validation_nondimensional_mse",
            ),
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
