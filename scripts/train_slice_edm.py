"""Train EDM-preconditioned diffusion on precomputed HIT 2-D slices."""

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
    _attention_topologies_from_geometry,
    _instantiate_model,
    _loader,
    _NodeRegressionCollator,
    _positive_int,
    _task_batch_to_device,
    _write_dataset_artifacts,
)
from scripts.train_slice_diffusion import _validate_diffusion_standardizers
from torch.utils.tensorboard import SummaryWriter

from graph_attention.data import PrecomputedSlicePTDataset, make_grouped_split_manifest
from graph_attention.geometry import cartesian_4_neighbor_edge_index
from graph_attention.tasks import EDMDenoisingTask, EDMProblem
from graph_attention.training import fit_train_standardizers
from graph_attention.utils.provenance import collect_runtime_provenance


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_slice_edm(cfg)


def run_slice_edm(cfg: DictConfig) -> dict[str, Any]:
    """Train one single-GPU EDM HIT-slice model with equal-sample loss reduction."""

    if "generative" not in cfg:
        raise ValueError("add a generative config, e.g. +generative=hit_slice_diffusion")
    settings = cfg.generative
    seed = _positive_int(cfg.seed, "seed", allow_zero=True)
    max_epochs = _positive_int(settings.max_epochs, "generative.max_epochs")
    batch_size = _positive_int(settings.batch_size, "generative.batch_size")
    num_workers = _positive_int(
        settings.num_workers,
        "generative.num_workers",
        allow_zero=True,
    )

    device = torch.device(str(settings.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA EDM requested but torch.cuda.is_available() is false")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("the first EDM runner is intentionally single-process")

    run_name = cfg.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be set explicitly for an EDM run")
    output_dir = Path(str(settings.output_root)).expanduser().resolve() / run_name
    if output_dir.exists():
        raise FileExistsError(f"EDM output directory already exists: {output_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    runtime_provenance = collect_runtime_provenance(repo_root)

    dataset = instantiate(cfg.data)
    if not isinstance(dataset, PrecomputedSlicePTDataset):
        raise TypeError("HIT EDM requires data=hit_slice_pt")
    task = instantiate(cfg.task)
    if not isinstance(task, EDMDenoisingTask):
        raise TypeError("HIT EDM requires task=hit_edm_diffusion")

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
    _validate_diffusion_standardizers(standardizers)

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
    attention_edge_indices = _attention_topologies_from_geometry(
        cfg.geometry,
        edge_index=edge_index,
        num_nodes=dataset.grid_shape_2d[0] * dataset.grid_shape_2d[1],
    )
    collator = _NodeRegressionCollator(
        task,
        dataset.field_catalog,
        edge_index,
        attention_edge_indices=attention_edge_indices,
    )
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
    probe_problem = task.make_validation_problem(probe_scaled)
    model_probe = task.make_model_probe(probe_problem)
    model, initialization = _instantiate_model(cfg.model, model_probe, seed=seed)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    device_standardizers = standardizers.to(device=device, dtype=torch.float32)

    edm_noise_seed = seed + 1000
    training_generator = torch.Generator(device=device).manual_seed(edm_noise_seed)

    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_epoch = -1
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    tensorboard_dir = output_dir / "tensorboard"
    global_step = 0

    with SummaryWriter(log_dir=str(tensorboard_dir)) as tensorboard:
        for epoch in range(max_epochs):
            model.train()
            train_loss_sum = 0.0
            train_samples = 0

            for host_batch in train_loader:
                batch = _task_batch_to_device(
                    host_batch,
                    device=device,
                    dtype=torch.float32,
                )
                scaled = device_standardizers.transform(batch)
                problem = task.make_training_problem(
                    scaled,
                    generator=training_generator,
                )

                optimizer.zero_grad(set_to_none=True)
                predictions = _forward_model(model, problem)
                losses = task.edm_loss(predictions, problem)
                losses.mean.backward()
                optimizer.step()

                train_loss_sum += float(losses.loss_sum.detach().cpu())
                train_samples += losses.sample_count
                tensorboard.add_scalar(
                    "edm_loss/train_step",
                    float(losses.mean.detach().cpu()),
                    global_step,
                )
                global_step += 1

            validation_loss = _evaluate_edm(
                model,
                validation_loader,
                task=task,
                standardizers=device_standardizers,
                device=device,
            )
            train_loss = train_loss_sum / train_samples
            history.append(
                {
                    "epoch": epoch,
                    "train_edm_loss": train_loss,
                    "validation_edm_loss": validation_loss,
                }
            )

            tensorboard.add_scalar("edm_loss/train_epoch", train_loss, epoch)
            tensorboard.add_scalar("edm_loss/validation_epoch", validation_loss, epoch)
            tensorboard.add_scalar(
                "optimizer/learning_rate",
                float(optimizer.param_groups[0]["lr"]),
                epoch,
            )
            tensorboard.flush()
            print(
                f"epoch={epoch:04d} train_edm_loss={train_loss:.8e} "
                f"validation_edm_loss={validation_loss:.8e}"
            )

            checkpoint = {
                "epoch": epoch,
                "validation_edm_loss": validation_loss,
                "model_class": type(model).__name__,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "runtime_provenance": runtime_provenance,
                "initialization": initialization,
                "edm_noise_seed": edm_noise_seed,
            }
            torch.save(checkpoint, last_path)
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_epoch = epoch
                torch.save(checkpoint, best_path)

    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    test_loss = _evaluate_edm(
        model,
        test_loader,
        task=task,
        standardizers=device_standardizers,
        device=device,
    )
    _write_history(output_dir / "history.csv", history)

    group_by_sample = dict(zip(dataset.sample_ids, group_ids, strict=True))
    summary = {
        "run_name": run_name,
        "task": "edm_preconditioned_continuous_noise_denoising",
        "model": type(model).__name__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initialization": initialization,
        "state_channels": list(probe_scaled.input_channels),
        "prediction_type": task.prediction_type,
        "training_noise_distribution": "log_sigma ~ Normal(p_mean, p_std^2)",
        "p_mean": task.p_mean,
        "p_std": task.p_std,
        "sigma_data": task.sigma_data,
        "preconditioning": {
            "c_skip": "sigma_data^2/(sigma^2+sigma_data^2)",
            "c_out": "sigma*sigma_data/sqrt(sigma^2+sigma_data^2)",
            "c_in": "1/sqrt(sigma^2+sigma_data^2)",
            "c_noise": "log(sigma)/4",
        },
        "loss": (
            "((sigma^2+sigma_data^2)/(sigma*sigma_data)^2)*"
            "MSE(D(x+noise,sigma),x)"
        ),
        "time_conditioning": "edm_log_sigma_over_4",
        "forward_process": "x_sigma=x0+sigma*epsilon",
        "best_epoch": best_epoch,
        "selection_metric": "validation_edm_loss",
        "best_validation_edm_loss": best_validation,
        "test_edm_loss_at_best_validation": test_loss,
        "generation": "separate_script:scripts.generate_slice_edm",
        "tensorboard_log_dir": tensorboard_dir.name,
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
        "attention_topologies": {
            name: int(topology.shape[1]) for name, topology in attention_edge_indices.items()
        },
        "split_group_metadata_key": group_key,
        "physical_nondimensionalization": task.physical_nondimensionalization,
        "statistical_scaling": standardizers.weighting,
        "edm_noise_seed": edm_noise_seed,
        "validation_seed": task.validation_seed,
        "device": str(device),
        "dtype": "float32",
        "seed": seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def _forward_model(model: torch.nn.Module, problem: EDMProblem) -> torch.Tensor:
    batch = problem.model_batch
    model_kwargs = {
        "edge_index": batch.edge_index,
        "coords": batch.coords,
        "batch_index": batch.batch_index,
        "conditioning": batch.conditioning,
    }
    if batch.attention_edge_indices:
        model_kwargs["attention_edge_indices"] = batch.attention_edge_indices
    return model(batch.inputs, **model_kwargs)


def _evaluate_edm(
    model: torch.nn.Module,
    loader: Any,
    *,
    task: EDMDenoisingTask,
    standardizers: Any,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for host_batch in loader:
            batch = _task_batch_to_device(
                host_batch,
                device=device,
                dtype=torch.float32,
            )
            scaled = standardizers.transform(batch)
            problem = task.make_validation_problem(scaled)
            predictions = _forward_model(model, problem)
            losses = task.edm_loss(predictions, problem)
            loss_sum += float(losses.loss_sum.cpu())
            sample_count += losses.sample_count
    if sample_count == 0:
        raise ValueError("EDM evaluation loader contains no samples")
    return loss_sum / sample_count


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("epoch", "train_edm_loss", "validation_edm_loss"),
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
