"""Train Improved DDPM on precomputed HIT 2-D slices."""

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
from graph_attention.tasks import ImprovedDiffusionDenoisingTask
from graph_attention.training import fit_train_standardizers
from graph_attention.utils.provenance import collect_runtime_provenance


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_slice_improved_diffusion(cfg)


def run_slice_improved_diffusion(cfg: DictConfig) -> dict[str, Any]:
    """Train one single-GPU Improved-DDPM HIT-slice model."""

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
        raise RuntimeError(
            "CUDA improved diffusion requested but torch.cuda.is_available() is false"
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("the first Improved-DDPM runner is intentionally single-process")

    run_name = cfg.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be set explicitly for an Improved-DDPM run")
    output_dir = Path(str(settings.output_root)).expanduser().resolve() / run_name
    if output_dir.exists():
        raise FileExistsError(f"Improved-DDPM output directory already exists: {output_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    runtime_provenance = collect_runtime_provenance(repo_root)

    dataset = instantiate(cfg.data)
    if not isinstance(dataset, PrecomputedSlicePTDataset):
        raise TypeError("HIT Improved-DDPM requires data=hit_slice_pt")
    task = instantiate(cfg.task)
    if not isinstance(task, ImprovedDiffusionDenoisingTask):
        raise TypeError("HIT Improved-DDPM requires task=hit_improved_diffusion")

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

    diffusion_noise_seed = seed + 1000
    timestep_sampler_seed = seed + 2000
    training_noise_generator = torch.Generator(device=device).manual_seed(diffusion_noise_seed)
    timestep_generator = torch.Generator(device="cpu").manual_seed(timestep_sampler_seed)
    timestep_sampler = task.make_timestep_sampler()

    history: list[dict[str, float | int | bool]] = []
    best_validation = float("inf")
    best_epoch = -1
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    tensorboard_dir = output_dir / "tensorboard"
    global_step = 0

    with SummaryWriter(log_dir=str(tensorboard_dir)) as tensorboard:
        for epoch in range(max_epochs):
            model.train()
            weighted_objective_sum = 0.0
            sampled_hybrid_sum = 0.0
            sampled_simple_sum = 0.0
            sampled_vlb_sum = 0.0
            train_samples = 0

            for host_batch in train_loader:
                batch = _task_batch_to_device(
                    host_batch,
                    device=device,
                    dtype=torch.float32,
                )
                scaled = device_standardizers.transform(batch)
                timesteps, importance_weights = timestep_sampler.sample(
                    scaled.num_graphs,
                    generator=timestep_generator,
                    device=device,
                )
                problem = task.make_training_problem(
                    scaled,
                    timesteps=timesteps,
                    generator=training_noise_generator,
                )

                optimizer.zero_grad(set_to_none=True)
                predictions = _forward_model(model, problem.model_batch)
                losses = task.hybrid_loss(predictions, problem)
                timestep_sampler.update(timesteps, losses.hybrid_per_sample.detach())
                weighted_objective = (
                    losses.hybrid_per_sample * importance_weights
                ).mean()
                weighted_objective.backward()
                optimizer.step()

                sample_count = losses.sample_count
                weighted_objective_sum += float(weighted_objective.detach().cpu()) * sample_count
                sampled_hybrid_sum += float(losses.hybrid_per_sample.detach().sum().cpu())
                sampled_simple_sum += float(losses.simple_per_sample.detach().sum().cpu())
                sampled_vlb_sum += float(losses.vlb_per_sample.detach().sum().cpu())
                train_samples += sample_count

                tensorboard.add_scalar(
                    "hybrid/train_importance_weighted_step",
                    float(weighted_objective.detach().cpu()),
                    global_step,
                )
                global_step += 1

            validation = _evaluate_improved_diffusion(
                model,
                validation_loader,
                task=task,
                standardizers=device_standardizers,
                device=device,
            )
            train_weighted = weighted_objective_sum / train_samples
            train_sampled_hybrid = sampled_hybrid_sum / train_samples
            train_sampled_simple = sampled_simple_sum / train_samples
            train_sampled_vlb = sampled_vlb_sum / train_samples
            history.append(
                {
                    "epoch": epoch,
                    "train_importance_weighted_hybrid": train_weighted,
                    "train_sampled_hybrid": train_sampled_hybrid,
                    "train_sampled_simple_mse": train_sampled_simple,
                    "train_sampled_vlb": train_sampled_vlb,
                    "validation_hybrid": validation["hybrid"],
                    "validation_simple_mse": validation["simple_mse"],
                    "validation_vlb": validation["vlb"],
                    "importance_sampler_warmed_up": timestep_sampler.warmed_up,
                }
            )

            tensorboard.add_scalar("hybrid/train_importance_weighted_epoch", train_weighted, epoch)
            tensorboard.add_scalar("hybrid/validation_epoch", validation["hybrid"], epoch)
            tensorboard.add_scalar("epsilon_mse/validation_epoch", validation["simple_mse"], epoch)
            tensorboard.add_scalar("vlb/validation_epoch", validation["vlb"], epoch)
            tensorboard.add_scalar(
                "importance_sampler/warmed_up",
                float(timestep_sampler.warmed_up),
                epoch,
            )
            tensorboard.add_scalar(
                "optimizer/learning_rate",
                float(optimizer.param_groups[0]["lr"]),
                epoch,
            )
            tensorboard.flush()

            print(
                f"epoch={epoch:04d} "
                f"train_weighted_hybrid={train_weighted:.8e} "
                f"validation_hybrid={validation['hybrid']:.8e} "
                f"validation_epsilon_mse={validation['simple_mse']:.8e} "
                f"validation_vlb={validation['vlb']:.8e} "
                f"importance_warm={timestep_sampler.warmed_up}"
            )

            checkpoint = _checkpoint_payload(
                model,
                optimizer,
                epoch=epoch,
                validation=validation,
                runtime_provenance=runtime_provenance,
                initialization=initialization,
                diffusion_noise_seed=diffusion_noise_seed,
                timestep_sampler_seed=timestep_sampler_seed,
            )
            torch.save(checkpoint, last_path)
            if validation["hybrid"] < best_validation:
                best_validation = validation["hybrid"]
                best_epoch = epoch
                torch.save(checkpoint, best_path)

    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = _evaluate_improved_diffusion(
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
        "task": "improved_ddpm_epsilon_plus_learned_range_variance",
        "model": type(model).__name__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initialization": initialization,
        "state_channels": list(probe_scaled.input_channels),
        "prediction_type": task.prediction_type,
        "model_output_channels": 2 * probe_scaled.inputs.shape[1],
        "learned_reverse_variance": True,
        "reverse_variance_parameterization": (
            "log_variance=(v+1)/2*log(beta_t)+(1-(v+1)/2)*log(posterior_variance_clipped_t)"
        ),
        "hybrid_loss": "L_simple + vlb_weight * L_vlb",
        "vlb_weight": task.vlb_weight,
        "vlb_epsilon_gradient": "detached",
        "timestep_sampling": "loss_second_moment_importance_sampling",
        "importance_history_per_timestep": task.importance_history_per_timestep,
        "importance_uniform_probability": task.importance_uniform_probability,
        "timesteps": task.timesteps,
        "noise_schedule": "cosine",
        "cosine_s": task.cosine_s,
        "time_conditioning": "normalized_discrete_t_over_T_scalar_unchanged_from_M13",
        "forward_process": "x_t=sqrt(alpha_bar_t)*x0+sqrt(1-alpha_bar_t)*epsilon",
        "best_epoch": best_epoch,
        "selection_metric": "validation_hybrid_loss",
        "best_validation_hybrid_loss": best_validation,
        "test_hybrid_loss_at_best_validation": test_metrics["hybrid"],
        "test_epsilon_mse_at_best_validation": test_metrics["simple_mse"],
        "test_vlb_at_best_validation": test_metrics["vlb"],
        "generation": "separate_script:scripts.generate_slice_improved_diffusion",
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
        "diffusion_noise_seed": diffusion_noise_seed,
        "timestep_sampler_seed": timestep_sampler_seed,
        "validation_seed": task.validation_seed,
        "device": str(device),
        "dtype": "float32",
        "seed": seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def _forward_model(model: torch.nn.Module, batch: Any) -> torch.Tensor:
    model_kwargs = {
        "edge_index": batch.edge_index,
        "coords": batch.coords,
        "batch_index": batch.batch_index,
        "conditioning": batch.conditioning,
    }
    if batch.attention_edge_indices:
        model_kwargs["attention_edge_indices"] = batch.attention_edge_indices
    return model(batch.inputs, **model_kwargs)


def _evaluate_improved_diffusion(
    model: torch.nn.Module,
    loader: Any,
    *,
    task: ImprovedDiffusionDenoisingTask,
    standardizers: Any,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    hybrid_sum = 0.0
    simple_sum = 0.0
    vlb_sum = 0.0
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
            predictions = _forward_model(model, problem.model_batch)
            losses = task.hybrid_loss(predictions, problem)
            hybrid_sum += float(losses.hybrid_per_sample.sum().cpu())
            simple_sum += float(losses.simple_per_sample.sum().cpu())
            vlb_sum += float(losses.vlb_per_sample.sum().cpu())
            sample_count += losses.sample_count
    if sample_count == 0:
        raise ValueError("Improved-DDPM evaluation loader contains no samples")
    return {
        "hybrid": hybrid_sum / sample_count,
        "simple_mse": simple_sum / sample_count,
        "vlb": vlb_sum / sample_count,
    }


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    validation: dict[str, float],
    runtime_provenance: dict[str, Any],
    initialization: dict[str, Any],
    diffusion_noise_seed: int,
    timestep_sampler_seed: int,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "validation_hybrid_loss": validation["hybrid"],
        "validation_epsilon_mse": validation["simple_mse"],
        "validation_vlb": validation["vlb"],
        "model_class": type(model).__name__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "runtime_provenance": runtime_provenance,
        "initialization": initialization,
        "diffusion_noise_seed": diffusion_noise_seed,
        "timestep_sampler_seed": timestep_sampler_seed,
    }


def _write_history(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
