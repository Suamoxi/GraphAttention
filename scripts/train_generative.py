"""Common single-GPU generative training runner for packed CFD graphs.

The runner owns orchestration only: dataset/split/loading, train-only scaling,
model construction, optimization, checkpoints, and artifacts. Noise corruption,
model-facing problem construction, loss semantics, and task metadata remain task
owned. Existing DDPM and EDM tasks are adapted without changing their numerical
training behavior; future tasks can implement the small generic task interface
used by ``_GenericTaskAdapter``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from graph_attention.data import make_grouped_split_manifest
from graph_attention.objectives import SampleLossAggregate, sample_reduced_mse
from graph_attention.tasks import (
    DiffusionDenoisingTask,
    EDMDenoisingTask,
    EDMProblem,
    NodeRegressionBatch,
    NodeRegressionTask,
)
from graph_attention.training import fit_train_standardizers, train_equal_sample_optimizer_step
from graph_attention.training.data_pipeline import (
    GraphTaskCollator,
    dataset_group_ids,
    dataset_sample_ids,
    make_loader,
    task_batch_to_device,
)
from graph_attention.training.model_factory import instantiate_controlled_model
from graph_attention.utils.provenance import collect_runtime_provenance


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_generative_training(cfg)


def run_generative_training(cfg: DictConfig) -> dict[str, Any]:
    """Train one configured generative task without dataset-specific runner logic."""

    if "generative" not in cfg:
        raise ValueError("add a generative config, e.g. +generative=hit_slice_diffusion")
    settings = cfg.generative
    seed = _positive_int(cfg.seed, "seed", allow_zero=True)
    max_epochs = _positive_int(settings.max_epochs, "generative.max_epochs")
    batch_size = _positive_int(settings.batch_size, "generative.batch_size")
    num_workers = _positive_int(settings.num_workers, "generative.num_workers", allow_zero=True)

    device = torch.device(str(settings.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA generative training requested but torch.cuda.is_available() is false")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("the common generative runner is intentionally single-process for now")

    run_name = cfg.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be set explicitly for a generative run")
    output_dir = Path(str(settings.output_root)).expanduser().resolve() / run_name
    if output_dir.exists():
        raise FileExistsError(f"generative output directory already exists: {output_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    runtime_provenance = collect_runtime_provenance(repo_root)

    dataset = instantiate(cfg.data)
    if not hasattr(dataset, "field_catalog"):
        raise TypeError("configured dataset must expose a field_catalog")
    task = instantiate(cfg.task)
    if not isinstance(task, NodeRegressionTask):
        raise TypeError("configured generative task must derive from NodeRegressionTask")
    adapter = _task_adapter(task)

    sample_ids = dataset_sample_ids(dataset)
    raw_group_key = settings.get("group_metadata_key", None)
    group_key = None if raw_group_key is None else str(raw_group_key)
    group_ids = dataset_group_ids(dataset, sample_ids, group_key)
    split = make_grouped_split_manifest(
        sample_ids,
        group_ids,
        seed=seed,
        train_ratio=float(settings.train_ratio),
        validation_ratio=float(settings.validation_ratio),
    )
    if not split.validation_ids:
        raise ValueError("configured split produced an empty validation set")
    if not split.test_ids:
        raise ValueError("configured split produced an empty test set")

    index_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    train_indices = [index_by_id[sample_id] for sample_id in split.train_ids]
    validation_indices = [index_by_id[sample_id] for sample_id in split.validation_ids]
    test_indices = [index_by_id[sample_id] for sample_id in split.test_ids]

    standardizers = fit_train_standardizers(
        task,
        (dataset[index] for index in train_indices),
        dataset.field_catalog,
        split,
    )
    _validate_generative_standardizers(standardizers)

    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
    _write_dataset_artifacts(
        output_dir,
        dataset=dataset,
        sample_ids=sample_ids,
        group_ids=group_ids,
        group_key=group_key,
        split=split,
        runtime_provenance=runtime_provenance,
        standardizers=standardizers,
    )

    collator = GraphTaskCollator(task, dataset.field_catalog, cfg.geometry)
    train_loader = make_loader(
        dataset,
        train_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collator=collator,
        shuffle=True,
        seed=seed,
    )
    validation_loader = make_loader(
        dataset,
        validation_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=seed,
    )
    test_loader = make_loader(
        dataset,
        test_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=seed,
    )

    probe_host = next(iter(validation_loader))
    probe_scaled = standardizers.transform(probe_host)
    probe_problem = task.make_validation_problem(probe_scaled)
    model_probe = adapter.model_batch(probe_problem)
    model, initialization = instantiate_controlled_model(cfg.model, model_probe, seed=seed)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    device_standardizers = standardizers.to(device=device, dtype=torch.float32)

    noise_seed = seed + 1000
    training_generator = torch.Generator(device=device).manual_seed(noise_seed)

    metric = adapter.metric_name
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
                batch = task_batch_to_device(host_batch, device=device, dtype=torch.float32)
                scaled = device_standardizers.transform(batch)
                problem = task.make_training_problem(scaled, generator=training_generator)
                step_loss, step_loss_sum, step_samples = adapter.optimizer_step(
                    model,
                    optimizer,
                    problem,
                )

                train_loss_sum += step_loss_sum
                train_samples += step_samples
                tensorboard.add_scalar(f"{metric}/train_step", step_loss, global_step)
                global_step += 1

            validation_loss = _evaluate(
                model,
                validation_loader,
                task=task,
                adapter=adapter,
                standardizers=device_standardizers,
                device=device,
            )
            train_loss = train_loss_sum / train_samples
            history.append(
                {
                    "epoch": epoch,
                    f"train_{metric}": train_loss,
                    f"validation_{metric}": validation_loss,
                }
            )

            tensorboard.add_scalar(f"{metric}/train_epoch", train_loss, epoch)
            tensorboard.add_scalar(f"{metric}/validation_epoch", validation_loss, epoch)
            tensorboard.add_scalar(
                "optimizer/learning_rate",
                float(optimizer.param_groups[0]["lr"]),
                epoch,
            )
            tensorboard.flush()
            print(
                f"epoch={epoch:04d} train_{metric}={train_loss:.8e} "
                f"validation_{metric}={validation_loss:.8e}"
            )

            checkpoint = adapter.checkpoint_payload(
                model,
                optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                runtime_provenance=runtime_provenance,
                initialization=initialization,
                noise_seed=noise_seed,
            )
            torch.save(checkpoint, last_path)
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_epoch = epoch
                torch.save(checkpoint, best_path)

    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    test_loss = _evaluate(
        model,
        test_loader,
        task=task,
        adapter=adapter,
        standardizers=device_standardizers,
        device=device,
    )
    _write_history(output_dir / "history.csv", history, metric)

    group_by_sample = dict(zip(sample_ids, group_ids, strict=True))
    summary = {
        "run_name": run_name,
        **adapter.summary_metadata(),
        "model": type(model).__name__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initialization": initialization,
        "state_channels": list(probe_scaled.input_channels),
        "best_epoch": best_epoch,
        "selection_metric": f"validation_{metric}",
        f"best_validation_{metric}": best_validation,
        f"test_{metric}_at_best_validation": test_loss,
        "tensorboard_log_dir": tensorboard_dir.name,
        "num_samples": len(dataset),
        "num_train_samples": len(train_indices),
        "num_validation_samples": len(validation_indices),
        "num_test_samples": len(test_indices),
        "num_groups": len(set(group_ids)),
        "num_train_groups": len({group_by_sample[value] for value in split.train_ids}),
        "num_validation_groups": len({group_by_sample[value] for value in split.validation_ids}),
        "num_test_groups": len({group_by_sample[value] for value in split.test_ids}),
        **_mesh_summary(dataset, probe_host),
        "split_group_metadata_key": group_key,
        "physical_nondimensionalization": task.physical_nondimensionalization,
        "statistical_scaling": standardizers.weighting,
        adapter.noise_seed_name: noise_seed,
        "validation_seed": task.validation_seed,
        "device": str(device),
        "dtype": "float32",
        "seed": seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def _evaluate(
    model: torch.nn.Module,
    loader: Any,
    *,
    task: NodeRegressionTask,
    adapter: "_TaskAdapter",
    standardizers: Any,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for host_batch in loader:
            batch = task_batch_to_device(host_batch, device=device, dtype=torch.float32)
            scaled = standardizers.transform(batch)
            problem = task.make_validation_problem(scaled)
            aggregate = adapter.loss(model, problem)
            loss_sum += float(aggregate.loss_sum.cpu())
            sample_count += aggregate.sample_count
    if sample_count == 0:
        raise ValueError("generative evaluation loader contains no samples")
    return loss_sum / sample_count


class _TaskAdapter:
    metric_name: str
    noise_seed_name: str

    def __init__(self, task: NodeRegressionTask) -> None:
        self.task = task

    def model_batch(self, problem: Any) -> NodeRegressionBatch:
        raise NotImplementedError

    def loss(self, model: torch.nn.Module, problem: Any) -> SampleLossAggregate:
        raise NotImplementedError

    def optimizer_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        problem: Any,
    ) -> tuple[float, float, int]:
        optimizer.zero_grad(set_to_none=True)
        aggregate = self.loss(model, problem)
        aggregate.mean.backward()
        optimizer.step()
        return (
            float(aggregate.mean.detach().cpu()),
            float(aggregate.loss_sum.detach().cpu()),
            aggregate.sample_count,
        )

    def summary_metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def checkpoint_payload(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        validation_loss: float,
        runtime_provenance: dict[str, Any],
        initialization: dict[str, Any],
        noise_seed: int,
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            f"validation_{self.metric_name}": validation_loss,
            "model_class": type(model).__name__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "runtime_provenance": runtime_provenance,
            "initialization": initialization,
            self.noise_seed_name: noise_seed,
        }


class _DDPMAdapter(_TaskAdapter):
    metric_name = "epsilon_mse"
    noise_seed_name = "diffusion_noise_seed"

    def __init__(self, task: DiffusionDenoisingTask) -> None:
        super().__init__(task)
        self.task = task

    def model_batch(self, problem: Any) -> NodeRegressionBatch:
        if not isinstance(problem, NodeRegressionBatch):
            raise TypeError("DDPM task must return NodeRegressionBatch problems")
        return problem

    def loss(self, model: torch.nn.Module, problem: Any) -> SampleLossAggregate:
        batch = self.model_batch(problem)
        predictions = _forward_model(model, batch)
        return sample_reduced_mse(
            predictions,
            batch.targets,
            batch.ptr,
            node_weights=batch.node_weights,
        )

    def optimizer_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        problem: Any,
    ) -> tuple[float, float, int]:
        batch = self.model_batch(problem)
        result = train_equal_sample_optimizer_step(
            model,
            optimizer,
            [batch],
            local_sample_count=batch.num_graphs,
        )
        step_loss = float(result.objective.cpu())
        return (
            step_loss,
            step_loss * result.local_sample_count,
            result.local_sample_count,
        )

    def summary_metadata(self) -> dict[str, Any]:
        schedule = str(getattr(self.task, "noise_schedule", "cosine"))
        payload: dict[str, Any] = {
            "task": "discrete_ddpm_epsilon_prediction",
            "prediction_type": "epsilon",
            "timesteps": self.task.timesteps,
            "noise_schedule": schedule,
            "time_conditioning": "normalized_discrete_t_over_T",
            "forward_process": "x_t=sqrt(alpha_bar_t)*x0+sqrt(1-alpha_bar_t)*epsilon",
            "generation": "separate_script:scripts.generate_slice_diffusion",
        }
        if schedule == "linear_beta":
            payload["beta_start"] = self.task.beta_start
            payload["beta_end"] = self.task.beta_end
        else:
            payload["cosine_s"] = self.task.cosine_s
        return payload


class _EDMAdapter(_TaskAdapter):
    metric_name = "edm_loss"
    noise_seed_name = "edm_noise_seed"

    def __init__(self, task: EDMDenoisingTask) -> None:
        super().__init__(task)
        self.task = task

    def model_batch(self, problem: Any) -> NodeRegressionBatch:
        if not isinstance(problem, EDMProblem):
            raise TypeError("EDM task must return EDMProblem values")
        return self.task.make_model_probe(problem)

    def loss(self, model: torch.nn.Module, problem: Any) -> SampleLossAggregate:
        if not isinstance(problem, EDMProblem):
            raise TypeError("EDM task must return EDMProblem values")
        predictions = _forward_model(model, problem.model_batch)
        losses = self.task.edm_loss(predictions, problem)
        return SampleLossAggregate(
            per_sample=losses.per_sample,
            loss_sum=losses.loss_sum,
            sample_count=losses.sample_count,
        )

    def optimizer_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        problem: Any,
    ) -> tuple[float, float, int]:
        if not isinstance(problem, EDMProblem):
            raise TypeError("EDM task must return EDMProblem values")

        # Preserve the original M22 operation order exactly: the task returns
        # EDMLoss, backward is called on EDMLoss.mean, and epoch accumulation
        # uses EDMLoss.loss_sum rather than reconstructing it from a Python mean.
        optimizer.zero_grad(set_to_none=True)
        predictions = _forward_model(model, problem.model_batch)
        losses = self.task.edm_loss(predictions, problem)
        losses.mean.backward()
        optimizer.step()
        return (
            float(losses.mean.detach().cpu()),
            float(losses.loss_sum.detach().cpu()),
            losses.sample_count,
        )

    def summary_metadata(self) -> dict[str, Any]:
        return {
            "task": "edm_preconditioned_continuous_noise_denoising",
            "prediction_type": self.task.prediction_type,
            "training_noise_distribution": "log_sigma ~ Normal(p_mean, p_std^2)",
            "p_mean": self.task.p_mean,
            "p_std": self.task.p_std,
            "sigma_data": self.task.sigma_data,
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
            "generation": "separate_script:scripts.generate_slice_edm",
        }


class _GenericTaskAdapter(_TaskAdapter):
    """Structural adapter for future generative tasks such as continuous VP-SDE."""

    def __init__(self, task: NodeRegressionTask) -> None:
        super().__init__(task)
        self.metric_name = str(task.training_metric_name)
        self.noise_seed_name = str(task.training_seed_name)

    def model_batch(self, problem: Any) -> NodeRegressionBatch:
        batch = self.task.make_model_probe(problem)
        if not isinstance(batch, NodeRegressionBatch):
            raise TypeError("make_model_probe must return NodeRegressionBatch")
        return batch

    def loss(self, model: torch.nn.Module, problem: Any) -> SampleLossAggregate:
        predictions = _forward_model(model, self.model_batch(problem))
        aggregate = self.task.training_loss(predictions, problem)
        if not isinstance(aggregate, SampleLossAggregate):
            raise TypeError("training_loss must return SampleLossAggregate")
        return aggregate

    def summary_metadata(self) -> dict[str, Any]:
        payload = self.task.training_summary_metadata()
        if not isinstance(payload, dict):
            raise TypeError("training_summary_metadata must return dict")
        return payload


def _task_adapter(task: NodeRegressionTask) -> _TaskAdapter:
    if isinstance(task, EDMDenoisingTask):
        return _EDMAdapter(task)
    if isinstance(task, DiffusionDenoisingTask):
        return _DDPMAdapter(task)

    required = (
        "training_metric_name",
        "training_seed_name",
        "make_training_problem",
        "make_validation_problem",
        "make_model_probe",
        "training_loss",
        "training_summary_metadata",
        "validation_seed",
    )
    missing = [name for name in required if not hasattr(task, name)]
    if missing:
        raise TypeError(
            "generative task does not implement the common training interface; "
            f"missing={missing}"
        )
    return _GenericTaskAdapter(task)


def _forward_model(model: torch.nn.Module, batch: NodeRegressionBatch) -> torch.Tensor:
    model_kwargs = {
        "edge_index": batch.edge_index,
        "coords": batch.coords,
        "batch_index": batch.batch_index,
        "conditioning": batch.conditioning,
    }
    if batch.attention_edge_indices:
        model_kwargs["attention_edge_indices"] = batch.attention_edge_indices
    return model(batch.inputs, **model_kwargs)


def _validate_generative_standardizers(standardizers: Any) -> None:
    if standardizers.inputs.channel_names != standardizers.targets.channel_names:
        raise RuntimeError("generative input/target standardizers have different channels")
    if not torch.equal(standardizers.inputs.mean, standardizers.targets.mean):
        raise RuntimeError("generative input/target means must be identical")
    if not torch.equal(standardizers.inputs.scale, standardizers.targets.scale):
        raise RuntimeError("generative input/target scales must be identical")


def _write_dataset_artifacts(
    output_dir: Path,
    *,
    dataset: Any,
    sample_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
    group_key: str | None,
    split: Any,
    runtime_provenance: dict[str, Any],
    standardizers: Any,
) -> None:
    group_by_sample = dict(zip(sample_ids, group_ids, strict=True))
    payload: dict[str, Any] = {
        "group_metadata_key": group_key,
        "sample_ids": list(sample_ids),
        "group_ids": list(group_ids),
        "train_ids": list(split.train_ids),
        "validation_ids": list(split.validation_ids),
        "test_ids": list(split.test_ids),
        "train_groups": sorted({group_by_sample[value] for value in split.train_ids}),
        "validation_groups": sorted({group_by_sample[value] for value in split.validation_ids}),
        "test_groups": sorted({group_by_sample[value] for value in split.test_ids}),
        "runtime_provenance": runtime_provenance,
    }
    if hasattr(dataset, "files"):
        payload["sample_files"] = [str(path) for path in dataset.files]
    if hasattr(dataset, "mesh_file"):
        payload["shared_mesh_file"] = str(dataset.mesh_file)
    if hasattr(dataset, "case_file"):
        payload["case_definition_file"] = str(dataset.case_file)

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


def _mesh_summary(dataset: Any, probe: NodeRegressionBatch) -> dict[str, Any]:
    first_stop = int(probe.ptr[1])
    first_graph_edges = (
        (probe.edge_index[0] < first_stop) & (probe.edge_index[1] < first_stop)
    )
    attention_counts = {
        name: int(
            ((topology[0] < first_stop) & (topology[1] < first_stop)).sum()
        )
        for name, topology in probe.attention_edge_indices.items()
    }
    payload: dict[str, Any] = {
        "directed_edges_per_slice": int(first_graph_edges.sum()),
        "attention_topologies": attention_counts,
    }
    grid_shape = getattr(dataset, "grid_shape_2d", None)
    if grid_shape is not None:
        payload["grid_shape_2d"] = list(grid_shape)
        payload["nodes_per_slice"] = int(grid_shape[0] * grid_shape[1])
    else:
        payload["probe_nodes"] = first_stop
    return payload


def _write_history(
    path: Path,
    rows: list[dict[str, float | int]],
    metric: str,
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("epoch", f"train_{metric}", f"validation_{metric}"),
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
