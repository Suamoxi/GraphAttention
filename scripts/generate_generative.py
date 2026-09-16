"""Generic single-GPU generation runner for trained graph generative tasks.

The runner owns source-artifact loading, dataset/test-split reconstruction,
standardization, model restoration, population packing, and generation artifacts.
Sampler dynamics remain task-owned through ``sample_standardized`` and
``sampler_name``. This keeps VP-SDE generation out of task-specific launcher
scripts while preserving the existing benchmark artifact schema.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from graph_attention.tasks import NodeRegressionBatch, NodeRegressionTask
from graph_attention.training import ChannelStandardizer, TaskStandardizers
from graph_attention.training.data_pipeline import (
    GraphTaskCollator,
    dataset_sample_ids,
    make_loader,
    task_batch_to_device,
)
from graph_attention.training.model_factory import instantiate_controlled_model


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="generate_generative",
)
def main(cfg: DictConfig) -> None:
    run_generative_generation(cfg)


def run_generative_generation(cfg: DictConfig) -> dict[str, Any]:
    """Generate one unpaired test-sized population from a trained generative run."""

    run_dir = _existing_directory(cfg.run_dir, "run_dir")
    source_config_path = run_dir / "resolved_config.yaml"
    source_summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "dataset_split_manifest.json"
    standardizers_path = run_dir / "standardizers.pt"
    checkpoint_path = run_dir / str(cfg.checkpoint)
    for path, label in (
        (source_config_path, "resolved_config.yaml"),
        (source_summary_path, "summary.json"),
        (manifest_path, "dataset_split_manifest.json"),
        (standardizers_path, "standardizers.pt"),
        (checkpoint_path, "checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"source generative {label} does not exist: {path}")

    source_cfg = OmegaConf.load(source_config_path)
    source_summary = json.loads(source_summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    seed = _nonnegative_int(source_cfg.seed, "source seed")

    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA generative sampling requested but CUDA is unavailable")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("generic generative sampling is intentionally single-process")

    batch_size = _positive_int(cfg.batch_size, "batch_size")
    num_workers = _nonnegative_int(cfg.num_workers, "num_workers")
    sampling = _sampling_settings(cfg.sampling)

    dataset = instantiate(source_cfg.data)
    if not hasattr(dataset, "field_catalog"):
        raise TypeError("configured dataset must expose a field_catalog")
    task = instantiate(source_cfg.task)
    if not isinstance(task, NodeRegressionTask):
        raise TypeError("configured generative task must derive from NodeRegressionTask")
    for name in ("make_validation_problem", "make_model_probe", "sample_standardized", "sampler_name"):
        if not callable(getattr(task, name, None)):
            raise TypeError(f"configured generative task must implement callable '{name}'")

    standardizers = _load_standardizers(standardizers_path)
    _validate_generative_standardizers(standardizers)
    if standardizers.physical_nondimensionalization != task.physical_nondimensionalization:
        raise ValueError("saved standardizers and source task disagree on nondimensionalization")
    device_standardizers = standardizers.to(device=device, dtype=torch.float32)

    test_ids = _manifest_test_ids(manifest)
    sample_ids = dataset_sample_ids(dataset)
    index_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    missing_test_ids = [sample_id for sample_id in test_ids if sample_id not in index_by_id]
    if missing_test_ids:
        raise ValueError(
            "source test split contains sample IDs absent from the current dataset: "
            f"{missing_test_ids[:5]}"
        )
    test_indices = [index_by_id[sample_id] for sample_id in test_ids]

    collator = GraphTaskCollator(task, dataset.field_catalog, source_cfg.geometry)
    test_loader = make_loader(
        dataset,
        test_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collator=collator,
        shuffle=False,
        seed=seed,
    )

    probe_host = next(iter(test_loader))
    probe_scaled = standardizers.transform(probe_host)
    probe_problem = task.make_validation_problem(probe_scaled)
    model_probe = task.make_model_probe(probe_problem)
    if not isinstance(model_probe, NodeRegressionBatch):
        raise TypeError("make_model_probe must return NodeRegressionBatch")
    model, _ = instantiate_controlled_model(source_cfg.model, model_probe, seed=seed)
    model = model.to(device=device, dtype=torch.float32)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("generative checkpoint does not contain model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    sampler = task.sampler_name(
        steps=sampling["steps"],
        method=sampling["method"],
        solver=sampling["solver"],
        final_denoise=sampling["final_denoise"],
    )
    generation_name = _generation_name(
        sampler=sampler,
        sampling_eps=sampling["sampling_eps"],
        seed=sampling["seed"],
        override=cfg.output_name,
    )
    output_dir = run_dir / "generations" / generation_name
    if output_dir.exists() and not bool(cfg.overwrite):
        raise FileExistsError(
            f"generation output already exists: {output_dir}; set overwrite=true explicitly"
        )

    artifact, generation = _generate_test_population(
        model,
        test_loader,
        task=task,
        standardizers=device_standardizers,
        device=device,
        sampling=sampling,
        sampler=sampler,
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(artifact, output_dir / "generated_test.pt")
    OmegaConf.save(source_cfg, output_dir / "resolved_config.yaml", resolve=True)
    OmegaConf.save(cfg, output_dir / "generation_config.yaml", resolve=True)
    shutil.copy2(manifest_path, output_dir / "dataset_split_manifest.json")

    grid_shape = source_summary.get("grid_shape_2d")
    nodes_per_sample = source_summary.get("nodes_per_slice")
    summary = {
        "run_name": generation_name,
        "benchmark_run_name": f"{run_dir.name}__{generation_name}",
        "source_run_name": run_dir.name,
        "source_run_dir": str(run_dir),
        "source_checkpoint": checkpoint_path.name,
        "source_best_epoch": source_summary.get("best_epoch"),
        "task": source_summary.get("task", type(task).__name__),
        "model": type(model).__name__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_channels": list(artifact["channel_names"]),
        "prediction_type": source_summary.get("prediction_type"),
        "sampling_method": sampling["method"],
        "sampling_solver": sampling["solver"],
        "sampling_steps": sampling["steps"],
        "sampling_eps": sampling["sampling_eps"],
        "sampling_final_denoise": sampling["final_denoise"],
        "sampling_seed": sampling["seed"],
        "sampler": sampler,
        "model_evaluations": generation["model_evaluations"],
        "initial_state_distribution": "standard_normal_at_t1",
        "sampling_random_stream": "task_owned_method_specific_deterministic_stream",
        "grid_shape_2d": grid_shape,
        "nodes_per_slice": nodes_per_sample,
        "reference_population": "test",
        "comparison_mode": "unpaired_population",
        "generated_reference_pairing": False,
        "generation": generation,
        "device": str(device),
        "dtype": "float32",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def _generate_test_population(
    model: torch.nn.Module,
    loader: Any,
    *,
    task: NodeRegressionTask,
    standardizers: TaskStandardizers,
    device: torch.device,
    sampling: dict[str, Any],
    sampler: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    generated_standardized_parts: list[torch.Tensor] = []
    generated_nondimensional_parts: list[torch.Tensor] = []
    reference_nondimensional_parts: list[torch.Tensor] = []
    generated_ids: list[str] = []
    reference_ids: list[str] = []
    node_counts: list[int] = []
    channel_names: tuple[str, ...] | None = None
    generated_count = 0

    with torch.inference_mode():
        for host_batch in loader:
            batch = task_batch_to_device(host_batch, device=device, dtype=torch.float32)
            scaled = standardizers.transform(batch)
            batch_generated_ids = tuple(
                f"gen_{generated_count + index:06d}" for index in range(scaled.num_graphs)
            )
            generated_count += scaled.num_graphs
            generated_standardized = task.sample_standardized(
                model,
                scaled,
                steps=sampling["steps"],
                method=sampling["method"],
                solver=sampling["solver"],
                sampling_eps=sampling["sampling_eps"],
                final_denoise=sampling["final_denoise"],
                sampling_seed=sampling["seed"],
                sampling_keys=batch_generated_ids,
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
            reference_nondimensional_parts.append(batch.inputs.cpu())
            generated_ids.extend(batch_generated_ids)
            reference_ids.extend(batch.source.sample_ids)
            node_counts.extend(
                int(value) for value in (batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().tolist()
            )

    if channel_names is None:
        raise ValueError("generative sampling loader contains no samples")

    nfe = _model_evaluations(
        steps=sampling["steps"],
        method=sampling["method"],
        solver=sampling["solver"],
        final_denoise=sampling["final_denoise"],
    )
    artifact = {
        "generated_ids": tuple(generated_ids),
        "reference_ids": tuple(reference_ids),
        "node_counts": tuple(node_counts),
        "channel_names": channel_names,
        "generated_standardized": torch.cat(generated_standardized_parts, dim=0),
        "generated_nondimensional": torch.cat(generated_nondimensional_parts, dim=0),
        "target_nondimensional": torch.cat(reference_nondimensional_parts, dim=0),
        "sampling_steps": sampling["steps"],
        "sampling_method": sampling["method"],
        "sampling_solver": sampling["solver"],
        "sampling_eps": sampling["sampling_eps"],
        "sampling_final_denoise": sampling["final_denoise"],
        "sampler": sampler,
        "sampling_seed": sampling["seed"],
        "model_evaluations": nfe,
        "generated_reference_pairing": False,
    }
    generation = {
        "artifact": "generated_test.pt",
        "sampler": sampler,
        "sampling_method": sampling["method"],
        "sampling_solver": sampling["solver"],
        "sampling_steps": sampling["steps"],
        "sampling_eps": sampling["sampling_eps"],
        "sampling_final_denoise": sampling["final_denoise"],
        "sampling_seed": sampling["seed"],
        "initial_state_distribution": "standard_normal_at_t1",
        "random_stream_semantics": "task_owned_method_specific_deterministic_stream",
        "model_evaluations": nfe,
        "generated_ids": "independent_deterministic_sampling_keys",
        "reference_ids": "held_out_test_sample_ids",
        "reference_pairing": False,
        "num_generated_samples": len(generated_ids),
        "num_reference_samples": len(reference_ids),
    }
    return artifact, generation


def _sampling_settings(config: DictConfig) -> dict[str, Any]:
    steps = _positive_int(config.steps, "sampling.steps")
    method = str(config.method)
    solver = str(config.solver)
    sampling_eps = float(config.sampling_eps)
    if not torch.isfinite(torch.tensor(sampling_eps)) or not 0.0 < sampling_eps < 1.0:
        raise ValueError("sampling.sampling_eps must lie strictly between 0 and 1")
    final_denoise = config.final_denoise
    if not isinstance(final_denoise, bool):
        raise TypeError("sampling.final_denoise must be boolean")
    seed = _nonnegative_int(config.seed, "sampling.seed")
    return {
        "steps": steps,
        "method": method,
        "solver": solver,
        "sampling_eps": sampling_eps,
        "final_denoise": final_denoise,
        "seed": seed,
    }


def _model_evaluations(
    *,
    steps: int,
    method: str,
    solver: str,
    final_denoise: bool,
) -> int:
    if method == "probability_flow_ode":
        if solver == "euler":
            evaluations = steps
        elif solver == "heun":
            evaluations = 2 * steps
        else:
            raise ValueError("unsupported probability-flow ODE solver")
    elif method == "reverse_sde":
        if solver != "euler_maruyama":
            raise ValueError("unsupported reverse-SDE solver")
        evaluations = steps
    else:
        raise ValueError("unsupported generative sampling method")
    return evaluations + int(final_denoise)


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
            physical_nondimensionalization=bool(payload["physical_nondimensionalization"]),
            weighting=str(payload["weighting"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("standardizers.pt does not match the training artifact schema") from exc


def _validate_generative_standardizers(standardizers: TaskStandardizers) -> None:
    if standardizers.inputs.channel_names != standardizers.targets.channel_names:
        raise RuntimeError("generative input/target standardizers have different channels")
    if not torch.equal(standardizers.inputs.mean, standardizers.targets.mean):
        raise RuntimeError("generative input/target means must be identical")
    if not torch.equal(standardizers.inputs.scale, standardizers.targets.scale):
        raise RuntimeError("generative input/target scales must be identical")


def _manifest_test_ids(manifest: dict[str, Any]) -> tuple[str, ...]:
    values = manifest.get("test_ids")
    if not isinstance(values, list) or not values:
        raise ValueError("dataset split manifest must contain a non-empty test_ids list")
    test_ids = tuple(values)
    if any(not isinstance(value, str) or not value for value in test_ids):
        raise ValueError("dataset split manifest test_ids must be non-empty strings")
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("dataset split manifest test_ids must be unique")
    return test_ids


def _generation_name(
    *,
    sampler: str,
    sampling_eps: float,
    seed: int,
    override: object,
) -> str:
    if override is not None:
        value = str(override).strip()
        if not value:
            raise ValueError("output_name must be non-empty when provided")
        if Path(value).name != value or value in {".", ".."}:
            raise ValueError("output_name must be one directory name, not a path")
        return value
    endpoint = f"{sampling_eps:.8g}".replace(".", "p")
    return f"{sampler}_eps{endpoint}_seed{seed}"


def _existing_directory(value: object, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a filesystem path")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not an accessible directory: {path}")
    return path


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


if __name__ == "__main__":
    main()
