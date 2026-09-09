"""Generate HIT 2-D slices from an existing trained diffusion run."""

from __future__ import annotations

import json
import shutil
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
)
from scripts.train_slice_diffusion import _validate_diffusion_standardizers

from graph_attention.data import PrecomputedSlicePTDataset
from graph_attention.geometry import cartesian_4_neighbor_edge_index
from graph_attention.tasks import DiffusionDenoisingTask
from graph_attention.training import ChannelStandardizer, TaskStandardizers


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="generate_slice_diffusion",
)
def main(cfg: DictConfig) -> None:
    run_diffusion_generation(cfg)


def run_diffusion_generation(cfg: DictConfig) -> dict[str, Any]:
    """Generate one unpaired test-sized population from a trained diffusion model."""

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
            raise FileNotFoundError(f"source diffusion {label} does not exist: {path}")

    source_cfg = OmegaConf.load(source_config_path)
    source_summary = json.loads(source_summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    seed = _positive_int(source_cfg.seed, "source seed", allow_zero=True)

    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diffusion generation requested but CUDA is unavailable")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("diffusion generation is intentionally single-process")

    batch_size = _positive_int(cfg.batch_size, "batch_size")
    num_workers = _positive_int(cfg.num_workers, "num_workers", allow_zero=True)
    sampling_steps = _positive_int(cfg.sampling.steps, "sampling.steps")
    sampling_seed = _positive_int(cfg.sampling.seed, "sampling.seed", allow_zero=True)
    sampling_eta = float(cfg.sampling.eta)
    if not 0.0 <= sampling_eta <= 1.0:
        raise ValueError("sampling.eta must lie in [0, 1]")

    dataset = instantiate(source_cfg.data)
    if not isinstance(dataset, PrecomputedSlicePTDataset):
        raise TypeError("source diffusion run must use data=hit_slice_pt")
    task = instantiate(source_cfg.task)
    if not isinstance(task, DiffusionDenoisingTask):
        raise TypeError("source diffusion run must use task=hit_diffusion")
    if sampling_steps > task.timesteps:
        raise ValueError(f"sampling.steps must be <= task.timesteps ({task.timesteps})")

    standardizers = _load_standardizers(standardizers_path)
    _validate_diffusion_standardizers(standardizers)
    if standardizers.physical_nondimensionalization != task.physical_nondimensionalization:
        raise ValueError("saved standardizers and source task disagree on nondimensionalization")
    device_standardizers = standardizers.to(device=device, dtype=torch.float32)

    test_ids = _manifest_test_ids(manifest)
    index_by_id = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}
    missing_test_ids = [sample_id for sample_id in test_ids if sample_id not in index_by_id]
    if missing_test_ids:
        raise ValueError(
            "source test split contains sample IDs absent from the current dataset: "
            f"{missing_test_ids[:5]}"
        )
    test_indices = [index_by_id[sample_id] for sample_id in test_ids]

    edge_index = cartesian_4_neighbor_edge_index(dataset.grid_shape_2d)
    attention_edge_indices = _attention_topologies_from_geometry(
        source_cfg.geometry,
        edge_index=edge_index,
        num_nodes=dataset.grid_shape_2d[0] * dataset.grid_shape_2d[1],
    )
    collator = _NodeRegressionCollator(
        task,
        dataset.field_catalog,
        edge_index,
        attention_edge_indices=attention_edge_indices,
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

    probe = next(iter(test_loader))
    probe_scaled = standardizers.transform(probe)
    probe_diffusion = task.make_validation_problem(probe_scaled)
    model, _ = _instantiate_model(source_cfg.model, probe_diffusion, seed=seed)
    model = model.to(device=device, dtype=torch.float32)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("diffusion checkpoint does not contain model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    sampler = task.sampler_name(steps=sampling_steps, eta=sampling_eta)
    generation_name = _generation_name(
        sampler=sampler,
        steps=sampling_steps,
        eta=sampling_eta,
        seed=sampling_seed,
        override=cfg.output_name,
    )
    output_dir = run_dir / "generations" / generation_name
    if output_dir.exists() and not bool(cfg.overwrite):
        raise FileExistsError(
            f"diffusion generation output already exists: {output_dir}; "
            "set overwrite=true explicitly"
        )

    artifact, generation = _generate_test_population(
        model,
        test_loader,
        task=task,
        standardizers=device_standardizers,
        device=device,
        sampling_steps=sampling_steps,
        sampling_eta=sampling_eta,
        sampling_seed=sampling_seed,
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(artifact, output_dir / "generated_test.pt")
    OmegaConf.save(source_cfg, output_dir / "resolved_config.yaml", resolve=True)
    OmegaConf.save(cfg, output_dir / "generation_config.yaml", resolve=True)
    shutil.copy2(manifest_path, output_dir / "dataset_split_manifest.json")

    summary = {
        "run_name": generation_name,
        "benchmark_run_name": f"{run_dir.name}__{generation_name}",
        "source_run_name": run_dir.name,
        "source_run_dir": str(run_dir),
        "source_checkpoint": checkpoint_path.name,
        "source_best_epoch": source_summary.get("best_epoch"),
        "task": "discrete_ddpm_epsilon_prediction",
        "model": type(model).__name__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": source_summary.get("best_epoch"),
        "state_channels": list(artifact["channel_names"]),
        "prediction_type": "epsilon",
        "timesteps": task.timesteps,
        "noise_schedule": "cosine",
        "cosine_s": task.cosine_s,
        "time_conditioning": "normalized_discrete_t_over_T",
        "grid_shape_2d": list(dataset.grid_shape_2d),
        "nodes_per_slice": dataset.grid_shape_2d[0] * dataset.grid_shape_2d[1],
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
    task: DiffusionDenoisingTask,
    standardizers: TaskStandardizers,
    device: torch.device,
    sampling_steps: int,
    sampling_eta: float,
    sampling_seed: int,
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
            batch = _task_batch_to_device(
                host_batch,
                device=device,
                dtype=torch.float32,
            )
            scaled = standardizers.transform(batch)
            batch_generated_ids = tuple(
                f"gen_{generated_count + index:06d}" for index in range(scaled.num_graphs)
            )
            generated_count += scaled.num_graphs
            generated_standardized = task.sample_standardized(
                model,
                scaled,
                steps=sampling_steps,
                eta=sampling_eta,
                sampling_seed=sampling_seed,
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
                int(value)
                for value in (batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().tolist()
            )

    if channel_names is None:
        raise ValueError("test generation loader contains no samples")

    sampler = task.sampler_name(steps=sampling_steps, eta=sampling_eta)
    artifact = {
        "generated_ids": tuple(generated_ids),
        "reference_ids": tuple(reference_ids),
        "node_counts": tuple(node_counts),
        "channel_names": channel_names,
        "generated_standardized": torch.cat(generated_standardized_parts, dim=0),
        "generated_nondimensional": torch.cat(generated_nondimensional_parts, dim=0),
        "target_nondimensional": torch.cat(reference_nondimensional_parts, dim=0),
        "sampling_steps": sampling_steps,
        "sampling_eta": sampling_eta,
        "sampler": sampler,
        "sampling_seed": sampling_seed,
        "model_evaluations": sampling_steps,
        "generated_reference_pairing": False,
    }
    generation = {
        "artifact": "generated_test.pt",
        "sampler": sampler,
        "sampling_steps": sampling_steps,
        "sampling_eta": sampling_eta,
        "sampling_seed": sampling_seed,
        "model_evaluations": sampling_steps,
        "is_exact_ancestral_ddpm": sampler == "ddpm_ancestral",
        "generated_ids": "independent_deterministic_sampling_keys",
        "reference_ids": "held_out_test_sample_ids",
        "reference_pairing": False,
        "num_generated_samples": len(generated_ids),
        "num_reference_samples": len(reference_ids),
    }
    return artifact, generation


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
        raise ValueError("standardizers.pt does not match the training artifact schema") from exc


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
    steps: int,
    eta: float,
    seed: int,
    override: object,
) -> str:
    if override is not None:
        if not isinstance(override, str) or not override.strip():
            raise ValueError("output_name must be null or a non-empty string")
        name = override.strip()
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("output_name must be one directory name, not a path")
        return name
    eta_label = f"{eta:.6g}".replace(".", "p")
    return f"{sampler}_steps{steps}_eta{eta_label}_seed{seed}"


def _existing_directory(value: object, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a filesystem path")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not an accessible directory: {path}")
    return path


if __name__ == "__main__":
    main()
