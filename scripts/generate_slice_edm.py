"""Generate HIT 2-D slices from an EDM-preconditioned diffusion run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from scripts.generate_slice_diffusion import (
    _load_standardizers,
    _manifest_test_ids,
)
from scripts.train_slice_ablation import (
    _attention_topologies_from_geometry,
    _loader,
    _NodeRegressionCollator,
    _positive_int,
    _task_batch_to_device,
)
from scripts.train_slice_diffusion import _validate_diffusion_standardizers

from graph_attention.data import PrecomputedSlicePTDataset
from graph_attention.geometry import cartesian_4_neighbor_edge_index
from graph_attention.tasks import EDMDenoisingTask
from graph_attention.training.model_factory import instantiate_controlled_model


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="generate_slice_edm",
)
def main(cfg: DictConfig) -> None:
    run_edm_generation(cfg)


def run_edm_generation(cfg: DictConfig) -> dict[str, Any]:
    """Generate one unpaired test-sized population from an EDM run."""

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
            raise FileNotFoundError(f"source EDM {label} does not exist: {path}")

    source_cfg = OmegaConf.load(source_config_path)
    source_summary = json.loads(source_summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    seed = _positive_int(source_cfg.seed, "source seed", allow_zero=True)

    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA EDM generation requested but CUDA is unavailable")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("EDM generation is intentionally single-process")

    batch_size = _positive_int(cfg.batch_size, "batch_size")
    num_workers = _positive_int(cfg.num_workers, "num_workers", allow_zero=True)
    sampling_steps = _positive_int(cfg.sampling.steps, "sampling.steps")
    sampling_seed = _positive_int(cfg.sampling.seed, "sampling.seed", allow_zero=True)
    sigma_min = _positive_float(cfg.sampling.sigma_min, "sampling.sigma_min")
    sigma_max = _positive_float(cfg.sampling.sigma_max, "sampling.sigma_max")
    rho = _positive_float(cfg.sampling.rho, "sampling.rho")
    solver = str(cfg.sampling.solver)
    if solver not in {"euler", "heun"}:
        raise ValueError("sampling.solver must be 'euler' or 'heun'")

    dataset = instantiate(source_cfg.data)
    if not isinstance(dataset, PrecomputedSlicePTDataset):
        raise TypeError("source EDM run must use data=hit_slice_pt")
    task = instantiate(source_cfg.task)
    if not isinstance(task, EDMDenoisingTask):
        raise TypeError("source run must use task=hit_edm_diffusion")

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
    probe_problem = task.make_validation_problem(probe_scaled)
    model_probe = task.make_model_probe(probe_problem)
    model, _ = instantiate_controlled_model(source_cfg.model, model_probe, seed=seed)
    model = model.to(device=device, dtype=torch.float32)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("EDM checkpoint does not contain model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    sampler = task.sampler_name(steps=sampling_steps, solver=solver)
    generation_name = _generation_name(
        sampler=sampler,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        seed=sampling_seed,
        override=cfg.output_name,
    )
    output_dir = run_dir / "generations" / generation_name
    if output_dir.exists() and not bool(cfg.overwrite):
        raise FileExistsError(
            f"EDM generation output already exists: {output_dir}; "
            "set overwrite=true explicitly"
        )

    artifact, generation = _generate_test_population(
        model,
        test_loader,
        task=task,
        standardizers=device_standardizers,
        device=device,
        sampling_steps=sampling_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        solver=solver,
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
        "task": "edm_preconditioned_continuous_noise_denoising",
        "model": type(model).__name__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_channels": list(artifact["channel_names"]),
        "prediction_type": task.prediction_type,
        "sigma_data": task.sigma_data,
        "training_p_mean": task.p_mean,
        "training_p_std": task.p_std,
        "sampling_schedule": "karras_power_law_sigma",
        "sampling_solver": solver,
        "sampling_steps": sampling_steps,
        "sampling_sigma_min": sigma_min,
        "sampling_sigma_max": sigma_max,
        "sampling_rho": rho,
        "sampling_churn": 0.0,
        "time_conditioning": "edm_log_sigma_over_4",
        "grid_shape_2d": list(dataset.grid_shape_2d),
        "nodes_per_slice": dataset.grid_shape_2d[0] * dataset.grid_shape_2d[1],
        "reference_population": "test",
        "comparison_mode": "unpaired_population",
        "generated_reference_pairing": False,
        "generation": generation,
        "device": str(device),
        "dtype": "float32_network_float64_sampler_state",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def _generate_test_population(
    model: torch.nn.Module,
    loader: Any,
    *,
    task: EDMDenoisingTask,
    standardizers: Any,
    device: torch.device,
    sampling_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    solver: str,
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
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                solver=solver,
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
                int(value) for value in (batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().tolist()
            )

    if channel_names is None:
        raise ValueError("EDM generation loader contains no samples")

    sampler = task.sampler_name(steps=sampling_steps, solver=solver)
    nfe = sampling_steps if solver == "euler" else 2 * sampling_steps - 1
    artifact = {
        "generated_ids": tuple(generated_ids),
        "reference_ids": tuple(reference_ids),
        "node_counts": tuple(node_counts),
        "channel_names": channel_names,
        "generated_standardized": torch.cat(generated_standardized_parts, dim=0),
        "generated_nondimensional": torch.cat(generated_nondimensional_parts, dim=0),
        "target_nondimensional": torch.cat(reference_nondimensional_parts, dim=0),
        "sampling_steps": sampling_steps,
        "sampler": sampler,
        "sampling_seed": sampling_seed,
        "model_evaluations": nfe,
        "generated_reference_pairing": False,
    }
    generation = {
        "artifact": "generated_test.pt",
        "sampler": sampler,
        "sampling_schedule": "karras_power_law_sigma",
        "sampling_solver": solver,
        "sampling_steps": sampling_steps,
        "sampling_sigma_min": sigma_min,
        "sampling_sigma_max": sigma_max,
        "sampling_rho": rho,
        "sampling_seed": sampling_seed,
        "sampling_churn": 0.0,
        "model_evaluations": nfe,
        "generated_ids": "independent_deterministic_sampling_keys",
        "reference_ids": "held_out_test_sample_ids",
        "reference_pairing": False,
        "num_generated_samples": len(generated_ids),
        "num_reference_samples": len(reference_ids),
    }
    return artifact, generation


def _generation_name(
    *,
    sampler: str,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    seed: int,
    override: object,
) -> str:
    if override is not None:
        value = str(override).strip()
        if not value:
            raise ValueError("output_name must be non-empty when provided")
        return value
    sigma_min_text = str(sigma_min).replace(".", "p")
    sigma_max_text = str(sigma_max).replace(".", "p")
    rho_text = str(rho).replace(".", "p")
    return (
        f"{sampler}_sigmin{sigma_min_text}_sigmax{sigma_max_text}_"
        f"rho{rho_text}_seed{seed}"
    )


def _existing_directory(value: object, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a filesystem path")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not an accessible directory: {path}")
    return path


def _positive_float(value: object, name: str) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


if __name__ == "__main__":
    main()
