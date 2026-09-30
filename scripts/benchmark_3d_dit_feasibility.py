"""Feasibility benchmark for Full DiT and NAT-DiT on one real 3-D HIT volume."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from graph_attention.data import AVBPDirectoryHDF5Dataset
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model


def _forward_model(model: torch.nn.Module, batch: Any) -> torch.Tensor:
    kwargs = {
        "edge_index": batch.edge_index,
        "coords": batch.coords,
        "batch_index": batch.batch_index,
        "conditioning": batch.conditioning,
    }
    if batch.attention_edge_indices:
        kwargs["attention_edge_indices"] = batch.attention_edge_indices
    return model(batch.inputs, **kwargs)


def _memory_gib(device: torch.device) -> dict[str, float]:
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def _benchmark_variant(
    label: str,
    model_cfg: Any,
    probe: Any,
    problem: Any,
    task: FlowMatchingTask,
    *,
    device: torch.device,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = None
    optimizer = None
    try:
        model, initialization = instantiate_controlled_model(
            model_cfg,
            probe,
            seed=42,
        )
        model = model.to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = _forward_model(model, problem)
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - start
        forward_memory = _memory_gib(device)

        del output
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        predictions = _forward_model(model, problem)
        aggregate = task.training_loss(predictions, problem)
        aggregate.mean.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - start
        train_memory = _memory_gib(device)

        return {
            "label": label,
            "status": "ok",
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "initialization": initialization,
            "forward_seconds": forward_seconds,
            "forward": forward_memory,
            "training_step_seconds": train_seconds,
            "training": train_memory,
            "loss": float(aggregate.mean.detach().cpu()),
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "label": label,
            "status": "cuda_oom",
            "error": str(exc),
        }
    finally:
        if optimizer is not None:
            del optimizer
        if model is not None:
            del model
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("/scratch/coop/theret/HIT_LES_FORCED/RUN/SOLUT"),
    )
    parser.add_argument(
        "--mesh-file",
        type=Path,
        default=Path("/scratch/coop/theret/HIT_LES_FORCED/MESH/mesh.mesh.h5"),
    )
    parser.add_argument(
        "--case-file",
        type=Path,
        default=Path("cases/HIT_LES_FORCED.yaml"),
    )
    parser.add_argument("--snapshot-pattern", default="*.h5")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/scratch/coop/theret/GraphAttention_runs/"
            "hit_3d_dit_feasibility/summary.json"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("3-D feasibility benchmark requires CUDA")
    device = torch.device("cuda")

    dataset = AVBPDirectoryHDF5Dataset(
        snapshot_dir=args.snapshot_dir,
        mesh_file=args.mesh_file,
        snapshot_pattern=args.snapshot_pattern,
        mesh_id="HIT_LES_FORCED:3d_fixed_mesh",
        case_file=args.case_file,
        case_id="HIT_LES_FORCED",
    )
    task = FlowMatchingTask(
        state_fields=("rho", "rhou", "rhov", "rhow", "rhoE"),
        physical_nondimensionalization=True,
        validation_seed=1234,
        time_embedding_scale=1000.0,
    )

    geometry_cfg = OmegaConf.create(
        {"attention_edge_indices": {"dilated": "exact_two_hop"}}
    )
    collator = GraphTaskCollator(task, dataset.field_catalog, geometry_cfg)

    topology_start = time.perf_counter()
    host_batch = collator([dataset[0]])
    topology_seconds = time.perf_counter() - topology_start

    batch = task_batch_to_device(host_batch, device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(1042)
    problem = task.make_training_problem(batch, generator=generator)

    full_cfg = OmegaConf.load("configs/model/full_dit_transformer.yaml")
    full_cfg.hidden_dim = 128
    full_cfg.num_heads = 4
    full_cfg.num_layers = 10
    full_cfg.mlp_ratio = 4
    full_cfg.condition_embed_dim = 128
    full_cfg.dropout = 0.0
    full_cfg.qkv_bias = True
    full_cfg.out_proj_bias = False
    full_cfg.use_sdpa = True

    dinat_cfg = OmegaConf.load("configs/model/dinat_dit.yaml")
    dinat_cfg.hidden_dim = 128
    dinat_cfg.num_heads = 4
    dinat_cfg.num_layers = 10
    dinat_cfg.mlp_ratio = 4
    dinat_cfg.condition_embed_dim = 128
    dinat_cfg.dropout = 0.0
    dinat_cfg.qkv_bias = True
    dinat_cfg.out_proj_bias = False
    dinat_cfg.sparse_attention_backend = "torch_sparse"

    payload = {
        "dataset": {
            "num_snapshots": len(dataset),
            "sample_id": host_batch.source.sample_ids[0],
            "nodes": int(host_batch.inputs.shape[0]),
            "state_channels": list(host_batch.input_channels),
            "spatial_dim": int(host_batch.coords.shape[1]),
            "local_directed_edges": int(host_batch.edge_index.shape[1]),
            "dilated_directed_edges": int(
                host_batch.attention_edge_indices["dilated"].shape[1]
            ),
            "topology_build_seconds": topology_seconds,
            "periodic_edges": False,
        },
        "batch_size": 1,
        "dtype": "float32",
        "device": torch.cuda.get_device_name(device),
        "variants": [],
    }

    for label, cfg in (
        ("full_dit", full_cfg),
        ("dinat_dit_optimized", dinat_cfg),
    ):
        print(f"===== {label} =====", flush=True)
        result = _benchmark_variant(
            label,
            cfg,
            problem,
            problem,
            task,
            device=device,
        )
        payload["variants"].append(result)
        print(json.dumps(result, indent=2), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
