"""Find the largest 3-D microbatch that survives real training steps on one GPU."""

from __future__ import annotations

import argparse
import gc
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


def _model_and_geometry(kind: str) -> tuple[Any, Any]:
    if kind == "dinat":
        model_cfg = OmegaConf.load("configs/model/dinat_dit.yaml")
        model_cfg.sparse_attention_backend = "torch_sparse"
        geometry_cfg = OmegaConf.create(
            {"attention_edge_indices": {"dilated": "exact_two_hop"}}
        )
    elif kind == "full":
        model_cfg = OmegaConf.load("configs/model/full_dit_transformer.yaml")
        model_cfg.use_sdpa = True
        geometry_cfg = OmegaConf.create({"attention_edge_indices": {}})
    else:
        raise ValueError(f"unsupported model kind: {kind}")

    model_cfg.hidden_dim = 128
    model_cfg.num_heads = 4
    model_cfg.num_layers = 10
    model_cfg.mlp_ratio = 4
    model_cfg.condition_embed_dim = 128
    model_cfg.dropout = 0.0
    model_cfg.qkv_bias = True
    model_cfg.out_proj_bias = False
    return model_cfg, geometry_cfg


def _attempt(
    *,
    batch_size: int,
    dataset: AVBPDirectoryHDF5Dataset,
    task: FlowMatchingTask,
    collator: GraphTaskCollator,
    model_cfg: Any,
    device: torch.device,
) -> dict[str, Any]:
    model = None
    optimizer = None
    host_batch = None
    batch = None
    problem = None
    try:
        samples = [dataset[index] for index in range(batch_size)]
        host_batch = collator(samples)
        batch = task_batch_to_device(host_batch, device=device, dtype=torch.float32)
        generator = torch.Generator(device=device).manual_seed(1042)
        problem = task.make_training_problem(batch, generator=generator)

        model, _ = instantiate_controlled_model(model_cfg, problem, seed=42)
        model = model.to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()

        # Two steps ensure AdamW state has been allocated and exercise reuse of
        # cached sparse topology on the second step.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            predictions = _forward_model(model, problem)
            aggregate = task.training_loss(predictions, problem)
            aggregate.mean.backward()
            optimizer.step()

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        return {
            "batch_size": batch_size,
            "status": "ok",
            "two_step_seconds": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "batch_size": batch_size,
            "status": "cuda_oom",
            "error": str(exc),
        }
    finally:
        optimizer = None
        model = None
        problem = None
        batch = None
        host_batch = None
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("dinat", "full"), required=True)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
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
    args = parser.parse_args()

    if args.max_batch < 1:
        raise ValueError("--max-batch must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("batch-size probe requires CUDA")
    device = torch.device("cuda")

    dataset = AVBPDirectoryHDF5Dataset(
        snapshot_dir=args.snapshot_dir,
        mesh_file=args.mesh_file,
        snapshot_pattern="*.h5",
        mesh_id="HIT_LES_FORCED:3d_fixed_mesh",
        case_file=args.case_file,
        case_id="HIT_LES_FORCED",
    )
    if args.max_batch > len(dataset):
        raise ValueError("--max-batch exceeds dataset size")

    task = FlowMatchingTask(
        state_fields=("rho", "rhou", "rhov", "rhow", "rhoE"),
        physical_nondimensionalization=True,
        validation_seed=1234,
        time_embedding_scale=1000.0,
    )
    model_cfg, geometry_cfg = _model_and_geometry(args.model)
    collator = GraphTaskCollator(task, dataset.field_catalog, geometry_cfg)

    attempts: list[dict[str, Any]] = []
    low = 1
    high = args.max_batch
    best = 0

    while low <= high:
        candidate = (low + high) // 2
        print(f"===== probing {args.model} batch={candidate} =====", flush=True)
        result = _attempt(
            batch_size=candidate,
            dataset=dataset,
            task=task,
            collator=collator,
            model_cfg=model_cfg,
            device=device,
        )
        attempts.append(result)
        print(json.dumps(result, indent=2), flush=True)

        if result["status"] == "ok":
            best = candidate
            low = candidate + 1
        else:
            high = candidate - 1

    if best < 1:
        raise RuntimeError("even batch size 1 failed")

    payload = {
        "model": args.model,
        "max_requested_batch": args.max_batch,
        "max_passing_batch": best,
        "device": torch.cuda.get_device_name(device),
        "attempts": attempts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("===== selected batch =====", flush=True)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
