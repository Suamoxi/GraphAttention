"""M41.1: compare original versus halo-partitioned M37 inference on one GH200.

Loads the real trained checkpoint and a held-out physical graph. Geometry-plan
construction and data loading are outside forward timings.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from graph_attention.geometry.partitioning import coordinate_block_partition_ids
from graph_attention.models.partitioned_dinat import (
    build_dinat_partition_plan,
    forward_partitioned_dinat,
)
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import (
    GraphTaskCollator, dataset_sample_ids, make_loader, task_batch_to_device,
)
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.benchmark_dit_efficiency import _load_standardizers


def _time_forward(callback, device, *, warmup: int, repeats: int):
    with torch.inference_mode():
        for _ in range(warmup):
            callback()
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        readings = []
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            callback()
            torch.cuda.synchronize(device)
            readings.append((time.perf_counter() - start) * 1000)
        peak = torch.cuda.max_memory_allocated(device)
    return {
        "times_ms": readings,
        "mean_ms": sum(readings) / len(readings),
        "peak_allocated_gib": peak / 1024**3,
        "incremental_peak_allocated_gib": (peak - baseline) / 1024**3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--blocks", nargs=3, type=int, default=(2, 2, 2))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("M41.1 GH200 benchmark requires CUDA")
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    device = torch.device("cuda")
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = OmegaConf.load(run_dir / "resolved_config.yaml")
    manifest = json.loads((run_dir / "dataset_split_manifest.json").read_text())
    standardizers = _load_standardizers(run_dir / "standardizers.pt")
    checkpoint = torch.load(
        run_dir / args.checkpoint, map_location="cpu", weights_only=True
    )

    dataset = instantiate(config.data)
    task = instantiate(config.task)
    if not isinstance(task, FlowMatchingTask):
        raise TypeError("M41.1 benchmark expects FlowMatchingTask")
    ids = dataset_sample_ids(dataset)
    test_id = manifest["test_ids"][0]
    if test_id not in ids:
        raise ValueError("held-out test sample missing from configured dataset")
    collator = GraphTaskCollator(task, dataset.field_catalog, config.geometry)
    loader = make_loader(
        dataset, [ids.index(test_id)], batch_size=1, num_workers=0,
        collator=collator, shuffle=False, seed=int(config.seed),
    )
    host = next(iter(loader))
    standardizers = standardizers.to(device=device, dtype=torch.float32)
    physical = task_batch_to_device(host, device=device, dtype=torch.float32)
    scaled = standardizers.transform(physical)
    problem = task.make_validation_problem(scaled)
    probe = task.make_model_probe(problem)
    model, _ = instantiate_controlled_model(
        config.model, probe, seed=int(config.seed),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()

    owners = coordinate_block_partition_ids(problem.coords, blocks=args.blocks)
    plan = build_dinat_partition_plan(
        model,
        coords=problem.coords,
        edge_index=problem.edge_index,
        attention_edge_indices=problem.attention_edge_indices,
        partition_ids=owners,
    )
    inputs = problem.inputs
    condition = problem.conditioning

    def original():
        return model(
            inputs,
            edge_index=problem.edge_index,
            coords=problem.coords,
            batch_index=problem.batch_index,
            conditioning=condition,
            attention_edge_indices=problem.attention_edge_indices,
        )

    def partitioned():
        return forward_partitioned_dinat(
            model, inputs, plan=plan, conditioning=condition,
        )

    with torch.inference_mode():
        reference = original()
        changed = partitioned()
        difference = changed - reference
        max_abs = float(difference.abs().max().item())
        rms = float(difference.square().mean().sqrt().item())
        torch.testing.assert_close(changed, reference, atol=2e-4, rtol=2e-4)

    baseline = _time_forward(original, device, warmup=args.warmup, repeats=args.repeats)
    result = _time_forward(partitioned, device, warmup=args.warmup, repeats=args.repeats)
    payload = {
        "benchmark": "m41_1_single_gpu_halo_v1",
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "gpu": torch.cuda.get_device_name(device),
        "blocks": list(args.blocks),
        "num_nodes": int(inputs.shape[0]),
        "num_partitions": len(plan.partitions),
        "num_halo_nodes_total": sum(p.num_halo_nodes for p in plan.partitions),
        "owned_nodes_total": sum(int(p.owned_global.numel()) for p in plan.partitions),
        "local_edges": int(problem.edge_index.shape[1]),
        "dilated_edges": int(problem.attention_edge_indices["dilated"].shape[1]),
        "max_abs_error": max_abs,
        "rms_error": rms,
        "atol": 2e-4,
        "rtol": 2e-4,
        "timing_scope": "forward_only_excludes_plan_build_data_and_transfers",
        "original": baseline,
        "partitioned": result,
        "speedup_original_over_partitioned": baseline["mean_ms"] / result["mean_ms"],
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
