"""M41.1: compare original versus halo-partitioned M37 inference on one GH200.

Loads the real trained checkpoint and a held-out physical graph. Geometry-plan
construction and data loading are outside forward timings.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scripts.benchmark_dit_efficiency import _load_standardizers

from graph_attention.geometry import edge_relative_displacement
from graph_attention.geometry.partitioning import coordinate_block_partition_ids
from graph_attention.models.partitioned_dinat import (
    build_dinat_partition_plan,
    forward_partitioned_dinat,
)
from graph_attention.models.owned_query_dinat import (
    build_owned_query_plan,
    forward_owned_query_dinat,
)
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import (
    GraphTaskCollator,
    dataset_sample_ids,
    make_loader,
    task_batch_to_device,
)
from graph_attention.training.model_factory import instantiate_controlled_model
from graph_attention.utils.provenance import collect_runtime_provenance


def _time_forward(callback, device, *, warmup: int, repeats: int):
    with torch.inference_mode():
        for _ in range(warmup):
            callback()
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
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
        allocated_after = torch.cuda.memory_allocated(device)
    return {
        "times_ms": readings,
        "mean_ms": statistics.mean(readings),
        "median_ms": statistics.median(readings),
        "stdev_ms": statistics.stdev(readings) if len(readings) > 1 else 0.0,
        "baseline_allocated_gib": baseline / 1024**3,
        "allocated_after_gib": allocated_after / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "peak_allocated_gib": peak / 1024**3,
        "incremental_peak_allocated_gib": (peak - baseline) / 1024**3,
    }


def _errors(actual, reference, *, atol, rtol):
    actual, reference = actual.detach().double().cpu(), reference.detach().double().cpu()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    delta = (actual - reference).abs()
    threshold = atol + rtol * reference.abs()
    return {
        "passed": finite and bool((delta <= threshold).all()),
        "finite": finite,
        "max_abs": float(delta.max()) if finite else None,
        "rms": float(delta.square().mean().sqrt()) if finite else None,
        "relative_l2": float(
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference).clamp_min(1e-30)
        )
        if finite
        else None,
        "violations": int((delta > threshold).sum()) if finite else actual.numel(),
        "atol": atol,
        "rtol": rtol,
    }


class _PartitionedModel(torch.nn.Module):
    """Benchmark-only adapter; reuse the task's actual sampler without rewriting it."""

    def __init__(self, model, plan, *, reference_batch=None, executor="layerwise_full"):
        super().__init__()
        self.model, self.plan = model, plan
        self.reference_batch = reference_batch
        self.executor = executor

    def forward(self, inputs, *, conditioning, **kwargs):
        # The plan represents one immutable physical graph. Refuse a sampler
        # call on different geometry/topology rather than silently reusing it.
        if self.reference_batch is not None:
            for name in ("coords", "edge_index", "batch_index"):
                if kwargs.get(name) is not getattr(self.reference_batch, name):
                    raise ValueError(f"partition plan cannot be used with a different {name}")
            topologies = kwargs.get("attention_edge_indices")
            if (
                topologies is None
                or topologies.get("dilated")
                is not self.reference_batch.attention_edge_indices["dilated"]
            ):
                raise ValueError("partition plan cannot be used with a different dilated topology")
        run = (
            forward_owned_query_dinat
            if self.executor == "owned_query"
            else forward_partitioned_dinat
        )
        return run(self.model, inputs, plan=self.plan, conditioning=conditioning)


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _generate_with_trace(model, generate):
    """Capture only the first velocity input plus model-evaluation count."""
    first_inputs = []
    calls = 0

    def capture(_module, args):
        nonlocal calls
        calls += 1
        if not first_inputs:
            first_inputs.append(args[0].detach().cpu().clone())

    hook = model.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            sample = generate(model).cpu()
    finally:
        hook.remove()
    if not first_inputs:
        raise RuntimeError("Heun generation did not evaluate the model")
    return sample, calls, first_inputs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--blocks", nargs=3, type=int, default=(2, 2, 2))
    parser.add_argument(
        "--executor", choices=("layerwise_full", "owned_query"),
        default="layerwise_full",
        help="M41.1 reference or M41.2 owned-query inference",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sampling-seed", type=int, default=5678)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--expected-nodes", type=int, default=35937)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("M41.1 GH200 benchmark requires CUDA")
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    if args.sampling_steps < 1 or args.atol <= 0 or args.rtol < 0:
        raise ValueError("invalid sampling steps or tolerances")
    # Explicit FP32 reference policy; no autocast or TF32 approximation.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = OmegaConf.load(run_dir / "resolved_config.yaml")
    manifest = json.loads((run_dir / "dataset_split_manifest.json").read_text())
    standardizers = _load_standardizers(run_dir / "standardizers.pt")
    checkpoint = torch.load(run_dir / args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("M37 checkpoint must contain model_state_dict")

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
        dataset,
        [ids.index(test_id)],
        batch_size=1,
        num_workers=0,
        collator=collator,
        shuffle=False,
        seed=int(config.seed),
    )
    host = next(iter(loader))
    standardizers = standardizers.to(device=device, dtype=torch.float32)
    physical = task_batch_to_device(host, device=device, dtype=torch.float32)
    scaled = standardizers.transform(physical)
    problem = task.make_validation_problem(scaled)
    probe = task.make_model_probe(problem)
    model, _ = instantiate_controlled_model(
        config.model,
        probe,
        seed=int(config.seed),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()

    if scaled.num_graphs != 1 or len(problem.inputs) != args.expected_nodes:
        raise ValueError("benchmark requires one graph with --expected-nodes nodes")
    if model.sparse_attention_backend != "torch_sparse":
        raise ValueError("M37 target validation requires saved torch_sparse backend")
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

    executor_fn = (
        forward_owned_query_dinat
        if args.executor == "owned_query"
        else forward_partitioned_dinat
    )

    def partitioned():
        return executor_fn(model, inputs, plan=plan, conditioning=condition)

    # Capture untimed reference states on CPU; never retain them during timing.
    snapshots = []
    hooks = [
        block.register_forward_hook(
            lambda _m, _a, output: snapshots.append(output.detach().cpu().clone())
        )
        for block in model.blocks
    ]
    try:
        with torch.inference_mode():
            reference = original().cpu()
    finally:
        for hook in hooks:
            hook.remove()
    baseline = _time_forward(original, device, warmup=args.warmup, repeats=args.repeats)

    def generate(which):
        return task.sample_standardized(
            which,
            scaled,
            steps=args.sampling_steps,
            solver="heun",
            sampling_seed=args.sampling_seed,
            sampling_keys=(test_id,),
        )

    generated_reference, original_nfe, original_noise = _generate_with_trace(model, generate)
    generation_baseline = _time_forward(
        lambda: generate(model),
        device,
        warmup=0,
        repeats=args.repeats,
    )

    # Release reference-only global CSR caches before measuring partitioned
    # memory. Otherwise partitioned peak includes BOTH full-graph and halo CSR.
    torch.cuda.synchronize(device)
    before_cache_release = torch.cuda.memory_allocated(device)
    model._torch_sparse_topology_cache.clear()
    model._self_attention_edge_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    before_plan = torch.cuda.memory_allocated(device)
    released_reference_cache_gib = max(0, before_cache_release - before_plan) / 1024**3

    plan_start = time.perf_counter()
    owners = coordinate_block_partition_ids(problem.coords, blocks=args.blocks)
    plan_builder = (
        build_owned_query_plan
        if args.executor == "owned_query"
        else build_dinat_partition_plan
    )
    plan = plan_builder(
        model,
        coords=problem.coords,
        edge_index=problem.edge_index,
        attention_edge_indices=problem.attention_edge_indices,
        partition_ids=owners,
    )
    torch.cuda.synchronize(device)
    plan_seconds = time.perf_counter() - plan_start
    partitions = plan.base.partitions if args.executor == "owned_query" else plan.partitions
    plan_persistent_allocated_gib = (
        torch.cuda.memory_allocated(device) - before_plan
    ) / 1024**3
    topology_checks = []
    # Exact edge order/multiplicity, not just set equality; self loops included.
    for name, edges in (
        ("local", problem.edge_index),
        ("dilated", problem.attention_edge_indices["dilated"]),
    ):
        effective = model._attention_edges_with_optional_self(
            name,
            edges,
            num_nodes=len(inputs),
        )
        preserved = 0
        for part in partitions:
            attention = getattr(part, name)
            reconstructed = part.read_global[attention.edge_index]
            expected = effective[:, owners[effective[1]] == part.partition_id]
            assert torch.equal(reconstructed, expected), "partition edge mismatch"
            expected_displacement = edge_relative_displacement(problem.coords, expected)
            assert torch.equal(
                attention.displacement, expected_displacement
            ), "edge geometry mismatch"
            if attention.sparse_topology is not None:
                csr = attention.sparse_topology
                assert csr.num_nodes == len(part.read_global)
                assert torch.equal(csr.row_indices, attention.edge_index[1, csr.edge_order])
                assert torch.equal(csr.col_indices, attention.edge_index[0, csr.edge_order])
                assert torch.equal(
                    csr.crow_indices.diff(),
                    torch.bincount(attention.edge_index[1], minlength=csr.num_nodes),
                )
                assert int(csr.crow_indices[-1]) == attention.edge_index.shape[1]
                assert all(
                    not value.is_inference()
                    for value in (
                        csr.crow_indices, csr.col_indices, csr.row_indices,
                        csr.edge_order, csr.transpose_crow_indices,
                        csr.transpose_col_indices, csr.transpose_order,
                    )
                ), "cached CSR metadata must not be inference tensors"
            if args.executor == "owned_query":
                rectangle = getattr(plan.partitions[part.partition_id], name)
                expected_rows = torch.searchsorted(
                    part.owned_local, attention.edge_index[1]
                )[rectangle.edge_order]
                assert rectangle.pattern.shape == (
                    len(part.owned_global), len(part.read_global)
                ), "rectangular CSR shape mismatch"
                assert torch.equal(rectangle.row_indices, expected_rows)
                assert torch.equal(
                    rectangle.col_indices,
                    attention.edge_index[0, rectangle.edge_order],
                )
                assert torch.equal(
                    rectangle.pattern.crow_indices().diff(),
                    torch.bincount(expected_rows, minlength=len(part.owned_global)),
                ), "rectangular CSR row mismatch"
                assert not rectangle.pattern.crow_indices().is_inference()
            preserved += reconstructed.shape[1]
        topology_checks.append(
            {"topology": name, "edges": preserved, "passed": preserved == effective.shape[1]}
        )
    owned = torch.cat([part.owned_global for part in partitions])
    assert torch.equal(owned.sort().values, torch.arange(len(inputs), device=device))

    with torch.inference_mode():
        changed, layers = executor_fn(
            model,
            inputs,
            plan=plan,
            conditioning=condition,
            return_layer_states=True,
        )
        errors = {
            "output": _errors(changed, reference, atol=args.atol, rtol=args.rtol),
            "layers": [
                _errors(new, old, atol=args.atol, rtol=args.rtol)
                for new, old in zip(layers, snapshots, strict=True)
            ],
        }
        del changed, layers
        adapter = _PartitionedModel(
            model, plan, reference_batch=scaled, executor=args.executor
        ).eval()
        generated, partitioned_nfe, partitioned_noise = _generate_with_trace(
            adapter, generate
        )
        errors["heun"] = _errors(
            generated,
            generated_reference,
            atol=args.atol,
            rtol=args.rtol,
        )
        del generated
    generation_checks = {
        "expected_nfe": 2 * args.sampling_steps,
        "reference_nfe": original_nfe,
        "partitioned_nfe": partitioned_nfe,
        "identical_initial_noise": bool(torch.equal(original_noise, partitioned_noise)),
        "passed": (
            original_nfe == partitioned_nfe == 2 * args.sampling_steps
            and bool(torch.equal(original_noise, partitioned_noise))
        ),
    }
    result = _time_forward(partitioned, device, warmup=args.warmup, repeats=args.repeats)
    generation_result = _time_forward(
        lambda: generate(adapter),
        device,
        warmup=0,
        repeats=args.repeats,
    )
    checks = [
        errors["output"], errors["heun"], *errors["layers"],
        *topology_checks, generation_checks,
    ]
    payload = {
        "benchmark": (
            "m41_2_owned_query_halo_v1"
            if args.executor == "owned_query"
            else "m41_1_single_gpu_halo_v2"
        ),
        "executor": args.executor,
        "passed": all(check["passed"] for check in checks),
        "evidence": "TARGET_VALIDATED"
        if all(c["passed"] for c in checks) and "GH200" in torch.cuda.get_device_name(device)
        else "LOCAL_BENCHMARK",
        "provenance": collect_runtime_provenance(),
        "arguments": vars(args),
        "precision": "float32; no autocast; TF32 disabled",
        "resolved_config": OmegaConf.to_container(config, resolve=True),
        "dataset_split_manifest": manifest,
        "artifact_sha256": {
            name: _sha256(run_dir / name)
            for name in (
                args.checkpoint,
                "standardizers.pt",
                "resolved_config.yaml",
                "dataset_split_manifest.json",
            )
        },
        "sample_id": test_id,
        "num_nodes": len(inputs),
        "backend": model.sparse_attention_backend,
        "blocks": args.blocks,
        "num_partitions": len(partitions),
        "plan_build_seconds": plan_seconds,
        "plan_persistent_allocated_gib": plan_persistent_allocated_gib,
        "released_reference_cache_gib": released_reference_cache_gib,
        "partitions": [
            {
                "id": p.partition_id,
                "owned": len(p.owned_global),
                "halo": p.num_halo_nodes,
                "read": len(p.read_global),
                "local_edges": p.local.edge_index.shape[1],
                "dilated_edges": p.dilated.edge_index.shape[1],
            }
            for p in partitions
        ],
        "nodewise_work_ratio": sum(len(p.read_global) for p in partitions) / len(inputs),
        "topology_checks": topology_checks,
        "generation_checks": generation_checks,
        "errors": errors,
        "timing_scope": "warm forward; excludes data/transfers/plan; includes gathers and writes",
        "memory_scope": (
            "each path retains only its own CSR cache; full reference CSR is released "
            "before plan build. Peak allocated includes model, data, and persistent "
            "topology; incremental peak excludes resident topology. "
            "Plan memory is device allocation delta during plan build."
        ),
        "original": baseline,
        "partitioned": result,
        "forward_overhead_percent": 100 * (result["mean_ms"] / baseline["mean_ms"] - 1),
        "speedup_original_over_partitioned": baseline["mean_ms"] / result["mean_ms"],
        "generation": {
            "steps": args.sampling_steps,
            "nfe": 2 * args.sampling_steps,
            "seed": args.sampling_seed,
            "original": generation_baseline,
            "partitioned": generation_result,
        },
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))
    if not payload["passed"]:
        raise SystemExit(
            "Partitioned inference numerical gate failed; inspect saved per-layer/Heun errors"
        )


if __name__ == "__main__":
    main()
