"""Benchmark the M7 framework/null-model baseline with explicit provenance."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from graph_attention.data import (
    AVBP_FIELD_CATALOG,
    AVBPHDF5Dataset,
    AVBPSampleSpec,
    FieldCatalog,
    PackedBatch,
    Sample,
    SplitManifest,
    SyntheticMeshDataset,
)
from graph_attention.geometry import mesh_with_hex_edge_index
from graph_attention.models import NodeLinearBaseline
from graph_attention.tasks import NodeRegressionBatch, NodeRegressionTask
from graph_attention.training import fit_train_standardizers, train_equal_sample_optimizer_step
from graph_attention.utils.benchmarking import BenchmarkMeasurement, measure_callable
from graph_attention.utils.provenance import collect_runtime_provenance

_EVIDENCE_LEVELS = ("LOCAL_BENCHMARK", "TARGET_VALIDATED")
_AVBP_STATE_FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")
_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="PyTorch device, e.g. cuda or cpu")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--evidence", choices=_EVIDENCE_LEVELS, default="LOCAL_BENCHMARK")
    parser.add_argument("--output", type=Path, required=True)

    subparsers = parser.add_subparsers(dest="workload", required=True)

    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--graphs", type=int, default=4)
    synthetic.add_argument("--nodes-per-graph", type=int, default=8192)
    synthetic.add_argument("--spatial-dim", type=int, choices=(2, 3), default=3)
    synthetic.add_argument("--seed", type=int, default=42)

    avbp = subparsers.add_parser("avbp")
    avbp.add_argument("--snapshot", type=Path, required=True)
    avbp.add_argument("--mesh", type=Path, required=True)
    avbp.add_argument("--case-file", type=Path, required=True)
    avbp.add_argument("--case-id", default="HIT_LES_FORCED")
    avbp.add_argument("--mesh-id", default="HIT_LES_FORCED")
    avbp.add_argument(
        "--connectivity-indexing",
        choices=("auto", "zero", "one"),
        default="one",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    dtype = _DTYPES[args.dtype]
    repo_root = Path(__file__).resolve().parents[1]

    setup_started = perf_counter()
    if args.workload == "synthetic":
        samples, catalog, task, workload = _synthetic_workload(args)
    else:
        samples, catalog, task, workload = _avbp_workload(args)
    setup_seconds = perf_counter() - setup_started

    host_prepare = measure_callable(
        lambda: task.pack_and_prepare(samples, catalog),
        device="cpu",
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    split = SplitManifest(train_ids=tuple(sample.sample_id for sample in samples))
    standardizers = fit_train_standardizers(task, samples, catalog, split)
    host_batch = task.pack_and_prepare(samples, catalog)
    device_batch = _task_batch_to_device(host_batch, device=device, dtype=dtype)
    device_standardizers = standardizers.to(device=device, dtype=dtype)

    torch.manual_seed(17)
    model = NodeLinearBaseline(
        in_channels=device_batch.inputs.shape[1],
        out_channels=device_batch.targets.shape[1],
        conditioning_channels=device_batch.conditioning.shape[1],
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)

    model.eval()
    scaled_batch = device_standardizers.transform(device_batch)

    def forward() -> None:
        with torch.inference_mode():
            model(
                scaled_batch.inputs,
                batch_index=scaled_batch.batch_index,
                conditioning=scaled_batch.conditioning,
            )

    forward_measurement = measure_callable(
        forward,
        device=device,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    del scaled_batch
    model.train()

    def training_iteration() -> None:
        train_equal_sample_optimizer_step(
            model,
            optimizer,
            [device_batch],
            local_sample_count=device_batch.num_graphs,
            standardizers=device_standardizers,
        )

    training_measurement = measure_callable(
        training_iteration,
        device=device,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    workload.update(
        {
            "num_graphs": device_batch.num_graphs,
            "num_nodes": int(device_batch.inputs.shape[0]),
            "num_edges": int(device_batch.edge_index.shape[1]),
            "input_channels": list(device_batch.input_channels),
            "target_channels": list(device_batch.target_channels),
            "conditioning_names": list(device_batch.conditioning_names),
        }
    )

    result = {
        "benchmark": "M7_node_linear_baseline",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "evidence": args.evidence,
        "workload": workload,
        "execution": {
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "setup_seconds_unbenchmarked": setup_seconds,
            "model": "NodeLinearBaseline",
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "optimizer": "AdamW(lr=0.0, weight_decay=0.0)",
            "statistical_scaling": standardizers.weighting,
        },
        "measurements": {
            "host_pack_and_task_prepare": _measurement_payload(
                host_prepare,
                graphs=device_batch.num_graphs,
                nodes=device_batch.inputs.shape[0],
            ),
            "forward": _measurement_payload(
                forward_measurement,
                graphs=device_batch.num_graphs,
                nodes=device_batch.inputs.shape[0],
            ),
            "training_iteration": _measurement_payload(
                training_measurement,
                graphs=device_batch.num_graphs,
                nodes=device_batch.inputs.shape[0],
            ),
        },
        "runtime_provenance": collect_runtime_provenance(repo_root),
        "scheduler_environment": _scheduler_environment(),
        "cuda_device": _cuda_device_metadata(device),
        "notes": [
            "Host HDF5 I/O and mesh-edge construction are excluded from repeated timings.",
            "The null baseline does not consume edge_index or coordinates in its forward pass.",
            "GPU timings synchronize before and after every measured call.",
            "Training uses zero-learning-rate AdamW to exercise optimizer mechanics without parameter drift.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote benchmark result to {args.output}")


def _synthetic_workload(
    args: argparse.Namespace,
) -> tuple[tuple[Sample, ...], FieldCatalog, NodeRegressionTask, dict[str, Any]]:
    if args.graphs <= 0:
        raise ValueError("--graphs must be positive")
    if args.nodes_per_graph < 3:
        raise ValueError("--nodes-per-graph must be at least 3")

    dataset = SyntheticMeshDataset(
        num_samples=args.graphs,
        min_nodes=args.nodes_per_graph,
        max_nodes=args.nodes_per_graph,
        spatial_dim=args.spatial_dim,
        seed=args.seed,
    )
    samples = tuple(dataset[index] for index in range(len(dataset)))
    task = NodeRegressionTask(input_fields=("momentum",), target_fields=("rho",))
    workload = {
        "source": "synthetic_non_physical",
        "graphs_requested": args.graphs,
        "nodes_per_graph_requested": args.nodes_per_graph,
        "spatial_dim": args.spatial_dim,
        "seed": args.seed,
        "physical_nondimensionalization": False,
    }
    return samples, dataset.field_catalog, task, workload


def _avbp_workload(
    args: argparse.Namespace,
) -> tuple[tuple[Sample, ...], FieldCatalog, NodeRegressionTask, dict[str, Any]]:
    spec = AVBPSampleSpec(
        sample_id="m7/avbp/000000",
        snapshot_file=args.snapshot,
        mesh_id=args.mesh_id,
        mesh_file=args.mesh,
        case_id=args.case_id,
    )
    dataset = AVBPHDF5Dataset(
        samples=(spec,),
        case_files={args.case_id: args.case_file},
        field_names=_AVBP_STATE_FIELDS,
        connectivity_indexing=args.connectivity_indexing,
    )
    sample = dataset[0]
    sample = replace(sample, mesh=mesh_with_hex_edge_index(sample.mesh))
    task = NodeRegressionTask(
        input_fields=_AVBP_STATE_FIELDS,
        target_fields=_AVBP_STATE_FIELDS,
        physical_nondimensionalization=True,
    )
    workload = {
        "source": "avbp_hdf5",
        "snapshot": str(args.snapshot.resolve()),
        "mesh": str(args.mesh.resolve()),
        "case_file": str(args.case_file.resolve()),
        "case_id": args.case_id,
        "mesh_id": args.mesh_id,
        "connectivity_indexing": args.connectivity_indexing,
        "physical_nondimensionalization": True,
        "benchmark_task": "five conservative state channels -> same five channels",
    }
    return (sample,), AVBP_FIELD_CATALOG, task, workload


def _task_batch_to_device(
    batch: NodeRegressionBatch,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> NodeRegressionBatch:
    source = _packed_structure_to_device(batch.source, device=device, dtype=dtype)
    return replace(
        batch,
        source=source,
        coords=batch.coords.to(device=device, dtype=dtype),
        inputs=batch.inputs.to(device=device, dtype=dtype),
        targets=batch.targets.to(device=device, dtype=dtype),
        conditioning=batch.conditioning.to(device=device, dtype=dtype),
    )


def _packed_structure_to_device(
    packed: PackedBatch,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PackedBatch:
    node_weights = packed.node_weights
    if node_weights is not None:
        node_weights = node_weights.to(device=device, dtype=dtype)
    return replace(
        packed,
        edge_index=packed.edge_index.to(device=device),
        batch_index=packed.batch_index.to(device=device),
        ptr=packed.ptr.to(device=device),
        node_weights=node_weights,
    )


def _measurement_payload(
    measurement: BenchmarkMeasurement,
    *,
    graphs: int,
    nodes: int,
) -> dict[str, Any]:
    payload = measurement.as_dict()
    median_ms = measurement.timing.median_ms
    payload["throughput"] = {
        "graphs_per_second": _items_per_second(graphs, median_ms),
        "nodes_per_second": _items_per_second(nodes, median_ms),
    }
    return payload


def _items_per_second(items: int, milliseconds: float) -> float | None:
    if milliseconds <= 0.0:
        return None
    return items * 1000.0 / milliseconds


def _scheduler_environment() -> dict[str, str]:
    keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "CUDA_VISIBLE_DEVICES",
    )
    return {key: value for key in keys if (value := os.environ.get(key)) is not None}


def _cuda_device_metadata(device: torch.device) -> dict[str, Any] | None:
    if device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device metadata requested without CUDA availability")
    properties = torch.cuda.get_device_properties(device)
    return {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multi_processor_count": properties.multi_processor_count,
    }


if __name__ == "__main__":
    main()
