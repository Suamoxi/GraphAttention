"""Benchmark the M8 sparse one-hop transformer with M7-compatible provenance."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from benchmark_m7 import (
    _DTYPES,
    _EVIDENCE_LEVELS,
    _avbp_workload,
    _cuda_device_metadata,
    _scheduler_environment,
    _synthetic_workload,
    _task_batch_to_device,
    _validate_evidence_request,
)
from graph_attention.data import SplitManifest
from graph_attention.models import SparseGraphTransformer
from graph_attention.training import fit_train_standardizers, train_equal_sample_optimizer_step
from graph_attention.utils.benchmarking import BenchmarkMeasurement, measure_callable
from graph_attention.utils.provenance import collect_runtime_provenance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="PyTorch device, e.g. cuda or cpu")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--evidence", choices=_EVIDENCE_LEVELS, default="LOCAL_BENCHMARK")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)

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
    runtime_provenance = collect_runtime_provenance(repo_root)
    _validate_evidence_request(args.evidence, device, runtime_provenance)

    if args.workload == "synthetic":
        samples, catalog, task, workload = _synthetic_workload(args)
    else:
        samples, catalog, task, workload = _avbp_workload(args)

    split = SplitManifest(train_ids=tuple(sample.sample_id for sample in samples))
    standardizers = fit_train_standardizers(task, samples, catalog, split)
    host_batch = task.pack_and_prepare(samples, catalog)
    device_batch = _task_batch_to_device(host_batch, device=device, dtype=dtype)
    device_standardizers = standardizers.to(device=device, dtype=dtype)

    torch.manual_seed(17)
    model = SparseGraphTransformer(
        in_channels=device_batch.inputs.shape[1],
        out_channels=device_batch.targets.shape[1],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        conditioning_channels=device_batch.conditioning.shape[1],
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)

    scaled_batch = device_standardizers.transform(device_batch)
    model.eval()

    def forward() -> None:
        with torch.inference_mode():
            model(
                scaled_batch.inputs,
                edge_index=scaled_batch.edge_index,
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

    num_nodes = int(device_batch.inputs.shape[0])
    num_edges = int(device_batch.edge_index.shape[1])
    workload.update(
        {
            "num_graphs": device_batch.num_graphs,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "input_channels": list(device_batch.input_channels),
            "target_channels": list(device_batch.target_channels),
            "conditioning_names": list(device_batch.conditioning_names),
        }
    )

    result = {
        "benchmark": "M8_sparse_graph_transformer",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "evidence": args.evidence,
        "workload": workload,
        "execution": {
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "model": "SparseGraphTransformer",
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "mlp_ratio": args.mlp_ratio,
            "optimizer": "AdamW(lr=0.0, weight_decay=0.0)",
            "statistical_scaling": standardizers.weighting,
            "attention_topology": "supplied_directed_one_hop_edges",
            "uses_coordinates": False,
            "adds_self_loops": False,
        },
        "measurements": {
            "forward": _measurement_payload(
                forward_measurement,
                graphs=device_batch.num_graphs,
                nodes=num_nodes,
                edges=num_edges,
            ),
            "training_iteration": _measurement_payload(
                training_measurement,
                graphs=device_batch.num_graphs,
                nodes=num_nodes,
                edges=num_edges,
            ),
        },
        "runtime_provenance": runtime_provenance,
        "scheduler_environment": _scheduler_environment(),
        "cuda_device": _cuda_device_metadata(device),
        "notes": [
            "M8 consumes edge_index in every sparse-attention layer.",
            "Coordinates and geometric edge features are intentionally absent until M9.",
            "HDF5 I/O, one-time edge construction, and host-to-device transfer are excluded.",
            "GPU timings synchronize before and after every measured call.",
            "Training uses zero-learning-rate AdamW to avoid benchmark parameter drift.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote benchmark result to {args.output}")


def _measurement_payload(
    measurement: BenchmarkMeasurement,
    *,
    graphs: int,
    nodes: int,
    edges: int,
) -> dict[str, Any]:
    payload = measurement.as_dict()
    median_ms = measurement.timing.median_ms
    payload["throughput"] = {
        "graphs_per_second": _items_per_second(graphs, median_ms),
        "nodes_per_second": _items_per_second(nodes, median_ms),
        "edges_per_second": _items_per_second(edges, median_ms),
    }
    return payload


def _items_per_second(items: int, milliseconds: float) -> float | None:
    if milliseconds <= 0.0:
        return None
    return items * 1000.0 / milliseconds


if __name__ == "__main__":
    main()
