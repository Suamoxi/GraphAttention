"""Paired FP32 CSR attention benchmark: original chunked vs Triton backward.

Synthetic Cartesian 65^3/129^3 connectivity has the same six one-hop and
eighteen exact-two-hop interior neighbours as the M47 graph. Actual HDF5
ordering/mesh construction is intentionally excluded from the GPU timings.
Unit tests separately cover arbitrary unstructured directed connectivity.

All timing measures exclude graph/CSR preparation and Triton compilation.
Profiler traces, if requested, are taken separately after unprofiled runs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import torch

from graph_attention.models.dinat_dit import DiNATDiTMultiheadAttention
from graph_attention.models.sparse_transformer import (
    _TorchSparseSDDMM, _TorchSparseSpMM, _build_torch_sparse_topology,
)
from graph_attention.models.triton_sparse_backward import (
    TritonSparseSDDMM, TritonSparseSpMM, require_triton,
)


def offsets(two_hop: bool) -> tuple[tuple[int, int, int], ...]:
    # Exact two hops on the interior of a simple 6-neighbour Cartesian graph.
    expected = 2 if two_hop else 1
    return tuple(
        (dx, dy, dz)
        for dx in range(-2, 3)
        for dy in range(-2, 3)
        for dz in range(-2, 3)
        if abs(dx) + abs(dy) + abs(dz) == expected
    )


def cartesian_edges(side: int, *, two_hop: bool, device: torch.device) -> torch.Tensor:
    if side < 5:
        raise ValueError("side must be >= 5")
    lattice = torch.arange(side**3, dtype=torch.long, device=device).reshape(
        side, side, side,
    )
    src_parts = []
    dst_parts = []
    for dx, dy, dz in offsets(two_hop):
        sx = slice(max(0, dx), min(side, side + dx))
        sy = slice(max(0, dy), min(side, side + dy))
        sz = slice(max(0, dz), min(side, side + dz))
        tx = slice(max(0, -dx), min(side, side - dx))
        ty = slice(max(0, -dy), min(side, side - dy))
        tz = slice(max(0, -dz), min(side, side - dz))
        src_parts.append(lattice[sx, sy, sz].reshape(-1))
        dst_parts.append(lattice[tx, ty, tz].reshape(-1))
    source = torch.cat(src_parts)
    target = torch.cat(dst_parts)
    return torch.stack((source, target), dim=0)


def operation_factory(
    operation: str, backend: str, *, nodes: int, edges: torch.Tensor,
    topo: Any, device: torch.device, generator_seed: int,
):
    torch.manual_seed(generator_seed)
    heads, dim = 4, 32
    if operation == "sddmm":
        q = torch.randn(nodes, heads, dim, device=device, requires_grad=True)
        k = torch.randn(nodes, heads, dim, device=device, requires_grad=True)

        def once():
            if backend == "torch_sparse":
                scores = _TorchSparseSDDMM.apply(
                    q, k, topo.crow_indices, topo.row_indices, topo.col_indices,
                    dim**-0.5,
                )
            else:
                scores = TritonSparseSDDMM.apply(q, k, topo, dim**-0.5)
            return scores.square().mean()

        def reset():
            q.grad = None
            k.grad = None

        return once, reset

    if operation == "spmm":
        w = torch.randn(edges.shape[1], heads, device=device, requires_grad=True)
        v = torch.randn(nodes, heads, dim, device=device, requires_grad=True)

        def once():
            if backend == "torch_sparse":
                output = _TorchSparseSpMM.apply(
                    w, v, topo.row_indices, topo.col_indices,
                )
            else:
                output = TritonSparseSpMM.apply(w, v, topo)
            return output.square().mean()

        def reset():
            w.grad = None
            v.grad = None

        return once, reset

    if operation == "geometric_attention":
        model = DiNATDiTMultiheadAttention(
            heads * dim, heads, 3, sparse_attention_backend=backend,
        ).to(device)
        model.train()
        x = torch.randn(nodes, heads * dim, device=device, requires_grad=True)
        # Relative-displacement input; the geometry MLP remains trainable.
        displacement = torch.randn(
            edges.shape[1], 3, device=device, dtype=torch.float32,
        )

        def once():
            output = model(
                x, edge_index=edges, batch_index=None,
                edge_displacement=displacement, sparse_adj=topo,
            )
            return output.square().mean()

        def reset():
            model.zero_grad(set_to_none=True)
            x.grad = None

        return once, reset

    raise ValueError("unsupported benchmark operation")


def run_timed(callback, reset, *, device: torch.device,
              warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        reset()
        loss = callback()
        loss.backward()
        torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    readings = []
    for _ in range(repeats):
        reset()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss = callback()
        torch.cuda.synchronize(device)
        forward = time.perf_counter() - start
        loss.backward()
        torch.cuda.synchronize(device)
        total = time.perf_counter() - start
        readings.append({
            "forward_seconds": forward,
            "backward_seconds": total - forward,
            "total_seconds": total,
        })
    return {
        "repetitions": readings,
        "median_forward_seconds": statistics.median(x["forward_seconds"] for x in readings),
        "median_backward_seconds": statistics.median(x["backward_seconds"] for x in readings),
        "median_total_seconds": statistics.median(x["total_seconds"] for x in readings),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def profile_backward(callback, reset, *, device: torch.device, folder: Path,
                     name: str) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    reset()
    loss = callback()
    loss.backward()
    torch.cuda.synchronize(device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=False,
        with_stack=False,
    ) as profiler:
        reset()
        loss = callback()
        loss.backward()
        torch.cuda.synchronize(device)
    profile_file = folder / f"{name}_trace.json"
    profiler.export_chrome_trace(str(profile_file))
    ops = sorted(
        profiler.key_averages(),
        key=lambda x: -float(getattr(
            x, "self_device_time_total", getattr(x, "self_cuda_time_total", 0.0),
        )),
    )
    return {
        "chrome_trace": str(profile_file),
        "top_gpu_ops": [
            {"name": op.key, "calls": int(op.count),
             "self_gpu_ms": float(getattr(
                 op, "self_device_time_total", getattr(op, "self_cuda_time_total", 0.0),
             )) / 1000.0}
            for op in ops[:20]
        ],
        "warning": "Profiler timings are not comparable to unprofiled wall times",
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    require_triton()
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError("benchmark requires >=1 warmup and >=2 repetitions")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    edge_index = cartesian_edges(
        args.side, two_hop=args.topology == "two_hop", device=device,
    )
    topology = _build_torch_sparse_topology(
        edge_index, num_nodes=args.side**3,
    )
    num_edges = edge_index.shape[1]
    operations = (
        ("sddmm", "spmm", "geometric_attention")
        if args.operation == "all" else (args.operation,)
    )

    args.output_dir.mkdir(parents=True)
    results = {}
    for op in operations:
        per_mode = {}
        for backend in ("torch_sparse", "torch_sparse_triton"):
            once, reset = operation_factory(
                op, backend, nodes=args.side**3, edges=edge_index,
                topo=topology, device=device, generator_seed=args.seed,
            )
            timing = run_timed(
                once, reset, device=device, warmup=args.warmup, repeats=args.repeats,
            )
            if args.profile:
                timing["profile"] = profile_backward(
                    once, reset, device=device, folder=args.output_dir,
                    name=f"{op}_{backend}",
                )
            per_mode[backend] = timing
            print(
                f"[CSR BACKWARD] side={args.side} {args.topology} {op} {backend} "
                f"fwd={timing['median_forward_seconds']:.5f} "
                f"bwd={timing['median_backward_seconds']:.5f} "
                f"peak={timing['peak_gpu_allocated_gib']:.3f} GiB",
                flush=True,
            )
            del once, reset
        baseline = per_mode["torch_sparse"]["median_backward_seconds"]
        candidate = per_mode["torch_sparse_triton"]["median_backward_seconds"]
        per_mode["backward_speedup"] = baseline / candidate if candidate else None
        results[op] = per_mode
    result = {
        "protocol": "m48_triton_sparse_backward_pilot_v1",
        "status": "completed_not_accepted",
        "scientific_scope": "synthetic connectivity; kernel benchmark, no training",
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True,
        ).stdout.strip(),
        "side": args.side, "num_nodes": args.side**3,
        "topology": args.topology, "num_edges": num_edges,
        "head_count": 4, "head_dim": 32, "dtype": "float32",
        "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
        "device": torch.cuda.get_device_name(device),
        "note": (
            "Compare only unprofiled median times; kernel launch and CUDA execution "
            "scheduling may depend on machine load. Run parity tests separately."
        ),
        "results": results,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, choices=(65, 129), required=True)
    parser.add_argument("--topology", choices=("one_hop", "two_hop"), required=True)
    parser.add_argument(
        "--operation", choices=("sddmm", "spmm", "geometric_attention", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--profile", action="store_true")
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
