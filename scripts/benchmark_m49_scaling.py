"""M49 isolated synthetic Cartesian compute scaling; no scientific training."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from graph_attention.geometry.cartesian_halo import partition_cartesian_edges
from graph_attention.models import AlternatingDilatedGeometricDiT
from graph_attention.models.full_dit import FullDiTGraphTransformer
from graph_attention.utils.provenance import collect_runtime_provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("full", "dinat"), required=True)
    p.add_argument("--side", type=int, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--backend", choices=("sdpa", "torch_sparse", "torch_sparse_triton"), required=True)
    p.add_argument("--checkpointing", action="store_true")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.side, args.batch, args.warmup, args.repeats) < 1 or args.side < 3:
        p.error("invalid workload")
    if (args.model == "full") != (args.backend == "sdpa"):
        p.error("backend incompatible with model")
    if args.model == "full" and args.checkpointing:
        p.error("checkpointing is available only for DiNAT")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"evidence": "TARGET_VALIDATED", "workload": "synthetic_nonperiodic_cartesian",
              **vars(args), "output": str(args.output),
              "provenance": collect_runtime_provenance(), "status": "started"}
    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output)
    save()
    stage = "setup"
    try:
        device = torch.device("cuda")
        torch.manual_seed(42)
        n = args.side ** 3
        b = args.batch
        axis = torch.linspace(0, 1, args.side)
        coords = torch.cartesian_prod(axis, axis, axis).repeat(b, 1).to(device)
        edges = torch.empty((2, 0), dtype=torch.long, device=device)
        extra = {}
        if args.model == "dinat":
            topo = partition_cartesian_edges(np.arange(n, dtype=np.int64), side=args.side,
                       lower=(0, 0, 0), upper=(args.side,) * 3)
            def packed(array):
                base = torch.from_numpy(array).to(device)
                return torch.cat([base + i*n for i in range(b)], dim=1)
            edges = packed(topo.local_edge_index)
            extra["attention_edge_indices"] = {"dilated": packed(topo.dilated_edge_index)}
            del topo
            if args.backend == "torch_sparse_triton":
                extra["graph_sizes"] = (n,) * b
        common = dict(in_channels=5, out_channels=5, hidden_dim=128, num_heads=4,
                      num_layers=10, spatial_dim=3, mlp_ratio=4, conditioning_channels=1,
                      condition_embed_dim=128, dropout=0.0)
        if args.model == "full":
            model = FullDiTGraphTransformer(**common, use_sdpa=True)
        else:
            model = AlternatingDilatedGeometricDiT(**common, sparse_attention_backend=args.backend,
                         activation_checkpointing=args.checkpointing)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(b*n, 5, device=device)
        target = torch.randn_like(x)
        index = torch.arange(b, device=device).repeat_interleave(n)
        condition = torch.full((b, 1), 500., device=device)
        kwargs = dict(coords=coords, edge_index=edges, batch_index=index, conditioning=condition, **extra)
        report.update(nodes_per_graph=n, total_nodes=b*n, local_edges=edges.shape[1],
                      dilated_edges=extra.get("attention_edge_indices", {}).get("dilated", edges).shape[1],
                      parameters=sum(v.numel() for v in model.parameters()), dtype="float32")
        def measure(fn):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            return time.perf_counter() - start
        def forward():
            with torch.inference_mode():
                model(x, **kwargs)
        def train():
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x, **kwargs) - target).square().mean()
            loss.backward()
            optimizer.step()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite loss")
        for name, fn in (("forward", forward), ("train", train)):
            stage = name
            model.train(name == "train")
            for _ in range(args.warmup):
                measure(fn)
            torch.cuda.reset_peak_memory_stats()
            times = [measure(fn) for _ in range(args.repeats)]
            med = statistics.median(times)
            report[name] = dict(seconds=times, median_seconds=med,
                      nodes_per_second=b*n/med, graphs_per_second=b/med,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            save()
        report["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        report.update(status="cuda_oom", stage=stage, error=str(exc))
    except Exception as exc:
        report.update(status="failed", stage=stage, error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
