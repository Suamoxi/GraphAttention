"""M43-B1: real M37 full-graph Flow Matching training-memory probe.

Feasibility only, NOT scientific training: per-snapshot standardization
is deliberately probe-only. No checkpoint or trained weights are saved.
Read one real AVBP HDF5 snapshot/mesh via the existing dataset and
GraphTaskCollator; build real exact-two-hop edges; perform two full
M37 10-layer torch_sparse AdamW training steps on one GPU.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time
import traceback

import torch
from omegaconf import OmegaConf

from graph_attention.data import AVBP_FIELD_CATALOG, AVBPHDF5Dataset, AVBPSampleSpec
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model


FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def gib(value: int | float) -> float:
    return float(value) / 1024**3


def peak_rss_gib() -> float:
    # ru_maxrss is kilobytes on the Linux Grace nodes.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024**2


def git_head() -> str | None:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return r.stdout.strip() if r.returncode == 0 else None


def gpu_memory() -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
        "allocated_gib": gib(torch.cuda.memory_allocated()),
        "reserved_gib": gib(torch.cuda.memory_reserved()),
        "device_free_gib": gib(free),
        "device_total_gib": gib(total),
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def normalized_probe_state(batch):
    """Probe-only standardization, NOT M6 training-split statistics."""
    values = batch.inputs
    if not bool(torch.isfinite(values).all()):
        raise ValueError("snapshot has non-finite input fields")
    mean = values.mean(dim=0, keepdim=True)
    std = values.std(dim=0, unbiased=False, keepdim=True)
    if not bool(torch.isfinite(std).all()):
        raise ValueError("snapshot contains non-finite channel dispersion")
    # Required only to prevent raw energy amplitude from destabilizing a
    # synthetic loss probe; cannot be reused for actual training.
    normalized = ((values - mean) / std.clamp_min(1.0e-6)).contiguous()
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("normalized probe state contains non-finite entries")
    extent = batch.coords.amax(dim=0) - batch.coords.amin(dim=0)
    box = extent.max()
    if not bool(torch.isfinite(box)) or float(box) <= 0:
        raise ValueError("mesh box has non-positive extent")
    return replace(
        batch, inputs=normalized, targets=normalized,
        coords=batch.coords / box,
    )


def model_config(*, hidden_dim: int = 128, num_layers: int = 10,
                 num_heads: int = 4):
    if hidden_dim <= 0 or num_layers <= 0 or num_heads <= 0:
        raise ValueError("model widths and layer count must be positive")
    if hidden_dim % num_heads:
        raise ValueError("hidden_dim must be divisible by num_heads")
    cfg = OmegaConf.load("configs/model/dinat_dit.yaml")
    cfg.hidden_dim = hidden_dim
    cfg.num_layers = num_layers
    cfg.num_heads = num_heads
    cfg.mlp_ratio = 4
    cfg.condition_embed_dim = 128
    cfg.dropout = 0.0
    cfg.qkv_bias = True
    cfg.out_proj_bias = False
    cfg.sparse_attention_backend = "torch_sparse"
    cfg.include_self_attention = False
    return cfg


def _model_forward(model: torch.nn.Module, problem):
    return model(
        problem.inputs,
        edge_index=problem.edge_index,
        coords=problem.coords,
        batch_index=problem.batch_index,
        conditioning=problem.conditioning,
        attention_edge_indices=problem.attention_edge_indices,
    )


def one_case_probe(
    *, case: str, snapshot: Path, mesh: Path, expected_side: int,
    seed: int, num_steps: int, output: Path,
    hidden_dim: int = 128, num_layers: int = 10, num_heads: int = 4,
) -> dict:
    if num_steps < 1 or expected_side < 3:
        raise ValueError("num_steps must be >=1 and expected_side >=3")
    report = {
        "status": "incomplete", "case": case,
        "snapshot_file": str(snapshot.resolve(strict=True)),
        "mesh_file": str(mesh.resolve(strict=True)),
        "expected_side": expected_side, "expected_nodes": expected_side**3,
        "seed": seed, "num_steps": num_steps,
        "model": "M37_AlternatingDilatedGeometricDiT",
        "model_config": {
            "hidden_dim": hidden_dim, "num_layers": num_layers,
            "num_heads": num_heads, "mlp_ratio": 4,
            "sparse_attention_backend": "torch_sparse",
            "include_self_attention": False,
        },
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "revision": git_head(),
        "pytorch_version": torch.__version__,
        "wall_seconds": {}, "per_step": [],
        "note": (
            "Diagnostic full M37 training steps, one snapshot, probe-only "
            "per-snapshot field z-score and spatial box normalization. "
            "Not M6 train-only scaling, not a scientific validation loss. "
            "No checkpoint is created."
        ),
    }
    write_report(output, report)
    stage = "cuda_preflight"
    start_total = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            report["status"] = "cuda_unavailable"
            return report
        device = torch.device("cuda:0")
        torch.manual_seed(seed)
        report["gpu"] = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_gib": gib(torch.cuda.get_device_properties(device).total_memory),
        }

        print("[M43-B1] Read snapshot and mesh", flush=True)
        stage = "read_hdf5_sample"
        t0 = time.perf_counter()
        dataset = AVBPHDF5Dataset(samples=[AVBPSampleSpec(
            sample_id=f"{case}:{snapshot.stem}",
            snapshot_file=snapshot,
            mesh_id=case,
            mesh_file=mesh,
        )], field_names=FIELDS)
        sample = dataset[0]
        report["wall_seconds"][stage] = time.perf_counter() - t0
        if sample.mesh.num_nodes != expected_side**3:
            raise ValueError(f"expected {expected_side**3} nodes, got {sample.mesh.num_nodes}")
        del dataset

        print("[M43-B1] Construct real local/exact-two-hop sparse graph", flush=True)
        stage = "build_cpu_graph_and_collate"
        t0 = time.perf_counter()
        task = FlowMatchingTask(
            state_fields=FIELDS, physical_nondimensionalization=False,
            time_embedding_scale=1000.0,
        )
        collator = GraphTaskCollator(
            task, AVBP_FIELD_CATALOG,
            {"attention_edge_indices": {"dilated": "exact_two_hop"}},
        )
        batch_host = collator([sample])
        report["wall_seconds"][stage] = time.perf_counter() - t0
        report["graph"] = {
            "nodes": int(batch_host.inputs.shape[0]),
            "one_hop_edges": int(batch_host.edge_index.shape[1]),
            "exact_two_hop_edges": int(
                batch_host.attention_edge_indices["dilated"].shape[1]
            ),
            "input_channels": list(batch_host.input_channels),
        }
        del sample, collator
        gc.collect()
        report["peak_cpu_rss_after_graph_gib"] = peak_rss_gib()
        write_report(output, report)

        stage = "transfer_and_scale"
        t0 = time.perf_counter()
        batch = task_batch_to_device(batch_host, device=device, dtype=torch.float32)
        batch = normalized_probe_state(batch)
        torch.cuda.synchronize(device)
        report["wall_seconds"][stage] = time.perf_counter() - t0
        report["gpu_after_transfer"] = gpu_memory()
        write_report(output, report)

        stage = "initialize_m37"
        t0 = time.perf_counter()
        cfg = model_config(
            hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
        )
        generator = torch.Generator(device=device).manual_seed(seed + 1000)
        probe = task.make_training_problem(batch, generator=generator)
        model, init = instantiate_controlled_model(cfg, probe, seed=seed)
        model = model.to(device=device, dtype=torch.float32).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        torch.cuda.synchronize(device)
        report["wall_seconds"][stage] = time.perf_counter() - t0
        report["model_parameters"] = int(sum(p.numel() for p in model.parameters()))
        report["initialization"] = init
        del probe
        write_report(output, report)

        torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(num_steps):
            print(f"[M43-B1] Training step {iteration+1}/{num_steps}", flush=True)
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            problem = task.make_training_problem(batch, generator=generator)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            stage = f"forward_{iteration+1}"
            prediction = _model_forward(model, problem)
            torch.cuda.synchronize(device)
            forward_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            stage = f"backward_{iteration+1}"
            loss = task.training_loss(prediction, problem).mean
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite Flow Matching loss")
            loss.backward()
            torch.cuda.synchronize(device)
            backward_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            stage = f"optimizer_{iteration+1}"
            optimizer.step()
            torch.cuda.synchronize(device)
            optim_seconds = time.perf_counter() - t0
            report["per_step"].append({
                "step": iteration + 1,
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "optimizer_seconds": optim_seconds,
                "step_seconds": time.perf_counter() - step_start,
                "loss_probe_only": float(loss.detach().cpu()),
                **gpu_memory(),
                "peak_cpu_rss_gib": peak_rss_gib(),
            })
            report["wall_seconds"][f"training_step_{iteration+1}"] = (
                time.perf_counter() - step_start
            )
            print(json.dumps(report["per_step"][-1]), flush=True)
            del prediction, loss, problem
            write_report(output, report)
        report["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        report["status"] = "cuda_oom"
        report["error"] = str(exc)[:2500]
        print(f"[M43-B1] CUDA OOM at {stage}: {exc}", flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-12000:]
        print(f"[M43-B1] ERROR at {stage}: {exc}", flush=True)
    finally:
        report["last_stage"] = stage
        report["total_wall_seconds"] = time.perf_counter() - start_total
        report["peak_cpu_rss_gib"] = peak_rss_gib()
        if torch.cuda.is_available():
            try:
                report["gpu_final"] = gpu_memory()
            except Exception:
                pass
        write_report(output, report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--side", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--heads", type=int, default=4)
    a = p.parse_args()
    result = one_case_probe(
        case=a.case, snapshot=a.snapshot, mesh=a.mesh,
        expected_side=a.side, num_steps=a.steps, seed=a.seed,
        hidden_dim=a.hidden_dim, num_layers=a.layers, num_heads=a.heads,
        output=a.output,
    )
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
