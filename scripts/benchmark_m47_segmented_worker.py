"""One real-HDF5 two-graph M37 optimizer-step probe (no scientific training).

Shares the M43-B1 data reader, per-snapshot normalization, Flow Matching
task, checkpointed model and AdamW. Modes differ only in graph execution:
packed indexing, packed segmented conditioning, or sequential accumulation.
Each worker runs in a separate fresh CUDA process. Step 1 saves a diagnostic
gradient/updated-parameter audit; later steps provide unprofiled medians.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket
import time
import traceback

import torch

from graph_attention.data import AVBP_FIELD_CATALOG, AVBPHDF5Dataset, AVBPSampleSpec
from graph_attention.objectives import sample_reduced_mse
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m43_b1_training_step import (
    FIELDS, gib, git_head, gpu_memory, model_config,
    normalized_probe_state, write_report,
)

MODES = ("packed_index", "packed_segmented", "sequential_accum")


def _hash_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _model_call(model, problem, *, mode: str, graph_sizes: tuple[int, ...]):
    return model(
        problem.inputs,
        edge_index=problem.edge_index, coords=problem.coords,
        batch_index=problem.batch_index,
        conditioning=problem.conditioning,
        attention_edge_indices=problem.attention_edge_indices,
        graph_sizes=graph_sizes if mode == "packed_segmented" else None,
    )


def _single_forward(model, problem, graph, *, graph_index: int, start: int, stop: int):
    return model(
        problem.inputs[start:stop],
        edge_index=graph.edge_index, coords=graph.coords,
        batch_index=None, conditioning=problem.conditioning[graph_index:graph_index+1],
        attention_edge_indices=graph.attention_edge_indices,
    )


def run_worker(*, manifest_path: Path, mode: str, report_path: Path,
               audit_path: Path, steps: int, seed: int, backend: str) -> dict:
    if mode not in MODES or steps < 3:
        raise ValueError("invalid mode or too few steps")
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    if len(cases) != 2:
        raise ValueError("this pilot requires exactly two graphs")

    record = {
        "status": "incomplete", "mode": mode, "steps": steps, "seed": seed,
        "sparse_attention_backend": backend, "cases": cases,
        "revision": git_head(), "host": socket.gethostname(),
        "model": {"layers": 10, "hidden_dim": 128, "heads": 4,
                  "checkpointing": True, "optimizer": "AdamW(lr=1e-4)"},
        "scope": "two real HDF5 graphs; probe-only per-sample normalization; no scientific training",
        "per_step": [], "audit_file": str(audit_path),
    }
    write_report(report_path, record)
    stage = "cuda_preflight"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = torch.device("cuda:0")
        torch.manual_seed(seed)
        record["gpu"] = {
            "name": torch.cuda.get_device_name(device),
            "total_gib": gib(torch.cuda.get_device_properties(device).total_memory),
        }

        stage = "read_real_hdf5"
        specs = [
            AVBPSampleSpec(
                sample_id=f"{case['case']}:{Path(case['snapshot_file']).stem}:{i}",
                snapshot_file=Path(case["snapshot_file"]),
                mesh_id=case["case"],
                mesh_file=Path(case["mesh_file"]),
            ) for i, case in enumerate(cases)
        ]
        dataset = AVBPHDF5Dataset(samples=specs, field_names=FIELDS)
        samples = [dataset[i] for i in range(2)]
        for sample, case in zip(samples, cases, strict=True):
            if sample.mesh.num_nodes != case["nodes"]:
                raise ValueError("real mesh node count differs from preflight")

        stage = "collate_graphs"
        task = FlowMatchingTask(
            state_fields=FIELDS, physical_nondimensionalization=False,
            time_embedding_scale=1000.0,
        )
        collator = GraphTaskCollator(
            task, AVBP_FIELD_CATALOG,
            {"attention_edge_indices": {"dilated": "exact_two_hop"}},
        )
        single_host = [collator([sample]) for sample in samples]
        packed_host = collator(samples)
        sizes = tuple(int(batch.inputs.shape[0]) for batch in single_host)
        if tuple(int(v) for v in packed_host.ptr) != (0, sizes[0], sum(sizes)):
            raise RuntimeError("unexpected packed node boundaries")
        for name, field in (
            ("coords", "coords"), ("inputs", "inputs"), ("targets", "targets"),
        ):
            if not torch.equal(
                getattr(packed_host, field),
                torch.cat([getattr(batch, field) for batch in single_host], dim=0),
            ):
                raise ValueError(f"packed {name} differs from concatenated single graphs")
        record["graph"] = {
            "sizes": sizes, "nodes": sum(sizes),
            "one_hop_edges": int(packed_host.edge_index.shape[1]),
            "two_hop_edges": int(packed_host.attention_edge_indices["dilated"].shape[1]),
            "one_hop_sha256": _hash_tensor(packed_host.edge_index),
            "two_hop_sha256": _hash_tensor(packed_host.attention_edge_indices["dilated"]),
            "coords_sha256": _hash_tensor(packed_host.coords),
            "state_sha256": _hash_tensor(packed_host.inputs),
        }

        stage = "transfer_and_normalize"
        single = [
            normalized_probe_state(
                task_batch_to_device(x, device=device, dtype=torch.float32)
            ) for x in single_host
        ]
        packed = task_batch_to_device(
            packed_host, device=device, dtype=torch.float32,
        )
        packed = replace(
            packed,
            inputs=torch.cat([x.inputs for x in single]),
            targets=torch.cat([x.targets for x in single]),
            coords=torch.cat([x.coords for x in single]),
        )
        if not torch.equal(
            packed.conditioning, torch.cat([x.conditioning for x in single])
        ):
            raise ValueError("graph conditioning differs between packed and single")
        record["normalized_state_sha256"] = _hash_tensor(packed.inputs)

        stage = "initialize_m37"
        cfg = model_config(
            hidden_dim=128, num_layers=10, num_heads=4, sparse_backend=backend,
        )
        cfg.activation_checkpointing = True
        generator = torch.Generator(device=device).manual_seed(seed + 1000)
        probe = task.make_training_problem(packed, generator=generator)
        model, _ = instantiate_controlled_model(cfg, probe, seed=seed)
        model = model.to(device=device, dtype=torch.float32).train()
        del probe
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        record["parameter_count"] = sum(p.numel() for p in model.parameters())
        state_digest = hashlib.sha256()
        for param in model.parameters():
            state_digest.update(param.detach().cpu().contiguous().numpy().tobytes())
        record["initial_weights_sha256"] = state_digest.hexdigest()
        write_report(report_path, record)

        stage = "optimizer_steps"
        boundaries = (0, sizes[0], sum(sizes))
        for iteration in range(steps):
            torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            problem = task.make_training_problem(packed, generator=generator)
            torch.cuda.synchronize(device)
            begin_step = time.perf_counter()
            forward_seconds, backward_seconds = 0.0, 0.0
            if mode == "sequential_accum":
                pieces = []
                for graph_i, graph in enumerate(single):
                    start, stop = boundaries[graph_i:graph_i + 2]
                    begin_forward = time.perf_counter()
                    pred = _single_forward(
                        model, problem, graph, graph_index=graph_i,
                        start=start, stop=stop,
                    )
                    torch.cuda.synchronize(device)
                    forward_seconds += time.perf_counter() - begin_forward
                    begin_backward = time.perf_counter()
                    graph_loss = sample_reduced_mse(
                        pred, problem.targets[start:stop], graph.ptr,
                        node_weights=graph.node_weights,
                    ).mean / len(single)
                    graph_loss.backward()
                    pieces.append(graph_loss.detach())
                    torch.cuda.synchronize(device)
                    backward_seconds += time.perf_counter() - begin_backward
                loss = sum(pieces)
            else:
                begin_forward = time.perf_counter()
                pred = _model_call(model, problem, mode=mode, graph_sizes=sizes)
                torch.cuda.synchronize(device)
                forward_seconds = time.perf_counter() - begin_forward
                begin_backward = time.perf_counter()
                loss = task.training_loss(pred, problem).mean
                loss.backward()
                torch.cuda.synchronize(device)
                backward_seconds = time.perf_counter() - begin_backward
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite equal-graph Flow Matching loss")
            # The first-step audit is outside all steady-state timings.
            if iteration == 0:
                grads = {
                    name: p.grad.detach().cpu().clone() if p.grad is not None else None
                    for name, p in model.named_parameters()
                }
                record["first_step_gradient_norm"] = float(
                    torch.sqrt(sum(
                        g.double().square().sum() for g in grads.values()
                        if g is not None
                    ))
                )
            begin_optimizer = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize(device)
            optimizer_seconds = time.perf_counter() - begin_optimizer
            step_seconds = time.perf_counter() - begin_step
            if iteration == 0:
                post_params = {
                    name: p.detach().cpu().clone()
                    for name, p in model.named_parameters()
                }
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"gradients": grads, "post_update_parameters": post_params},
                    audit_path,
                )
                del grads, post_params
            reading = {
                "step": iteration + 1,
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "optimizer_seconds": optimizer_seconds,
                "step_seconds": step_seconds,
                "loss": float(loss.detach().cpu()),
                "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
            }
            record["per_step"].append(reading)
            write_report(report_path, record)
            print(f"[SEGMENTED] {mode} step={iteration+1}/{steps} "
                  f"loss={reading['loss']:.8f} time={step_seconds:.6f}s", flush=True)
            del problem, loss
            if mode != "sequential_accum":
                del pred
        record["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        record["status"] = "cuda_oom"
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
    finally:
        record["last_stage"] = stage
        write_report(report_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("torch_sparse", "torch_sparse_triton"),
                        default="torch_sparse_triton")
    args = parser.parse_args()
    result = run_worker(
        manifest_path=args.manifest, mode=args.mode,
        report_path=args.report, audit_path=args.audit,
        steps=args.steps, seed=args.seed, backend=args.backend,
    )
    print(f"[SEGMENTED] status={result['status']} mode={args.mode}", flush=True)
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
