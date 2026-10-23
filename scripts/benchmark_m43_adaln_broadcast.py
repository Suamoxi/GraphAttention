"""Paired GH200 benchmark of ORIGINAL indexed vs NEW expanded adaLN broadcast.

Uses an unchanged M43-B checkpoint, original train-only scaler and one true
129^3 Flow Matching physical sample. All AdamW changes stay in memory;
NEVER updates last.pt, best.pt, history, standardizers or run_signature.

- CPU/GPU input and loss are identical between variants.
- The previous _broadcast_graph_features is reintroduced temporarily only
  by scoped monkeypatch for the reference mode (including checkpoint replay).
- Reload checkpoint weights and AdamW before EVERY timed step.
- Both modes get warm-ups and interleaved order to reduce thermal bias.
- Record synchronized forward/backward/optimizer time, peak allocation,
  per-parameter gradient parity and optimizer state/weight-update parity.
- Optional CUPTI reports the known indexing_backward_kernel independently.
This isolates a dominant physical-sample computation, NOT one entire
six-physical-sample M43-B optimizer update or a reproducible epoch speedup.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import time
from typing import Any, Mapping
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile

from graph_attention.data.multicase_avbp import (
    AVBPMultiCaseHITDataset, DEFAULT_M43_CASES,
)
from graph_attention.models import dit_common
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import GraphTaskCollator
from graph_attention.training.m43_logging_resume import _signature_sha256
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.train_m43_b_multires import (
    FIELDS, M37_RECIPE, _forward, _load_standardizers,
    _one_batch, build_recipe, load_training_checkpoint,
)


def _reference_broadcast(
    features: torch.Tensor, *, batch_index: torch.Tensor | None,
    num_nodes: int,
) -> torch.Tensor:
    """Exact historical pre-fast-path broadcasting source."""
    if features.ndim != 2:
        raise ValueError("graph features must have shape [B, C]")
    if batch_index is None:
        if features.shape[0] != 1:
            raise ValueError(
                "single-graph inputs require exactly one graph feature row"
            )
        return features.expand(num_nodes, -1)
    return features[batch_index]


@contextmanager
def _mode(label: str):
    if label == "indexed_reference":
        with patch.object(
            dit_common, "_broadcast_graph_features", _reference_broadcast,
        ):
            yield
    elif label == "expand_fastpath":
        yield
    else:
        raise ValueError(f"unrecognized comparison mode {label!r}")


def _synced_time(device: torch.device, action) -> float:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    action()
    torch.cuda.synchronize(device)
    return time.perf_counter() - start


def _check_input(args: argparse.Namespace) -> dict[str, Any]:
    training = args.train_run_dir.resolve()
    destination = args.output_dir.resolve()
    if destination == training or training in destination.parents:
        raise ValueError("diagnostics must not write inside the training run")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite benchmark: {destination}")
    for name in ("run_signature.json", "last.pt", "standardizers.pt"):
        if not (training / name).is_file():
            raise FileNotFoundError(training / name)
    signed = json.loads((training / "run_signature.json").read_text())
    original = signed["content"]
    if _signature_sha256(original) != signed["sha256"]:
        raise RuntimeError("M43 scientific manifest integrity mismatch")
    if original.get("mode") != "train" or original.get("model") != M37_RECIPE:
        raise ValueError("not an original M43-B M37 train signature")
    if original.get("checkpointing") is not True or original.get("accumulation") != 6:
        raise ValueError("this benchmark needs the original checkpointed M6 recipe")
    if not original.get("physical_nondimensionalization"):
        raise ValueError("original training must use physical nondimensionalization")
    if Path(original["root"]).resolve() != args.root.resolve():
        raise ValueError("DNS root differs from scientific run signature")
    refs = original["references"]
    for case in DEFAULT_M43_CASES:
        entry = refs[case.case_id]
        path = Path(entry["file"])
        if path.parent.resolve() != args.reference_dir.resolve():
            raise ValueError(f"{case.case_id}: reference directory differs")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"{case.case_id}: reference YAML checksum changed")
    sample_id = f"{args.case_id}:solut_hit_{args.snapshot_number:08d}"
    entries = [e for e in original["samples"] if e["id"] == sample_id]
    if len(entries) != 1:
        raise ValueError("requested snapshot absent from scientific manifest")
    entry = entries[0]
    snapshot = Path(entry["path"])
    mesh = Path(entry["mesh"])
    if (snapshot.stat().st_size != entry["snapshot_size"]
        or snapshot.stat().st_mtime_ns != entry["snapshot_mtime_ns"]
        or mesh.stat().st_size != entry["mesh_size"]
        or mesh.stat().st_mtime_ns != entry["mesh_mtime_ns"]):
        raise ValueError("snapshot/mesh metadata differs from original training")
    return {
        "source_signature_sha256": signed["sha256"],
        "source_training_revision": original["revision"],
        "source_snapshot_id": sample_id,
        "source_checkpoint_path": str(training / "last.pt"),
        "requested_root": str(args.root),
        "source_mesh_file": str(mesh),
        "reference_dir": str(args.reference_dir),
        "configuration": original["model"],
        "checkpointed": original["checkpointing"],
        "accumulation": original["accumulation"],
        "node_precision": "float32",
        "new_model_parameters": False,
        "checkpoint_writes": False,
    }


def _snapshot_parameters(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    optimizer_state = optimizer.state_dict()
    return {
        "weights": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "optimizer": {
            "state": {
                int(key): {
                    k: v.detach().cpu().clone() if isinstance(v, torch.Tensor)
                    else v
                    for k, v in state.items()
                }
                for key, state in optimizer_state["state"].items()
            },
            "groups": [
                {k: value for k, value in group.items() if k != "params"}
                for group in optimizer_state["param_groups"]
            ],
        },
    }


def _run_step(
    label: str, *,
    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    task: FlowMatchingTask, problem: Any,
    device: torch.device, collect_parity: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings: dict[str, float] = {}
    with _mode(label):
        prediction = None
        aggregate = None

        def forward():
            nonlocal prediction, aggregate
            prediction = _forward(model, problem)
            aggregate = task.training_loss(prediction, problem)

        timings["forward_loss_seconds"] = _synced_time(device, forward)
        assert aggregate is not None and prediction is not None
        if aggregate.sample_count != 1:
            raise RuntimeError("expected one physical snapshot")
        loss = float(aggregate.mean.detach().cpu())
        if not math.isfinite(loss):
            raise RuntimeError("non-finite original M43 Flow Matching loss")

        def backward():
            assert aggregate is not None
            aggregate.loss_sum.backward()

        timings["backward_checkpointed_seconds"] = _synced_time(device, backward)

        parity = None
        if collect_parity:
            # CPU copies are outside the timed phase and remain detached.
            parity = {
                "predicted_velocity": prediction.detach().cpu().clone(),
                "gradients": {
                    name: None if p.grad is None else p.grad.detach().cpu().clone()
                    for name, p in model.named_parameters()
                },
                "loss": loss,
            }

        # Single physical sample, not an actual M43 six-sample update:
        # numerical objective is its mean MSE (same sample_count=1).
        timings["optimizer_seconds"] = _synced_time(
            device, optimizer.step,
        )
        timings["model_step_seconds"] = sum(timings.values())
        memory = {
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / 1024**3
            ),
        }
        if collect_parity:
            assert parity is not None
            parity["post_update"] = _snapshot_parameters(model, optimizer)
    return {"loss": loss, "timings": timings, "memory": memory}, parity


def _compare_tensor(
    left: torch.Tensor, right: torch.Tensor,
) -> dict[str, float]:
    if left.shape != right.shape:
        raise RuntimeError("tensor shape differs between variants")
    difference = (right.double() - left.double())
    denom = float(torch.linalg.vector_norm(left.double()).item())
    return {
        "max_abs": float(difference.abs().max().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference).item() / max(denom, 1.e-14)
        ),
    }


def _parity(reference: dict, candidate: dict) -> dict:
    result = {
        "prediction": _compare_tensor(
            reference["predicted_velocity"], candidate["predicted_velocity"],
        ),
        "loss_abs": abs(reference["loss"] - candidate["loss"]),
        "gradients": {},
        "post_update_weights": {},
        "post_update_adamw": {},
    }
    for name, prev in reference["gradients"].items():
        current = candidate["gradients"][name]
        if (prev is None) != (current is None):
            raise RuntimeError(f"{name}: gradient presence differs")
        if prev is not None:
            result["gradients"][name] = _compare_tensor(prev, current)
    for name, prev in reference["post_update"]["weights"].items():
        result["post_update_weights"][name] = _compare_tensor(
            prev, candidate["post_update"]["weights"][name],
        )
    old_optim = reference["post_update"]["optimizer"]
    new_optim = candidate["post_update"]["optimizer"]
    if old_optim["groups"] != new_optim["groups"]:
        raise RuntimeError("AdamW hyperparameter groups differ")
    if set(old_optim["state"]) != set(new_optim["state"]):
        raise RuntimeError("AdamW parameter states differ")
    for pid, prev_state in old_optim["state"].items():
        current = new_optim["state"][pid]
        for key, value in prev_state.items():
            other = current[key]
            if isinstance(value, torch.Tensor):
                result["post_update_adamw"][f"{pid}/{key}"] = _compare_tensor(
                    value, other,
                )
            elif value != other:
                raise RuntimeError(f"AdamW state differs for {pid}/{key}")
    # Numerical equivalence, not bitwise determinism on CUDA scatter kernels.
    # Global reductions can change floating-point summation order.
    thresholds = {
        "prediction_max_abs": 1.e-5,
        "prediction_relative_l2": 1.e-5,
        "loss_abs": 1.e-6,
        "worst_gradient_relative_l2": 2.e-2,
        "worst_parameter_relative_l2": 2.e-3,
        "worst_adamw_relative_l2": 2.e-2,
    }
    metrics = {
        "prediction_max_abs": result["prediction"]["max_abs"],
        "prediction_relative_l2": result["prediction"]["relative_l2"],
        "loss_abs": result["loss_abs"],
        "worst_gradient_relative_l2": max(
            (r["relative_l2"] for r in result["gradients"].values()), default=0.
        ),
        "worst_parameter_relative_l2": max(
            (r["relative_l2"] for r in result["post_update_weights"].values()),
            default=0.,
        ),
        "worst_adamw_relative_l2": max(
            (r["relative_l2"] for r in result["post_update_adamw"].values()),
            default=0.,
        ),
    }
    result["thresholds"] = thresholds
    result["metrics"] = metrics
    result["passed"] = all(metrics[name] <= limit
                           for name, limit in thresholds.items())
    return result


def _cuda_bottleneck_profile(
    label: str, *,
    load_state, model, optimizer, task, problem, device,
) -> dict[str, Any]:
    load_state()
    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        record_shapes=False, with_stack=False,
        with_flops=False, profile_memory=False,
    ) as profiler:
        _run_step(
            label, model=model, optimizer=optimizer, task=task,
            problem=problem, device=device, collect_parity=False,
        )
    events = profiler.events()
    results: dict[str, Any] = {
        "kernel_count": 0,
        "indexing_backward_launches": 0,
        "indexing_backward_seconds": 0.0,
        "other_indexing_backward_launches": 0,
    }
    for event in events:
        for kernel in getattr(event, "kernels", ()) or ():
            results["kernel_count"] += 1
            name = str(kernel.name)
            if "indexing_backward_kernel<float, 4>" in name:
                results["indexing_backward_launches"] += 1
                results["indexing_backward_seconds"] += float(kernel.duration) / 1e6
            elif "indexing_backward_kernel" in name:
                results["other_indexing_backward_launches"] += 1
    return results


def benchmark(args: argparse.Namespace) -> dict:
    provenance = _check_input(args)
    if not torch.cuda.is_available():
        raise RuntimeError("requires an NVIDIA Grace/GH200 GPU")
    if os.getenv("CUDA_LAUNCH_BLOCKING") == "1":
        raise RuntimeError("CUDA_LAUNCH_BLOCKING=1 would perturb GPU timing")
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    original_checkpoint = args.train_run_dir / "last.pt"
    checkpoint_hash_before = hashlib.sha256(
        original_checkpoint.read_bytes()
    ).hexdigest()
    device = torch.device("cuda:0")
    original = json.loads(
        (args.train_run_dir / "run_signature.json").read_text()
    )["content"]
    torch.manual_seed(int(original["seed"]))

    dataset = AVBPMultiCaseHITDataset(
        root=args.root, cases=DEFAULT_M43_CASES,
        reference_dir=args.reference_dir,
        require_references=True, expected_snapshots_per_case=100,
    )
    sample_id = provenance["source_snapshot_id"]
    matching = [i for i,id_ in enumerate(dataset.sample_ids) if id_ == sample_id]
    if len(matching) != 1:
        raise RuntimeError("source snapshot not uniquely identified in dataset")
    index = matching[0]
    standardizers = _load_standardizers(
        args.train_run_dir / "standardizers.pt",
    )
    if not standardizers.physical_nondimensionalization:
        raise RuntimeError("requires original physical nondimensionalization")
    task = FlowMatchingTask(
        state_fields=FIELDS, physical_nondimensionalization=True,
        validation_seed=1234, time_embedding_scale=1000.,
    )
    collator = GraphTaskCollator(
        task, dataset.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    device_standardizers = standardizers.to(
        device=device, dtype=torch.float32,
    )
    prepared = _one_batch(
        dataset,index,collator=collator,
        standardizers=device_standardizers, device=device,
    )
    generator = torch.Generator(device=device).manual_seed(
        int(original["seed"]) + 1000
    )
    # A probe is needed only for declared input/output/conditioning shapes.
    probe = task.make_model_probe(
        task.make_training_problem(prepared,generator=generator)
    )
    model, _ = instantiate_controlled_model(
        build_recipe(checkpointing=True),
        probe, seed=int(original["seed"]),
    )
    model = model.to(device=device,dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1.e-4,weight_decay=0.)
    source_signature_sha256 = provenance["source_signature_sha256"]

    def load_original_checkpoint():
        loaded = load_training_checkpoint(
            args.train_run_dir / "last.pt",
            signature_sha256=source_signature_sha256,
            model=model,optimizer=optimizer,generator=generator,
        )
        return loaded

    checkpoint = load_original_checkpoint()
    problem = task.make_training_problem(prepared,generator=generator)
    del probe
    provenance.update({
        "checkpoint_epoch_zero_based": int(checkpoint["epoch"]),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "model_class": type(model).__name__,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "python": platform.python_version(),
        "host": socket.gethostname(),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True,
            text=True,check=False,
        ).stdout.strip(),
        "num_nodes": int(problem.inputs.shape[0]),
        "onehop_edges": int(problem.edge_index.shape[1]),
        "twohop_edges": int(
            problem.attention_edge_indices["dilated"].shape[1]
        ),
        "task_conditioning": problem.conditioning.detach().cpu().tolist(),
        "benchmark_trials": args.trials,
        "kernel_profile": args.profile_kernels,
        "hardware_precision": "float32",
        "optimizer_step_note": "isolated ONE physical sample; not original M6 step",
        "timing_note": "excludes sample IO/HDF5, collate, transfer and initialization",
    })
    (destination / "provenance.json").write_text(
        json.dumps(provenance,indent=2) + "\n"
    )
    labels = ("indexed_reference","expand_fastpath")
    for label in labels:
        load_original_checkpoint()
        print(f"[M43 fastpath] warmup {label}",flush=True)
        _run_step(
            label, model=model,optimizer=optimizer,task=task,
            problem=problem,device=device,collect_parity=False,
        )

    # Interleaved order; each trial begins at the same exact saved checkpoint.
    order = (["indexed_reference", "expand_fastpath"]
             if args.trials == 1 else
             [label for i in range(args.trials)
              for label in (labels if i % 2 == 0 else labels[::-1])])
    measurements: dict[str,list[dict[str, Any]]] = {
        label: [] for label in labels
    }
    parity_inputs: dict[str,dict] = {}
    for iteration,label in enumerate(order):
        load_original_checkpoint()
        print(f"[M43 fastpath] timed step {iteration+1}/{len(order)} "
              f"mode={label}",flush=True)
        record, capture = _run_step(
            label, model=model,optimizer=optimizer,task=task,
            problem=problem,device=device,
            collect_parity=(label not in parity_inputs),
        )
        measurements[label].append(record)
        if capture is not None:
            parity_inputs[label] = capture
        print(f"[M43 fastpath] {label}: "
              f"forward={record['timings']['forward_loss_seconds']:.4f}s "
              f"backward={record['timings']['backward_checkpointed_seconds']:.4f}s "
              f"peak={record['memory']['peak_allocated_gib']:.3f} GiB",
              flush=True)

    parity = _parity(
        parity_inputs["indexed_reference"],parity_inputs["expand_fastpath"],
    )
    medians = {}
    for label, records in measurements.items():
        medians[label] = {
            "timings": {
                key: float(statistics.median(
                    record["timings"][key] for record in records
                )) for key in records[0]["timings"]
            },
            "peak_allocated_gib": float(statistics.median(
                record["memory"]["peak_allocated_gib"] for record in records
            )),
            "peak_reserved_gib": float(statistics.median(
                record["memory"]["peak_reserved_gib"] for record in records
            )),
            "trial_records": records,
        }
    old = medians["indexed_reference"]
    new = medians["expand_fastpath"]
    speedups = {
        key: old["timings"][key]/new["timings"][key]
        for key in old["timings"]
        if new["timings"][key] > 0
    }
    kernel_profile = {}
    if args.profile_kernels:
        for label in labels:
            print(f"[M43 fastpath] CUPTI {label}",flush=True)
            kernel_profile[label] = _cuda_bottleneck_profile(
                label,load_state=load_original_checkpoint,
                model=model,optimizer=optimizer,task=task,
                problem=problem,device=device,
            )
    checkpoint_hash_after = hashlib.sha256(
        original_checkpoint.read_bytes()
    ).hexdigest()
    provenance["checkpoint_sha256_after"] = checkpoint_hash_after
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError(
            "source checkpoint changed while profiling; "
            "cannot trust paired measurements"
        )
    result = {
        "status": "passed" if parity["passed"] else "numerical_mismatch",
        "parity": parity,
        "timing": medians,
        "speedup_indexed_over_expand": speedups,
        "indexing_backward_kernel_profile": kernel_profile,
        "provenance": provenance,
        "notes": [
            "No scientific checkpoint, scaler, dataset, split or training history "
            "has been modified; all optimizer updates are discarded.",
            "Existing M43-B uses six sequential physical microbatches per optimizer "
            "step; this compares just the dominant 129^3 substep.",
            "The original broadcast implementation was scoped-mocked while "
            "profiling, including backward activation-checkpoint replay.",
            "Forward, parameter gradient and AdamW state parity are compared "
            "from identical checkpoint weights/optimizer and the same sampled "
            "Flow Matching noise/time.",
            "Do not resume scientific training without completing numerical and "
            "performance gates and an explicit checkpoint migration review.",
        ],
    }
    (destination / "benchmark_summary.json").write_text(
        json.dumps(result,indent=2) + "\n"
    )
    print(json.dumps({
        "status":result["status"],"parity_metrics":parity["metrics"],
        "median_step_speedup":speedups,
        "kernel_profile":kernel_profile,
        "output":str(destination),
    },indent=2),flush=True)
    if not parity["passed"]:
        raise RuntimeError(
            "full-checkpoint gradient/optimizer equivalence gate failed"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run-dir",type=Path,default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m43_b_multires/train_pilot_s42"
    ))
    parser.add_argument("--root",type=Path,default=Path(
        "/archivenfs/phlusim/HIT_DNS_COMP"
    ))
    parser.add_argument("--reference-dir",type=Path,default=Path(
        "/scratch/coop/theret/HIT_DNS_COMP_case_references"
    ))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--case-id",default="Re94/Ma0.2",
                        choices=tuple(
                            c.case_id for c in DEFAULT_M43_CASES if c.side==129
                        ))
    parser.add_argument("--snapshot-number",type=int,default=51)
    parser.add_argument("--trials",type=int,default=3)
    parser.add_argument(
        "--profile-kernels",action=argparse.BooleanOptionalAction,default=True,
    )
    args=parser.parse_args()
    for key in ("train_run_dir","root","reference_dir","output_dir"):
        setattr(args,key,getattr(args,key).expanduser().resolve())
    if args.trials<1 or args.trials>8:
        raise ValueError("trials must be between 1 and 8")
    if args.snapshot_number<1:
        raise ValueError("snapshot number must be positive")
    return args


if __name__=="__main__":
    benchmark(parse_args())
