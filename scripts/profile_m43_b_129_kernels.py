"""Profile the UNMODIFIED M37/M43-B 129^3 FP32 training kernels.

This is a read-only experiment using existing M43-B inputs, normalization,
M37 architecture and checkpoint weights/AdamW state. In-memory optimization
steps are discarded: NO checkpoint, dataset, history or signature is written.

One warm step primes CUDA kernels/topology, then reloading the checkpoint and
generator restores the exact starting model/optimizer state. One subsequent
full snapshot step is profiled, including its original non-reentrant block
checkpoint recomputation and the sampled-addmm / scatter / index_add kernels.

This does not claim to benchmark the entire SIX-physical-sample M43 optimizer
step. It isolates its dominant 129^3 single-snapshot forward/backward phase.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from graph_attention.data.multicase_avbp import (
    AVBPMultiCaseHITDataset,
    DEFAULT_M43_CASES,
)
from graph_attention.training import m43_kernel_profile_report as report_module
from graph_attention.training.data_pipeline import GraphTaskCollator
from graph_attention.training.model_factory import instantiate_controlled_model
from graph_attention.tasks import FlowMatchingTask
from scripts.train_m43_b_multires import (
    FIELDS,
    M37_RECIPE,
    _forward,
    _load_standardizers,
    _one_batch,
    build_recipe,
    load_training_checkpoint,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    # The profiling output directory must NOT coincide with the training run.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _sha256_signature(directory: Path) -> dict[str, Any]:
    signature_file = directory / "run_signature.json"
    checkpoint = directory / "last.pt"
    scaler = directory / "standardizers.pt"
    if not all(path.is_file() for path in (signature_file, checkpoint, scaler)):
        raise FileNotFoundError(
            "M43-B profile requires existing run_signature.json, last.pt, "
            "and standardizers.pt in --train-run-dir"
        )
    signature = json.loads(signature_file.read_text())
    if not isinstance(signature, dict) or not isinstance(signature.get("content"), dict):
        raise ValueError("malformed M43-B scientific run signature")
    if signature["content"].get("mode") != "train":
        raise ValueError("profile input is not an M43-B scientific training run")
    if signature["content"].get("model") != M37_RECIPE:
        raise ValueError("M43-B model recipe differs from M37 profiling expectation")
    from graph_attention.training.m43_logging_resume import _signature_sha256
    if _signature_sha256(signature["content"]) != signature.get("sha256"):
        raise ValueError("scientific run signature SHA256 mismatch")
    return signature


def _git_rev() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False, text=True, capture_output=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _chosen_index(
    dataset: AVBPMultiCaseHITDataset, *,
    case_id: str, snapshot_number: int,
) -> int:
    """Choose by actual basename, not positional order or test/train split."""
    if not isinstance(snapshot_number, int) or snapshot_number < 1:
        raise ValueError("snapshot number must be a positive integer")
    if case_id not in dataset.indices_by_case:
        raise ValueError(f"unknown case_id {case_id}")
    spec = next((case for case in dataset.cases if case.case_id == case_id), None)
    if spec is None or spec.side != 129:
        raise ValueError("M43-B kernel profiling requires a physical 129^3 case")
    expected = f"solut_hit_{snapshot_number:08d}"
    matching = [
        index for index in dataset.indices_by_case[case_id]
        if dataset.sample_specs[index].snapshot_file.stem == expected
    ]
    if len(matching) != 1:
        raise ValueError(f"{case_id}: required snapshot {expected} not uniquely found")
    return matching[0]


def _validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.resolve() == args.train_run_dir.resolve():
        raise ValueError("profiling output must not overwrite the training directory")
    if args.output_dir.resolve().is_relative_to(args.train_run_dir.resolve()):
        raise ValueError("keep diagnostic output completely separate from training")
    if args.snapshot_number < 1:
        raise ValueError("snapshot_number must be positive")
    if args.mode not in {"inspect", "profile"}:
        raise ValueError("unknown profiling mode")


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    signature = _sha256_signature(args.train_run_dir)
    state = signature["content"]
    if Path(state["root"]).resolve() != args.root.resolve():
        raise ValueError(
            "requested DNS archive path disagrees with the scientific run signature"
        )
    if state.get("accumulation") != 6 or not state.get("checkpointing"):
        raise ValueError("expected checkpointed M37 with six-sample accumulation")
    if args.reference_dir is not None:
        expected_files = {
            Path(record["file"]).parent for record in state.get("references", {}).values()
        }
        if expected_files and (expected_files != {args.reference_dir.resolve()}):
            raise ValueError(
                "reference directory differs from paths recorded in original run"
            )
    index = {"checkpoint_epoch": "load only in profile mode"}
    return {
        "status": "read_only_profile_inspected",
        "train_run_dir": str(args.train_run_dir),
        "source_signature_sha256": signature["sha256"],
        "source_training_revision": state["revision"],
        "executing_revision": _git_rev(),
        "root": str(args.root),
        "reference_dir": str(args.reference_dir),
        "case_id": args.case_id,
        "snapshot_basename": f"solut_hit_{args.snapshot_number:08d}.h5",
        "scientific_parameters": {
            "model": state["model"],
            "accumulation": state["accumulation"],
            "physical_nondimensionalization": state["physical_nondimensionalization"],
            "checkpointing": state["checkpointing"],
            "time_embedding_scale": state["time_embedding_scale"],
            "seed": state["seed"],
            "learning_rate": "restored from original optimizer state",
        },
        "checkpoint_provenance": index,
        "mutates_training_run": False,
        "per_step_comparison": "one 129^3 physical sample, not all 6 accumulated cases",
    }


def _cuda_sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _cuda_stats(device: torch.device) -> dict[str, float]:
    return {
        "memory_allocated_gib": torch.cuda.memory_allocated(device) / (1024 ** 3),
        "memory_reserved_gib": torch.cuda.memory_reserved(device) / (1024 ** 3),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
    }


def _one_real_step(
    *,
    dataset: AVBPMultiCaseHITDataset,
    index: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    task: FlowMatchingTask,
    standardizers: Any,
    collator: GraphTaskCollator,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    """Exactly the scientific 129^3 microbatch path, one AdamW update in memory.

    Original M43-B accumulates SIX such per-physical-sample loss sums, then
    divides gradients by six. Here exactly one sample is used, so the
    per-sample numerical graph and its gradients are unchanged; AdamW follows
    an ordinary standalone one-sample update purely for profiling.
    """
    model.train()
    times: dict[str, float] = {}
    _cuda_sync(device)

    with record_function("m43_profile/load_collate_transfer_noise"):
        t0 = time.perf_counter()
        prepared = _one_batch(
            dataset, index, collator=collator,
            standardizers=standardizers, device=device,
        )
        problem = task.make_training_problem(prepared, generator=generator)
        _cuda_sync(device)
        times["load_collate_transfer_noise_seconds"] = time.perf_counter() - t0

    optimizer.zero_grad(set_to_none=True)
    with record_function("m43_profile/model_forward_loss"):
        t0 = time.perf_counter()
        aggregate = task.training_loss(_forward(model, problem), problem)
        # Match original M43 scalar collection, including its synchronization.
        scalar = float(aggregate.mean.detach().cpu())
        if aggregate.sample_count != 1 or not (0 <= scalar < float("inf")):
            raise ValueError("invalid physical-sample Flow Matching MSE")
        _cuda_sync(device)
        times["model_forward_loss_seconds"] = time.perf_counter() - t0

    with record_function("m43_profile/backward_including_checkpoint_recompute"):
        t0 = time.perf_counter()
        aggregate.loss_sum.backward()
        _cuda_sync(device)
        times["backward_including_checkpoint_recompute_seconds"] = (
            time.perf_counter() - t0
        )

    with record_function("m43_profile/adamw_optimizer"):
        t0 = time.perf_counter()
        for parameter in model.parameters():
            if parameter.grad is not None:
                # M43 exact M6 normalization: 1 / samples in optimizer window.
                # Here the profiling window contains ONE physical sample.
                parameter.grad.mul_(1.0)
        optimizer.step()
        _cuda_sync(device)
        times["adamw_optimizer_seconds"] = time.perf_counter() - t0

    times["phase_total_seconds"] = sum(times.values())
    return {
        "times": times,
        "loss_velocity_mse": scalar,
        "optimizer_update_written_to_disk": False,
        "num_physical_samples_in_profiled_step": 1,
        "memory": _cuda_stats(device),
    }


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    metadata = inspect(args)
    if not torch.cuda.is_available():
        raise RuntimeError("M43-B full 129^3 profiler requires NVIDIA CUDA")
    if "CUDA_LAUNCH_BLOCKING" in os.environ and os.environ["CUDA_LAUNCH_BLOCKING"] == "1":
        raise RuntimeError(
            "CUDA_LAUNCH_BLOCKING=1 changes the normal workload timings; unset it"
        )
    if args.output_dir.exists():
        raise FileExistsError(
            "profile results already exist; choose a fresh --output-dir"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_dir / "input_provenance.json", metadata)

    state = _sha256_signature(args.train_run_dir)["content"]
    torch.manual_seed(int(state["seed"]))
    device = torch.device("cuda:0")
    dataset = AVBPMultiCaseHITDataset(
        root=args.root,
        cases=DEFAULT_M43_CASES,
        reference_dir=args.reference_dir,
        require_references=True,
        expected_snapshots_per_case=100,
    )
    index = _chosen_index(
        dataset, case_id=args.case_id,
        snapshot_number=args.snapshot_number,
    )
    standardizers = _load_standardizers(args.train_run_dir / "standardizers.pt")
    if not standardizers.physical_nondimensionalization:
        raise RuntimeError("the M43-B train-only scaler is not physically normalized")
    collator = GraphTaskCollator(
        FlowMatchingTask(
            state_fields=FIELDS,
            physical_nondimensionalization=True,
            validation_seed=1234,
            time_embedding_scale=1000.0,
        ),
        dataset.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    task = collator.task
    host_probe = collator([dataset[index]])
    problem_probe = task.make_validation_problem(standardizers.transform(host_probe))
    model, _ = instantiate_controlled_model(
        build_recipe(checkpointing=True), problem_probe,
        seed=int(state["seed"]),
    )
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.0)
    generator = torch.Generator(device=device).manual_seed(int(state["seed"]) + 1000)
    saved_path = args.train_run_dir / "last.pt"
    signed = _sha256_signature(args.train_run_dir)
    checkpoint = load_training_checkpoint(
        saved_path,
        signature_sha256=signed["sha256"],
        model=model, optimizer=optimizer, generator=generator,
    )
    metadata["checkpoint_epoch_zero_based"] = int(checkpoint["epoch"])
    metadata["checkpoint_model_class"] = str(checkpoint["model_class"])
    metadata["node_count"] = int(host_probe.inputs.shape[0])
    metadata["one_hop_directed_edges"] = int(host_probe.edge_index.shape[1])
    metadata["two_hop_directed_edges"] = int(
        host_probe.attention_edge_indices["dilated"].shape[1]
    )
    metadata["cuda_device"] = torch.cuda.get_device_name(device)
    metadata["cuda_capability"] = list(torch.cuda.get_device_capability(device))
    metadata["pytorch_version"] = torch.__version__
    metadata["host"] = socket.gethostname()
    metadata["python_version"] = platform.python_version()
    _write_json(args.output_dir / "input_provenance.json", metadata)

    device_standardizers = standardizers.to(device=device, dtype=torch.float32)
    del host_probe, problem_probe

    # Prime torch_sparse CSR and autograd kernels without profiler overhead.
    print("[M43-PROFILE] Warm-up 129^3 forward + backward + optimizer", flush=True)
    warm = _one_real_step(
        dataset=dataset, index=index, model=model, optimizer=optimizer,
        task=task, standardizers=device_standardizers, collator=collator,
        generator=generator, device=device,
    )
    # Restore original scientific state after warm-up: profile starts from
    # exactly the checkpoint's parameters, AdamW state and training RNG.
    load_training_checkpoint(
        saved_path,
        signature_sha256=signed["sha256"],
        model=model, optimizer=optimizer, generator=generator,
    )
    torch.cuda.reset_peak_memory_stats(device)
    print("[M43-PROFILE] Active CUDA trace: one full 129^3 training step", flush=True)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        with_flops=False,
        profile_memory=False,
    ) as profiler:
        measured = _one_real_step(
            dataset=dataset, index=index, model=model, optimizer=optimizer,
            task=task, standardizers=device_standardizers, collator=collator,
            generator=generator, device=device,
        )

    # CUDA events are already synchronized at every phase boundary.
    kernel_report = report_module.kernel_groups(profiler.events(), limit=50)
    operator_report = report_module.operator_groups(
        profiler.key_averages(), limit=50,
    )
    if kernel_report["kernel_count"] < 1:
        raise RuntimeError(
            "CUDA profiling generated no device kernels; inspect Kineto/CUPTI support"
        )
    report = {
        **metadata,
        "status": "profile_complete",
        "warmup": warm,
        "profiled": measured,
        "cuda_kernels": kernel_report,
        "pytorch_operators": operator_report,
        "interpretation_notes": [
            "Warm-up and profiled steps begin from the same checkpoint state "
            "after checkpoint reload; all optimizer changes are discarded.",
            "Loss represents one 129^3 physical snapshot, not six-case "
            "gradient-accumulation/update throughput.",
            "Backward phase includes non-reentrant checkpoint recomputation. "
            "Do not mistake recomputation kernels for optimizer processing.",
            "CUDA kernel duration sums, CPU operator totals and synchronized "
            "wall times are not mathematically interchangeable.",
            "CUPTI recording can add overhead and GPU memory allocation; "
            "compare profile-stage timings to the unprofiled warm step.",
            "No scientific model architecture, attention operator, precision, "
            "forward method or optimizer rule was changed.",
        ],
    }
    _write_json(args.output_dir / "profile_report.json", report)
    operator_text = report_module.format_table(
        operator_report["top_pytorch_operators"],
        name_key="name", seconds_key="self_gpu_seconds",
    )
    kernel_text = report_module.format_table(
        kernel_report["top_cuda_kernels"],
        name_key="name", seconds_key="total_seconds",
    )
    (args.output_dir / "top_pytorch_operators.txt").write_text(operator_text)
    (args.output_dir / "top_cuda_kernels.txt").write_text(kernel_text)
    if not args.skip_trace:
        print("[M43-PROFILE] Writing Chrome CUDA trace (may be large)", flush=True)
        profiler.export_chrome_trace(str(args.output_dir / "cuda_trace.json"))
    print(json.dumps({
        "status": "profile_complete",
        "output": str(args.output_dir),
        "checkpoint_epoch": metadata["checkpoint_epoch_zero_based"],
        "elapsed_seconds_by_phase": measured["times"],
        "kernel_time_seconds_by_phase": kernel_report["kernel_time_seconds_by_phase"],
        "top_cuda_kernel_names": [
            v["name"] for v in kernel_report["top_cuda_kernels"][:8]
        ],
    }, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "profile"), default="inspect")
    parser.add_argument(
        "--root", type=Path,
        default=Path("/archivenfs/phlusim/HIT_DNS_COMP"),
    )
    parser.add_argument(
        "--reference-dir", type=Path,
        default=Path("/scratch/coop/theret/HIT_DNS_COMP_case_references"),
    )
    parser.add_argument(
        "--train-run-dir", type=Path,
        default=Path(
            "/scratch/coop/theret/GraphAttention_runs/m43_b_multires/train_pilot_s42"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        required=True,
    )
    parser.add_argument(
        "--case-id",
        choices=[case.case_id for case in DEFAULT_M43_CASES if case.side == 129],
        default="Re94/Ma0.2",
    )
    parser.add_argument(
        "--snapshot-number", type=int, default=51,
    )
    parser.add_argument(
        "--skip-trace", action="store_true",
        help="Do not export potentially large Chrome CUDA trace",
    )
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    args.reference_dir = args.reference_dir.expanduser().resolve()
    args.train_run_dir = args.train_run_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    _validate_args(args)
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "inspect":
        print(json.dumps(inspect(args), indent=2), flush=True)
    else:
        run_profile(args)


if __name__ == "__main__":
    main()
