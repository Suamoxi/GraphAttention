"""Isolated multi-update M43-B adaLN continuation from the frozen 8-epoch checkpoint.

Both code paths execute the UNMODIFIED scripts.train_m43_b_multires.train_epoch:
- identical epoch-8 case-stratified interleave, 6 physical microbatches/update;
- identical train-only scaler, physical nondimensionalization, Flow Matching
  t/noise generator and equal-physical-sample gradient averaging;
- identical AdamW weights and moments restored before every arm;
- deterministic held-out VALIDATION only after each update, never test data.

Two independent replays PER path quantify normal CUDA sparse non-determinism.
No checkpoint, history, manifest, scaler or other input is ever rewritten.
All diagnostic model/optimizer steps are in memory and discarded.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import socket
import time
from typing import Any, Mapping
from unittest.mock import patch

import torch

from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, DEFAULT_M43_CASES
from graph_attention.data.multicase_split import balanced_case_interleave
from graph_attention.models import dit_common
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import GraphTaskCollator
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.benchmark_m43_adaln_broadcast import (
    _check_input, _reference_broadcast, _snapshot_parameters,
)
from scripts.train_m43_b_multires import (
    FIELDS, _indices_by_case, _load_standardizers, build_recipe,
    evaluate, file_sha256, load_training_checkpoint, train_epoch,
)


SOURCE_FILES = (
    "run_signature.json", "last.pt", "best.pt", "standardizers.pt",
    "split.json", "history.json", "history.csv",
)
ARM_ORDER = (
    ("indexed_a", True),
    ("expand_a", False),
    ("indexed_b", True),
    ("expand_b", False),
)
CASE_IDS = tuple(case.case_id for case in DEFAULT_M43_CASES)


def _validated_host_revision(revision: str) -> str:
    """Host SLURM job verifies git worktree; container cannot always read .git.

    Read a REQUIRED host-side full commit SHA passed as an explicit CLI
    argument. Refuse missing, shortened or malformed provenance rather than
    running Git again in the immutable Singularity image.
    """
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("--git-revision must be a 40-character host-verified Git SHA")
    return revision.lower()


def _generator_sha256(generator: torch.Generator) -> str:
    return hashlib.sha256(
        generator.get_state().cpu().numpy().tobytes()
    ).hexdigest()


def _validation_pick(
    validation_ids: tuple[str, ...],
    case_by_id: Mapping[str, str],
    *,
    per_case: int,
) -> tuple[str, ...]:
    """Fixed deterministic subset; never select held-out TEST samples."""
    if isinstance(per_case, bool) or not 1 <= per_case <= 3:
        raise ValueError("validation_per_case must be an integer in [1,3]")
    chosen: list[str] = []
    for case in CASE_IDS:
        pool = [name for name in validation_ids if case_by_id[name] == case]
        if len(pool) < per_case:
            raise ValueError(f"{case}: insufficient held-out validation samples")
        positions = [((i + 1) * len(pool)) // (per_case + 1)
                     for i in range(per_case)]
        if len(set(positions)) != per_case:
            raise AssertionError("validation picks must be unique")
        chosen.extend(pool[i] for i in positions)
    return tuple(chosen)


def _training_windows(
    order: tuple[int, ...],
    dataset: AVBPMultiCaseHITDataset,
    *,
    steps: int,
) -> tuple[tuple[str, ...], ...]:
    """Verify original balanced interleave: exactly one of six cases/window."""
    if isinstance(steps, bool) or not 1 <= steps <= 6:
        raise ValueError("steps must be an integer in [1,6]")
    if len(order) < steps * 6:
        raise ValueError("fewer original training samples than requested updates")
    windows = []
    for step in range(steps):
        indices = order[6 * step: 6 * (step + 1)]
        ids = tuple(dataset.sample_ids[index] for index in indices)
        cases = tuple(dataset.case_by_sample_id[item] for item in ids)
        if cases != CASE_IDS or len(set(ids)) != 6:
            raise RuntimeError(
                f"step {step+1}: not one six-case balanced M43-B window: {cases}"
            )
        windows.append(ids)
    if len(set(item for window in windows for item in window)) != steps * 6:
        raise RuntimeError("repeat of a physical sample across continuation steps")
    return tuple(windows)


def _tensor_comparison(
    reference: Mapping[str, torch.Tensor | None],
    candidate: Mapping[str, torch.Tensor | None],
) -> dict[str, Any]:
    """Combined L2, maximum absolute delta, top relative and absolute drifts."""
    if set(reference) != set(candidate):
        raise RuntimeError("parameter keys differ across continuation arms")
    squared_differences = squared_reference = 0.0
    max_abs = 0.0
    per_tensor: list[dict[str, Any]] = []
    for name in sorted(reference):
        old, new = reference[name], candidate[name]
        if old is None or new is None:
            if old is not None or new is not None:
                raise RuntimeError(f"{name}: gradient presence mismatch")
            continue
        if old.shape != new.shape or old.dtype != new.dtype:
            raise RuntimeError(f"{name}: shape/dtype differs")
        if not torch.isfinite(old).all() or not torch.isfinite(new).all():
            raise FloatingPointError(f"{name}: nonfinite diagnostic tensor")
        diff = new.double() - old.double()
        old64 = old.double()
        diff_l2 = float(torch.linalg.vector_norm(diff))
        ref_l2 = float(torch.linalg.vector_norm(old64))
        squared_differences += diff_l2 ** 2
        squared_reference += ref_l2 ** 2
        largest = float(diff.abs().max()) if diff.numel() else 0.0
        max_abs = max(max_abs, largest)
        per_tensor.append({
            "name": name,
            "reference_l2": ref_l2,
            "difference_l2": diff_l2,
            "relative_l2": diff_l2 / max(ref_l2, 1.e-14),
            "max_abs": largest,
        })
    meaningful = [p for p in per_tensor if p["reference_l2"] >= 1.e-4]
    return {
        "combined_relative_l2": (
            math.sqrt(squared_differences) / max(math.sqrt(squared_reference), 1.e-14)
        ),
        "combined_absolute_l2": math.sqrt(squared_differences),
        "reference_l2": math.sqrt(squared_reference),
        "max_abs": max_abs,
        "tensors": len(per_tensor),
        "worst_relative": sorted(
            per_tensor, key=lambda r: r["relative_l2"], reverse=True,
        )[:8],
        "worst_meaningful_relative": sorted(
            meaningful, key=lambda r: r["relative_l2"], reverse=True,
        )[:8],
        "worst_absolute": sorted(
            per_tensor, key=lambda r: r["difference_l2"], reverse=True,
        )[:8],
    }


def _state_comparison(reference: dict, candidate: dict) -> dict[str, Any]:
    a, b = reference["post_update"], candidate["post_update"]
    left, right = a["optimizer"], b["optimizer"]
    if left["groups"] != right["groups"] or set(left["state"]) != set(right["state"]):
        raise RuntimeError("AdamW configuration or parameter group IDs differ")
    moments: dict[str, dict[str, torch.Tensor]] = {
        "exp_avg": {}, "exp_avg_sq": {},
    }
    for pid in sorted(left["state"]):
        past, now = left["state"][pid], right["state"][pid]
        if set(past) != set(now):
            raise RuntimeError(f"optimizer state key mismatch for parameter {pid}")
        for name in past:
            if name in moments:
                moments[name][str(pid)] = past[name]
                # Right moments are flattened in the corresponding map below.
            elif isinstance(past[name], torch.Tensor):
                if not torch.equal(past[name], now[name]):
                    raise RuntimeError(f"AdamW scalar/counter differs: {pid}/{name}")
            elif past[name] != now[name]:
                raise RuntimeError(f"AdamW scalar differs: {pid}/{name}")
    result = {
        "weights": _tensor_comparison(a["weights"], b["weights"]),
        "gradients": _tensor_comparison(reference["gradients"], candidate["gradients"]),
    }
    for name in moments:
        result[name] = _tensor_comparison(
            moments[name],
            {str(pid): right["state"][pid][name] for pid in sorted(right["state"])},
        )
    return result


def _continuation_comparison(
    reference_records: list[dict], candidate_records: list[dict],
    reference_states: list[dict], candidate_states: list[dict],
) -> list[dict[str, Any]]:
    if not (
        len(reference_records) == len(candidate_records)
        == len(reference_states) == len(candidate_states)
    ):
        raise ValueError("unequal optimizer-step counts")
    compared: list[dict[str, Any]] = []
    for i, (old, new, old_state, new_state) in enumerate(
        zip(reference_records, candidate_records, reference_states,
            candidate_states, strict=True)
    ):
        if old["step_sample_ids"] != new["step_sample_ids"]:
            raise RuntimeError("train sample ordering differs across arms")
        if old["generator_sha256_after_step"] != new["generator_sha256_after_step"]:
            raise RuntimeError("Flow Matching RNG progression differs across arms")
        if old["validation_ids"] != new["validation_ids"]:
            raise RuntimeError("validation population differs across arms")
        if old["samples_per_step"] != new["samples_per_step"] or old["samples_per_step"] != 6:
            raise RuntimeError("six-sample gradient accumulation differs")
        old_validation, new_validation = old["validation"], new["validation"]
        if old_validation["samples"] != new_validation["samples"]:
            raise RuntimeError("validation counts differ across arms")
        by_case = {
            case: {
                "indexed_mse": old_validation["by_case"][case]["mse"],
                "candidate_mse": new_validation["by_case"][case]["mse"],
                "absolute_difference": abs(
                    old_validation["by_case"][case]["mse"]
                    - new_validation["by_case"][case]["mse"]
                ),
            }
            for case in CASE_IDS
        }
        compared.append({
            "optimizer_step": i + 1,
            "train_velocity_mse_abs": abs(
                old["step_mean_velocity_mse"] - new["step_mean_velocity_mse"]
            ),
            "validation_velocity_mse_abs": abs(
                old_validation["mean"] - new_validation["mean"]
            ),
            "validation_velocity_mse_rel": abs(
                old_validation["mean"] - new_validation["mean"]
            ) / max(old_validation["mean"], 1.e-14),
            "validation_by_case": by_case,
            "state": _state_comparison(old_state, new_state),
        })
    return compared


class _StepObserver:
    """Instrumentation after optimizer.step() in unchanged M43 train_epoch."""

    def __init__(
        self, *, dataset, validation_indices, task, standardizers,
        collator, device, generator, windows,
    ):
        self.dataset = dataset
        self.validation_indices = validation_indices
        self.task = task
        self.standardizers = standardizers
        self.collator = collator
        self.device = device
        self.generator = generator
        self.windows = windows
        self.records: list[dict[str, Any]] = []
        self.states: list[dict[str, Any]] = []

    def log_step(self, *, model, optimizer, epoch, global_step,
                 step_mean_velocity_mse, samples_per_step,
                 elapsed_seconds, per_case_loss, per_resolution_loss,
                 forward_per_resolution_seconds, backward_per_resolution_seconds,
                 **unused):
        step = len(self.records)
        if step >= len(self.windows):
            raise RuntimeError("observer received more optimizer updates than scheduled")
        if samples_per_step != 6 or set(per_case_loss) != set(CASE_IDS):
            raise RuntimeError("M43 accumulation does not contain one sample per case")
        if not math.isfinite(float(step_mean_velocity_mse)):
            raise FloatingPointError("nonfinite training loss")
        # The original runner performs validation only at epoch end; this
        # dedicated diagnostic additionally observes a fixed held-out subset
        # after each update. Restore model mode and GLOBAL RNG afterwards.
        previous_training_mode = model.training
        try:
            with torch.random.fork_rng(devices=[self.device.index or 0]):
                validation = evaluate(
                    self.dataset, self.validation_indices,
                    model=model, task=self.task, standardizers=self.standardizers,
                    collator=self.collator, device=self.device,
                )
        finally:
            model.train(previous_training_mode)
        state = {
            "post_update": _snapshot_parameters(model, optimizer),
            "gradients": {
                name: None if param.grad is None else param.grad.detach().cpu().clone()
                for name, param in model.named_parameters()
            },
        }
        self.states.append(state)
        self.records.append({
            "optimizer_step": step + 1,
            "source_epoch": epoch,
            "step_sample_ids": list(self.windows[step]),
            "validation_ids": [
                self.dataset.sample_ids[i] for i in self.validation_indices
            ],
            "generator_sha256_after_step": _generator_sha256(self.generator),
            "step_mean_velocity_mse": float(step_mean_velocity_mse),
            "samples_per_step": int(samples_per_step),
            "per_case_train_loss": dict(per_case_loss),
            "per_resolution_train_loss": dict(per_resolution_loss),
            "forward_by_resolution_seconds": dict(forward_per_resolution_seconds),
            "backward_by_resolution_seconds": dict(backward_per_resolution_seconds),
            "optimizer_step_seconds_excluding_diagnostic_validation": float(
                elapsed_seconds
            ),
            "validation": validation,
        })
        print(
            f"[M43 continuation] arm-step={step+1}/{len(self.windows)} "
            f"train={step_mean_velocity_mse:.7e} "
            f"val={validation['mean']:.7e} "
            f"step_s={elapsed_seconds:.2f}",
            flush=True,
        )


@contextmanager
def _broadcast_mode(indexed: bool):
    if indexed:
        with patch.object(
            dit_common, "_broadcast_graph_features", _reference_broadcast
        ):
            yield
    else:
        yield


def _input_hashes(source: Path) -> dict[str, str]:
    return {
        path: file_sha256(source / path)
        for path in SOURCE_FILES if (source / path).is_file()
    }


def _verify_selected_samples(
    dataset: AVBPMultiCaseHITDataset,
    signature: Mapping[str, Any],
    sample_ids: set[str],
) -> None:
    declared = {row["id"]: row for row in signature["samples"]}
    if len(declared) != len(signature["samples"]):
        raise ValueError("duplicate IDs in original source scientific signature")
    if set(declared) != set(dataset.sample_ids):
        raise RuntimeError("reconstructed dataset differs from original signature")
    for sid in sample_ids:
        expected = declared[sid]
        spec = dataset.sample_specs[dataset.sample_ids.index(sid)]
        for key, actual_file, size_key, time_key in (
            ("path", Path(spec.snapshot_file), "snapshot_size", "snapshot_mtime_ns"),
            ("mesh", Path(spec.mesh_file), "mesh_size", "mesh_mtime_ns"),
        ):
            original_file = Path(expected[key])
            if actual_file.resolve() != original_file.resolve():
                raise RuntimeError(f"{sid}: {key} path differs from source signature")
            stat = actual_file.stat()
            if stat.st_size != expected[size_key] or stat.st_mtime_ns != expected[time_key]:
                raise RuntimeError(f"{sid}: {key} content metadata changed")
        if dataset.case_by_sample_id[sid] != expected["case_id"]:
            raise RuntimeError(f"{sid}: physical case differs from signature")


def _preflight(args: argparse.Namespace):
    source_run = args.train_run_dir
    output = args.output_dir
    for protected in (source_run, args.root, args.reference_dir):
        if (output == protected or output in protected.parents
            or protected in output.parents):
            raise ValueError(f"diagnostic output cannot contain or be inside {protected}")
    original_hashes = _input_hashes(source_run)
    if set(original_hashes) != set(SOURCE_FILES):
        raise FileNotFoundError(
            f"original scientific run files missing: {set(SOURCE_FILES)-set(original_hashes)}"
        )
    # Reuse the proven original GH200 benchmark signature and reference checks.
    original_info = _check_input(args)
    signature = json.loads(
        (source_run / "run_signature.json").read_text()
    )["content"]
    dataset = AVBPMultiCaseHITDataset(
        root=args.root, cases=DEFAULT_M43_CASES,
        reference_dir=args.reference_dir,
        require_references=True, expected_snapshots_per_case=100,
    )
    train_ids = tuple(signature["train_ids"])
    validation_ids = tuple(signature["validation_ids"])
    test_ids = tuple(signature["test_ids"])
    if (set(train_ids) & set(validation_ids)
        or set(train_ids) & set(test_ids)
        or set(validation_ids) & set(test_ids)):
        raise RuntimeError("source scientific train/validation/test leakage")
    split = json.loads((source_run / "split.json").read_text())
    for name, ids in (
        ("train", train_ids), ("validation", validation_ids), ("test", test_ids)
    ):
        if split[name + "_ids"] != list(ids):
            raise RuntimeError(f"original signature differs from split.json: {name}")
    id_to_index = {sid: i for i, sid in enumerate(dataset.sample_ids)}
    if not all(sid in id_to_index for sid in (*train_ids, *validation_ids, *test_ids)):
        raise RuntimeError("original split refers to missing physical samples")
    checkpoint = torch.load(
        source_run / "last.pt", map_location="cpu", weights_only=True,
    )
    if checkpoint.get("signature_sha256") != original_info["source_signature_sha256"]:
        raise RuntimeError("original checkpoint scientific signature mismatch")
    epoch = int(checkpoint["epoch"]) + 1
    if epoch != len(checkpoint["history"]):
        raise RuntimeError("checkpoint epoch/history inconsistent")
    training_cases = _indices_by_case(train_ids, dataset)
    order = balanced_case_interleave(
        training_cases, seed=int(signature["seed"]), epoch=epoch,
    )
    windows = _training_windows(order, dataset, steps=args.steps)
    selected_validation = _validation_pick(
        validation_ids, dataset.case_by_sample_id,
        per_case=args.validation_per_case,
    )
    _verify_selected_samples(
        dataset, signature,
        set(selected_validation) | {sid for window in windows for sid in window},
    )
    standardizers = _load_standardizers(source_run / "standardizers.pt")
    if (tuple(standardizers.train_sample_ids) != train_ids
        or not standardizers.physical_nondimensionalization):
        raise RuntimeError("train-only scaling was not restored from original split")
    return (original_info, signature, dataset, training_cases, windows,
            tuple(id_to_index[sid] for sid in selected_validation),
            standardizers, original_hashes, epoch)


def _diagnostic_gate(
    comparisons: list[dict[str, Any]],
    reference_repeat: list[dict[str, Any]],
    expand_repeat: list[dict[str, Any]],
) -> dict[str, Any]:
    """Explicit conservative observational criteria, not bitwise claims.

    The source 1e-5 prediction tolerance is not used here: unchanged sparse
    GPU kernels are non-bitwise-reproducible even for reference-vs-reference.
    """
    limits = {
        "combined_post_update_weight_relative_l2": 5.e-5,
        "combined_parameter_gradient_relative_l2": 5.e-4,
        "validation_velocity_mse_abs": 3.e-3,
    }
    checks = []
    for i, (cross, same_a, same_b) in enumerate(
        zip(comparisons, reference_repeat, expand_repeat, strict=True)
    ):
        key_values = {
            "combined_post_update_weight_relative_l2":
                cross["state"]["weights"]["combined_relative_l2"],
            "combined_parameter_gradient_relative_l2":
                cross["state"]["gradients"]["combined_relative_l2"],
            "validation_velocity_mse_abs":
                cross["validation_velocity_mse_abs"],
        }
        checks.append({
            "step": i + 1,
            "values": key_values,
            "threshold_passed": all(key_values[name] <= limit
                                    for name, limit in limits.items()),
            "same_mode_floor": {
                "weight_relative_l2": max(
                    same_a["state"]["weights"]["combined_relative_l2"],
                    same_b["state"]["weights"]["combined_relative_l2"],
                ),
                "gradient_relative_l2": max(
                    same_a["state"]["gradients"]["combined_relative_l2"],
                    same_b["state"]["gradients"]["combined_relative_l2"],
                ),
                "validation_abs": max(
                    same_a["validation_velocity_mse_abs"],
                    same_b["validation_velocity_mse_abs"],
                ),
            },
        })
    return {
        "type": "diagnostic_provisional_not_scientific_resume_authorization",
        "limits": limits,
        "step_checks": checks,
        "passed": all(item["threshold_passed"] for item in checks),
        "note": (
            "Passing these observational thresholds plus same-mode controls "
            "does not authorize unreviewed scientific checkpoint migration. "
            "No test-split evaluation was used."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("M43 continuation requires a GPU")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("diagnostic continuation is single-GPU, not DDP")
    if args.steps < 2:
        raise ValueError("use at least 2 optimizer steps to measure continuation drift")
    (origin, signature, dataset, cases, windows, validation_indices,
     standardizers, before, source_epoch) = _preflight(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    device = torch.device("cuda:0")
    torch.manual_seed(int(signature["seed"]))
    task = FlowMatchingTask(
        state_fields=FIELDS, physical_nondimensionalization=True,
        validation_seed=1234, time_embedding_scale=1000.,
    )
    collator = GraphTaskCollator(
        task, dataset.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    # The model is instantiated exactly as in the source scientific training
    # runner before checkpoint restoration. Probe never accesses TEST data.
    probe = collator([dataset[validation_indices[0]]])
    prepared_probe = standardizers.transform(probe)
    model_probe = task.make_validation_problem(prepared_probe)
    model, _ = instantiate_controlled_model(
        build_recipe(checkpointing=True), model_probe,
        seed=int(signature["seed"]),
    )
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.e-4, weight_decay=0.)
    generator = torch.Generator(device=device).manual_seed(
        int(signature["seed"]) + 1000
    )
    scaler_on_device = standardizers.to(device=device, dtype=torch.float32)
    del probe, prepared_probe, model_probe

    results: dict[str, Any] = {
        "status": "diagnostic_in_progress",
        "provenance": {
            **origin,
            "benchmark_git_revision": _validated_host_revision(args.git_revision),
            "revision_provenance": (
                "Host SLURM worktree validated clean and on reviewed ancestry "
                "before Singularity execution; revision passed as CLI argument"
            ),
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "source_checkpoint_epoch_zero_based": source_epoch - 1,
            "continuation_epoch_zero_based": source_epoch,
            "first_steps_of_epoch": args.steps,
            "microbatches_per_optimizer_step": 6,
            "physical_sample_weighting": "mean_of_six_sample_loss_sums",
            "step_sample_ids": [list(window) for window in windows],
            "validation_ids": [
                dataset.sample_ids[i] for i in validation_indices
            ],
            "validation_seed": 1234,
            "test_ids_consumed": False,
            "input_sha256_before": before,
            "validation_is_diagnostic_subset_not_full_epoch": True,
            "model_parameter_or_optimizer_state_changed_on_disk": False,
        },
        "arms": {},
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(results["provenance"], indent=2) + "\n"
    )
    snapshots: dict[str, list[dict[str, Any]]] = {}
    rng_before = None
    try:
        for label, use_indexed in ARM_ORDER:
            checkpoint = load_training_checkpoint(
                args.train_run_dir / "last.pt",
                signature_sha256=origin["source_signature_sha256"],
                model=model, optimizer=optimizer, generator=generator,
            )
            if int(checkpoint["epoch"]) + 1 != source_epoch:
                raise RuntimeError("checkpoint source epoch changed during diagnostics")
            start_rng = _generator_sha256(generator)
            if rng_before is not None and start_rng != rng_before:
                raise RuntimeError("original Flow Matching generator not reproduced")
            rng_before = start_rng
            observer = _StepObserver(
                dataset=dataset, validation_indices=validation_indices,
                task=task, standardizers=scaler_on_device,
                collator=collator, device=device,
                generator=generator, windows=windows,
            )
            torch.cuda.reset_peak_memory_stats(device)
            arm_started = time.perf_counter()
            print(
                f"[M43 continuation] BEGIN {label}; "
                f"source_epoch={source_epoch}, updates={args.steps}, "
                f"checkpoint_generator={start_rng[:16]}...",
                flush=True,
            )
            with _broadcast_mode(use_indexed):
                stats = train_epoch(
                    dataset, cases,
                    epoch=source_epoch, seed=int(signature["seed"]),
                    accumulation=6, model=model, optimizer=optimizer,
                    task=task, standardizers=scaler_on_device,
                    collator=collator, device=device, generator=generator,
                    max_steps=args.steps, tensorboard=observer,
                    optimizer_step_offset=0,
                )
            if len(observer.records) != args.steps:
                raise RuntimeError(f"{label}: missing optimizer updates")
            if stats["optimizer_steps"] != args.steps or stats["samples"] != 6 * args.steps:
                raise RuntimeError(f"{label}: source M43 accumulation changed")
            results["arms"][label] = {
                "broadcast": "legacy_indexed" if use_indexed else "expand_fastpath",
                "elapsed_seconds_including_diagnostic_validation": (
                    time.perf_counter() - arm_started
                ),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/1024**3,
                "initial_generator_sha256": start_rng,
                "final_generator_sha256": _generator_sha256(generator),
                "train_epoch_subset": stats,
                "steps": observer.records,
            }
            snapshots[label] = observer.states
            # Save incremental JSON, never production model/optimizer state.
            (args.output_dir / "continuation_progress.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(
                f"[M43 continuation] END {label}; "
                f"elapsed={time.perf_counter()-arm_started:.1f}s",
                flush=True,
            )
        comparisons = {
            "indexed_vs_expand": _continuation_comparison(
                results["arms"]["indexed_a"]["steps"],
                results["arms"]["expand_a"]["steps"],
                snapshots["indexed_a"], snapshots["expand_a"],
            ),
            "indexed_same_mode": _continuation_comparison(
                results["arms"]["indexed_a"]["steps"],
                results["arms"]["indexed_b"]["steps"],
                snapshots["indexed_a"], snapshots["indexed_b"],
            ),
            "expand_same_mode": _continuation_comparison(
                results["arms"]["expand_a"]["steps"],
                results["arms"]["expand_b"]["steps"],
                snapshots["expand_a"], snapshots["expand_b"],
            ),
            "second_pair_indexed_vs_expand": _continuation_comparison(
                results["arms"]["indexed_b"]["steps"],
                results["arms"]["expand_b"]["steps"],
                snapshots["indexed_b"], snapshots["expand_b"],
            ),
        }
        results["comparisons"] = comparisons
        results["gate"] = _diagnostic_gate(
            comparisons["indexed_vs_expand"],
            comparisons["indexed_same_mode"], comparisons["expand_same_mode"],
        )
        results["status"] = (
            "diagnostic_thresholds_passed_no_resume_authorization"
            if results["gate"]["passed"] else
            "diagnostic_thresholds_failed_no_resume_authorization"
        )
    finally:
        after = _input_hashes(args.train_run_dir)
        results["provenance"]["input_sha256_after"] = after
        results["provenance"]["source_files_unchanged"] = before == after
        results["elapsed_seconds"] = time.perf_counter() - start
        (args.output_dir / "continuation_summary.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        if before != after:
            raise RuntimeError("source scientific files changed during continuation")
    print(json.dumps({
        "status": results["status"],
        "gate": results["gate"],
        "output": str(args.output_dir),
        "source_files_unchanged": True,
    }, indent=2), flush=True)
    if not results["gate"]["passed"]:
        raise RuntimeError(
            "controlled continuation numerical diagnostic thresholds exceeded; "
            "review output without resuming scientific training"
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "/archivenfs/phlusim/HIT_DNS_COMP"
    ))
    parser.add_argument("--reference-dir", type=Path, default=Path(
        "/scratch/coop/theret/HIT_DNS_COMP_case_references"
    ))
    parser.add_argument("--train-run-dir", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m43_b_multires/train_pilot_s42"
    ))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--git-revision", required=True, type=_validated_host_revision,
        help="Full source SHA verified by the host-side SLURM git preflight",
    )
    parser.add_argument("--steps", default=3, type=int)
    parser.add_argument("--validation-per-case", default=1, type=int)
    # The existing input validator also inspects one original 129^3 sample.
    parser.add_argument("--case-id", default="Re94/Ma0.2")
    parser.add_argument("--snapshot-number", default=51, type=int)
    args = parser.parse_args()
    for name in ("root", "reference_dir", "train_run_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not 2 <= args.steps <= 6:
        parser.error("--steps must be in [2,6]")
    if not 1 <= args.validation_per_case <= 3:
        parser.error("--validation-per-case must be in [1,3]")
    return args


if __name__ == "__main__":
    run(parse_args())
