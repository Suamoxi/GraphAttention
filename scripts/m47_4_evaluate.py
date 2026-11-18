"""M47.4: validation-only velocity audit and independent physical generation.

Use fixed epoch-5/10 *last* checkpoints to compare identical training budgets.
M47.2's two-epoch contract is unchanged by default; this script explicitly
requests the continued-run contract, verifies source provenance, and uses
only the original validation partition for both FM paths and DNS references.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import torch

from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, DEFAULT_M43_CASES
from graph_attention.tasks import FlowMatchingTask
from graph_attention.tasks.flow_matching import _deterministic_sampling_source, _flow_problem
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m47_4_stage import PROVENANCE, read_json, source_record
from scripts.m47_fm_diagnostics import aggregate_velocity, roundtrip_audit, velocity_metrics
from scripts.m47_generate_ensembles import CASE_NAMES, analyze_case, load_contract
from scripts.m47_generation_metrics import CHANNELS
from scripts.m47_reference_conditioning import CONDITION_NAMES
from scripts.train_generative import _load_standardizers
from scripts.train_m43_b_multires import FIELDS, _forward, build_recipe


def validate_provenance(run_dir: Path, mode: str, epochs: int) -> dict[str, Any]:
    if epochs not in (5, 10) or run_dir.name != f"epoch{epochs:02d}":
        raise ValueError("only immutable M47.4 epoch05/epoch10 snapshots are allowed")
    provenance = read_json(run_dir.parent / PROVENANCE)
    source = Path(provenance["source_path"])
    if source_record(source, mode) != provenance:
        raise ValueError("original M47.1 checkpoint provenance differs")
    # This function is also independently unit-testable without CUDA.
    return provenance


def validate_checkpoint_metadata(checkpoint: dict[str, Any], *,
                                 signature_sha: str, epochs: int) -> None:
    if (checkpoint.get("signature_sha256") != signature_sha
        or checkpoint.get("epoch") != epochs - 1
        or len(checkpoint.get("history", [])) != epochs
        or checkpoint["history"][-1].get("epoch") != epochs - 1):
        raise ValueError("continued checkpoint is not the exact requested epoch")
    for name in ("model_state_dict", "optimizer_state_dict",
                 "training_generator_state"):
        if name not in checkpoint:
            raise ValueError(f"continued checkpoint missing {name}")


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("M47.4 evaluation requires a CUDA GPU")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite M47.4 evaluation: {args.output_dir}")
    if args.samples_per_case < 1 or args.steps < 1 or args.validation_per_case != 5:
        raise ValueError("M47.4 requires 5 validation snapshots/case and positive samples/steps")
    if args.seed < 0 or args.node_subsample < 1 or args.k_bins < 1:
        raise ValueError("invalid M47.4 metric parameters")
    if not 1 <= args.max_lag <= 32:
        raise ValueError("invalid correlation lag")
    requested_cases = tuple(args.case or ("Re54/Ma0.2",))
    if (len(set(requested_cases)) != len(requested_cases)
        or any(c not in CASE_NAMES for c in requested_cases)):
        raise ValueError("unexpected or duplicate physical generation case")
    provenance = validate_provenance(args.run_dir, args.condition, args.epochs)
    signature, training_summary, split = load_contract(
        run_dir=args.run_dir, ref_dir=args.reference_dir,
        mode=args.condition, root=args.archive_root,
        expected_epochs=args.epochs,
    )
    dataset = AVBPMultiCaseHITDataset(
        root=args.archive_root, cases=DEFAULT_M43_CASES,
        reference_dir=args.reference_dir, require_references=True,
        expected_snapshots_per_case=100,
        selected_sample_ids=tuple(split["validation_ids"]),
    )
    if set(dataset.sample_ids) != set(split["validation_ids"]):
        raise ValueError("validation DNS population differs from frozen M47.1 split")
    by_case = {
        case: [i for i, sid in enumerate(dataset.sample_ids)
               if dataset.case_by_sample_id[sid] == case]
        for case in CASE_NAMES
    }
    if any(len(by_case[case]) != 5 for case in CASE_NAMES):
        raise ValueError("M47.4 needs exactly five original validation fields/case")

    scalers = _load_standardizers(args.run_dir / "standardizers.pt")
    if (not scalers.physical_nondimensionalization
        or scalers.weighting != "sample_balanced"
        or tuple(scalers.train_sample_ids) != tuple(split["train_ids"])
        or scalers.inputs.channel_names != scalers.targets.channel_names
        or tuple(name.split(".")[0] for name in scalers.inputs.channel_names) != CHANNELS):
        raise ValueError("M47.4 scaler deviates from the frozen train-only M47.1 scaler")
    device = torch.device("cuda:0")
    scalers = scalers.to(device=device, dtype=torch.float32)
    task = FlowMatchingTask(
        state_fields=FIELDS, conditioning_parameters=CONDITION_NAMES[args.condition],
        physical_nondimensionalization=True, validation_seed=1234,
        time_embedding_scale=1000.0,
    )
    collator = GraphTaskCollator(
        task, dataset.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    probe = task.make_validation_problem(
        scalers.transform(task_batch_to_device(
            collator([dataset[by_case[CASE_NAMES[0]][0]]]),
            device=device, dtype=torch.float32,
        ))
    )
    model, _ = instantiate_controlled_model(build_recipe(), probe, seed=42)
    model = model.to(device=device, dtype=torch.float32)
    checkpoint = torch.load(args.run_dir / "last.pt", map_location="cpu",
                            weights_only=True)
    validate_checkpoint_metadata(checkpoint, signature_sha=signature["sha256"],
                                 epochs=args.epochs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    del checkpoint, probe

    times = (0.0, 0.25, 0.5, 0.75, 1.0)
    rows = []
    roundtrip_max = 0.0
    for case in CASE_NAMES:
        for i in by_case[case]:
            sid = dataset.sample_ids[i]
            raw = task_batch_to_device(collator([dataset[i]]),
                                       device=device, dtype=torch.float32)
            scaled = scalers.transform(raw)
            normalization = roundtrip_audit(raw, scaled, scalers)
            roundtrip_max = max(roundtrip_max, normalization["max_standardized_roundtrip"])
            if roundtrip_max > 1e-3:
                raise ValueError(f"M47.4 normalization failure: {sid}")
            source = _deterministic_sampling_source(
                scaled, args.seed, (f"M47.3/validation/{sid}",),
            )
            by_time = []
            for t in times:
                t_batch = torch.full((scaled.num_graphs,), t,
                                     dtype=source.dtype, device=device)
                problem = _flow_problem(scaled, t_batch, source,
                                        time_embedding_scale=task.time_embedding_scale)
                prediction = _forward(model, problem)
                by_time.append({
                    "t": t, "channels": velocity_metrics(
                        prediction, problem.targets,
                        tuple(name.split(".")[0] for name in scaled.input_channels),
                    ),
                })
            rows.append({"sample_id": sid, "case": case, "by_time": by_time,
                         "normalization": normalization})
            print(f"[M47.4] epoch={args.epochs} mode={args.condition} FM={sid}",
                  flush=True)
            del raw, scaled, source

    # Physical generation uses the *validation* DNS population. The generated
    # Gaussian keys are identical for all modes and both training checkpoints.
    physical = {}
    for case in requested_cases:
        sample_ids = [dataset.sample_ids[i] for i in by_case[case]]
        report = analyze_case(
            model=model, task=task, scalers=scalers, dataset=dataset,
            case_name=case, sample_ids=sample_ids,
            device=device, steps=args.steps, samples=args.samples_per_case,
            seed=args.seed, subsample_nodes=args.node_subsample,
            k_bins=args.k_bins, max_lag=args.max_lag, field_dir=None,
        )
        physical[case] = report

    result = {
        "version": "m47_4_validation_and_generation_v1",
        "status": "complete",
        "mode": args.condition,
        "epochs_completed": args.epochs,
        "evaluated_checkpoint": str(args.run_dir / "last.pt"),
        "checkpoint_kind": "last; fixed-epoch comparison, not best-on-validation",
        "training_signature_sha256": signature["sha256"],
        "original_training_revision": provenance["original_training_revision"],
        "evaluation_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True).stdout.strip(),
        "original_validation_population": "30 M47.1 validation snapshots; 5 per regime",
        "test_snapshot_state_used": False,
        "note": "M47.1 legacy trainer computes test-at-best at stage completion; evaluation here uses validation only",
        "shared_sampling_seed": args.seed,
        "flow_times": list(times),
        "validation_per_case": 5,
        "normalization_max_roundtrip": roundtrip_max,
        "validation_velocity_mse_at_fixed_times": aggregate_velocity(rows, CASE_NAMES, times),
        "training_validation_best_mse": training_summary["best_validation_velocity_mse"],
        "training_validation_best_epoch": training_summary["best_epoch"],
        "physical_generation_steps": args.steps,
        "physical_generation_samples_per_case": args.samples_per_case,
        "physical_generation_cases": list(requested_cases),
        "physical_generation": {
            case: {k: v for k, v in report.items()
                   if k not in ("reference_per_sample", "generated_per_sample")}
            for case, report in physical.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "validation_details.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    for case, report in physical.items():
        (args.output_dir / f"case_{case.replace('/', '_')}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"[M47.4] COMPLETE: {args.output_dir / 'summary.json'}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITION_NAMES), required=True)
    parser.add_argument("--epochs", type=int, choices=(5, 10), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references"))
    parser.add_argument("--archive-root", type=Path, default=Path(
        "/archivenfs/phlusim/HIT_DNS_COMP"))
    parser.add_argument("--case", action="append", choices=CASE_NAMES)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples-per-case", type=int, default=4)
    parser.add_argument("--validation-per-case", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--node-subsample", type=int, default=8192)
    parser.add_argument("--k-bins", type=int, default=32)
    parser.add_argument("--max-lag", type=int, default=8)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
