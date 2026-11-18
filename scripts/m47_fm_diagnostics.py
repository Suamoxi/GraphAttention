"""M47.3: read-only Flow Matching diagnostics on frozen M47.1 checkpoints.

All direct velocity diagnostics use the validation partition, not held-out test
snapshots. Fixed Gaussian paths are shared across models and across time.
Trajectory sampling uses the same Gaussian source and Heun updates as M47.2.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, DEFAULT_M43_CASES
from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from graph_attention.tasks import FlowMatchingTask
from graph_attention.tasks.flow_matching import (
    _deterministic_sampling_source, _flow_problem, _model_velocity,
)
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m47_generate_ensembles import CASE_NAMES, load_contract
from scripts.m47_generation_metrics import CHANNELS, field_diagnostics, relative_spectrum_l1
from scripts.m47_reference_conditioning import CONDITION_NAMES
from scripts.train_generative import _load_standardizers
from scripts.train_m43_b_multires import FIELDS, build_recipe, _forward


def velocity_metrics(
    prediction: torch.Tensor, target: torch.Tensor, channels: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    """Equal-node diagnostics in standardized velocity coordinates."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have matching [N,C] shapes")
    if len(channels) != prediction.shape[1] or prediction.shape[0] == 0:
        raise ValueError("invalid diagnostic channel names or empty graph")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("non-finite Flow Matching velocity")
    out = {}
    for column, name in enumerate(channels):
        p = prediction[:, column].double()
        y = target[:, column].double()
        mse = torch.mean((p - y).square()).item()
        zero_mse = torch.mean(y.square()).item()
        p_rms = torch.sqrt(torch.mean(p.square())).item()
        y_rms = math.sqrt(zero_mse)
        denominator = p_rms * y_rms
        cosine = (torch.mean(p * y).item() / denominator) if denominator else None
        out[name] = {
            "mse": mse,
            "zero_predictor_mse": zero_mse,
            "mse_over_zero_predictor": mse / zero_mse if zero_mse else None,
            "improvement_over_zero_predictor": 1.0 - mse / zero_mse if zero_mse else None,
            "predicted_rms": p_rms,
            "target_rms": y_rms,
            "predicted_mean": torch.mean(p).item(),
            "target_mean": torch.mean(y).item(),
            "cosine_similarity": cosine,
        }
    return out


def roundtrip_audit(raw: Any, scaled: Any, scalers: Any) -> dict[str, Any]:
    """Check affine channel transformations and input/target FM symmetry."""
    names = tuple(raw.input_channels)
    if tuple(raw.target_channels) != names or tuple(scaled.input_channels) != names:
        raise ValueError("FM input/target channels differ")
    if not torch.equal(raw.inputs, raw.targets):
        raise ValueError("FM physical inputs and targets differ")
    if not torch.equal(scaled.inputs, scaled.targets):
        raise ValueError("FM standardized inputs and targets differ")
    if not torch.equal(scalers.inputs.mean, scalers.targets.mean):
        raise ValueError("FM input/target fitted means differ")
    if not torch.equal(scalers.inputs.scale, scalers.targets.scale):
        raise ValueError("FM input/target fitted scales differ")
    back = scalers.inputs.inverse(scaled.inputs, names)
    again = scalers.inputs.transform(back, names)
    err = (again - scaled.inputs).abs().amax(dim=0).detach().cpu().tolist()
    physical = (back - raw.inputs).abs().amax(dim=0).detach().cpu().tolist()
    if not all(math.isfinite(x) for x in err + physical):
        raise ValueError("non-finite standardization roundtrip")
    return {
        "max_standardized_roundtrip_per_channel": dict(zip(names, err, strict=True)),
        "max_physical_roundtrip_per_channel": dict(zip(names, physical, strict=True)),
        "max_standardized_roundtrip": max(err),
    }


@torch.inference_mode()
def heun_snapshots(
    model: torch.nn.Module,
    task: FlowMatchingTask,
    batch: Any,
    *,
    steps: int,
    seed: int,
    sampling_key: str,
    observe_steps: set[int],
) -> dict[int, torch.Tensor]:
    """M47.2-equivalent Gaussian/Heun integration, storing only selected states."""
    if steps < 1 or any(s < 0 or s > steps for s in observe_steps):
        raise ValueError("invalid integration or observation step")
    state = _deterministic_sampling_source(batch, seed, (sampling_key,))
    states = {0: state.detach().clone()} if 0 in observe_steps else {}
    dt = 1.0 / steps
    for step in range(steps):
        times = torch.full(
            (batch.num_graphs,), step / steps,
            dtype=state.dtype, device=state.device,
        )
        k1 = _model_velocity(
            model, batch, state, times, time_embedding_scale=task.time_embedding_scale,
        )
        next_times = torch.full(
            (batch.num_graphs,), (step + 1) / steps,
            dtype=state.dtype, device=state.device,
        )
        k2 = _model_velocity(
            model, batch, state + dt * k1, next_times,
            time_embedding_scale=task.time_embedding_scale,
        )
        state = state + 0.5 * dt * (k1 + k2)
        if step + 1 in observe_steps:
            states[step + 1] = state.detach().clone()
    if not torch.isfinite(state).all():
        raise ValueError("non-finite generated state")
    return states


def observation_steps(steps: int, times: tuple[float, ...]) -> tuple[int, ...]:
    if steps < 1 or not times or any(not math.isfinite(t) or t < 0 or t > 1 for t in times):
        raise ValueError("times must be finite and lie in [0,1]")
    if len(set(times)) != len(times):
        raise ValueError("duplicate requested observation times")
    return tuple(sorted({round(t * steps) for t in times}))


def aggregate_velocity(
    rows: list[dict[str, Any]], cases: tuple[str, ...], times: tuple[float, ...],
) -> dict[str, Any]:
    """Equal validation-snapshot weighting (never pooled 64^3/128^3 nodes)."""
    out = {}
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        case_out = {}
        for i, t in enumerate(times):
            per_channel = {}
            for name in CHANNELS:
                data = [row["by_time"][i]["channels"][name] for row in case_rows]
                per_channel[name] = {
                    metric: float(np.mean([r[metric] for r in data]))
                    for metric in (
                        "mse", "zero_predictor_mse", "predicted_rms", "target_rms",
                    )
                }
                per_channel[name]["improvement_over_zero_predictor"] = (
                    1.0 - per_channel[name]["mse"] / per_channel[name]["zero_predictor_mse"]
                    if per_channel[name]["zero_predictor_mse"] > 0 else None
                )
                cosines = [r["cosine_similarity"] for r in data
                           if r["cosine_similarity"] is not None]
                per_channel[name]["mean_cosine_similarity"] = (
                    float(np.mean(cosines)) if cosines else None
                )
            case_out[str(t)] = {"samples": len(case_rows), "channels": per_channel}
        out[case] = case_out
    return out


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("M47.3 requires a CUDA GPU")
    if args.validation_per_case < 1 or args.steps < 1 or args.seed < 0:
        raise ValueError("invalid validation count, sampling steps or seed")
    if args.k_bins < 1 or not 1 <= args.max_lag <= 32:
        raise ValueError("invalid spatial diagnostic configuration")
    cases = tuple(args.case) if args.case else CASE_NAMES
    if len(set(cases)) != len(cases) or any(c not in CASE_NAMES for c in cases):
        raise ValueError("case selection must contain distinct approved cases")
    if not args.no_trajectory and args.trajectory_case not in cases:
        raise ValueError("trajectory case must be among selected validation cases")
    times = tuple(args.times)
    if not times or any(not 0.0 <= t <= 1.0 or not math.isfinite(t) for t in times):
        raise ValueError("invalid Flow Matching time grid")
    if len(set(times)) != len(times):
        raise ValueError("time grid contains duplicates")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostics: {args.output_dir}")

    run_dir = args.training_root / args.condition / "train_s42"
    signature, training_summary, split = load_contract(
        run_dir=run_dir, ref_dir=args.reference_dir, mode=args.condition,
        root=args.archive_root,
    )
    dataset = AVBPMultiCaseHITDataset(
        root=args.archive_root, cases=DEFAULT_M43_CASES,
        reference_dir=args.reference_dir, require_references=True,
        expected_snapshots_per_case=100,
        selected_sample_ids=tuple(split["validation_ids"]),
    )
    if set(dataset.sample_ids) != set(split["validation_ids"]):
        raise ValueError("validation sample population does not match checkpoint")
    indices = {
        case: [i for i, sid in enumerate(dataset.sample_ids)
               if dataset.case_by_sample_id[sid] == case][:args.validation_per_case]
        for case in cases
    }
    if any(not value for value in indices.values()):
        raise ValueError("missing requested validation regime")
    scalers = _load_standardizers(run_dir / "standardizers.pt")
    if (
        not scalers.physical_nondimensionalization
        or scalers.weighting != "sample_balanced"
        or tuple(scalers.train_sample_ids) != tuple(split["train_ids"])
        or tuple(scalers.inputs.channel_names) != tuple(scalers.targets.channel_names)
        or tuple(name.split(".")[0] for name in scalers.inputs.channel_names) != CHANNELS
    ):
        raise ValueError("saved train-only M47.1 scaler is incompatible")
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
            collator([dataset[indices[cases[0]][0]]]), device=device, dtype=torch.float32,
        ))
    )
    model, _ = instantiate_controlled_model(build_recipe(), probe, seed=42)
    model = model.to(device=device, dtype=torch.float32)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if (
        checkpoint.get("signature_sha256") != signature["sha256"]
        or checkpoint.get("epoch") != training_summary["best_epoch"]
    ):
        raise ValueError("checkpoint does not match frozen M47.1 manifest")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    del checkpoint, probe

    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    max_roundtrip = 0.0
    for case in cases:
        for i in indices[case]:
            sample_id = dataset.sample_ids[i]
            raw = task_batch_to_device(
                collator([dataset[i]]), device=device, dtype=torch.float32,
            )
            scaled = scalers.transform(raw)
            audit = roundtrip_audit(raw, scaled, scalers)
            max_roundtrip = max(max_roundtrip, audit["max_standardized_roundtrip"])
            if audit["max_standardized_roundtrip"] > 1.0e-3:
                raise ValueError(f"standardization roundtrip failed: {sample_id}")
            source = _deterministic_sampling_source(
                scaled, args.seed, (f"M47.3/validation/{sample_id}",),
            )
            by_time = []
            for t in times:
                times_tensor = torch.full(
                    (scaled.num_graphs,), t, dtype=source.dtype, device=device,
                )
                problem = _flow_problem(
                    scaled, times_tensor, source,
                    time_embedding_scale=task.time_embedding_scale,
                )
                prediction = _forward(model, problem)
                by_time.append({
                    "t": t,
                    "channels": velocity_metrics(
                        prediction, problem.targets,
                        tuple(n.split(".")[0] for n in scaled.input_channels),
                    ),
                })
            rows.append({
                "sample_id": sample_id, "case": case,
                "resolution": dataset.resolution_by_sample_id[sample_id],
                "normalization": audit, "by_time": by_time,
            })
            print(f"[M47.3] mode={args.condition} validation={sample_id}", flush=True)
            del raw, scaled, source

    trajectory = None
    if not args.no_trajectory:
        case = args.trajectory_case
        i = indices[case][0]
        sample = dataset[i]
        grid = infer_cartesian_grid_3d(
            sample.mesh.coords.detach().cpu().numpy(),
            periodic_endpoint_mode="drop_max",
        )
        velocity_scale = dataset.case_definitions[case].reference_scales["U_ref"].value
        raw = task_batch_to_device(collator([sample]), device=device, dtype=torch.float32)
        scaled = scalers.transform(raw)
        observed = observation_steps(args.steps, times)
        key = f"M47.2/{case}/replicate_0000"
        states = heun_snapshots(
            model, task, scaled, steps=args.steps, seed=args.seed,
            sampling_key=key, observe_steps=set(observed),
        )
        reference = field_diagnostics(
            raw.inputs.detach().cpu().numpy(), grid, raw.input_channels,
            k_bins=args.k_bins, max_lag=args.max_lag,
            velocity_scale_m_per_s=velocity_scale,
        )
        observations = []
        for step in observed:
            state = states[step]
            physical = scalers.inputs.inverse(state, scaled.input_channels)
            diag = field_diagnostics(
                physical.detach().cpu().numpy(), grid, scaled.input_channels,
                k_bins=args.k_bins, max_lag=args.max_lag,
                velocity_scale_m_per_s=velocity_scale,
            )
            diag["relative_spectrum_l1_to_validation_reference"] = relative_spectrum_l1(
                reference["velocity_energy_spectrum"], diag["velocity_energy_spectrum"],
            )
            observations.append({
                "step": step, "actual_time": step / args.steps,
                "standardized_rms": float(torch.sqrt(torch.mean(state.double().square())).item()),
                "diagnostics": diag,
            })
        trajectory = {
            "case": case, "sample_id_used_for_geometry": dataset.sample_ids[i],
            "latent_key": key, "sampling_seed": args.seed,
            "steps": args.steps, "solver": "heun",
            "requested_times": list(times),
            "time_grid_note": "nearest integration step; actual_time is authoritative",
            "reference_type": "one validation DNS snapshot; not a paired generation target",
            "reference": reference, "observations": observations,
        }
        del raw, scaled, states

    result = {
        "version": "m47_fm_diagnostics_v1",
        "status": "complete", "mode": args.condition,
        "training_signature_sha256": signature["sha256"],
        "training_checkpoint": str(run_dir / "best.pt"),
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "population": "validation-only; no test-snapshot tuning",
        "sampled_cases": list(cases),
        "validation_samples_per_case": {case: len(indices[case]) for case in cases},
        "times": list(times), "common_gaussian_seed": args.seed,
        "max_standardized_roundtrip": max_roundtrip,
        "metrics_definition": "channel MSE over nodes; zero predictor baseline is E[(x1-x0)^2]",
        "per_case_time": aggregate_velocity(rows, cases, times),
        "trajectory": trajectory,
    }
    (args.output_dir / "details.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"[M47.3] COMPLETE: {args.output_dir / 'summary.json'}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m47_conditioned"))
    parser.add_argument("--reference-dir", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references"))
    parser.add_argument("--archive-root", type=Path, default=Path(
        "/archivenfs/phlusim/HIT_DNS_COMP"))
    parser.add_argument("--condition", choices=tuple(CONDITION_NAMES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASE_NAMES)
    parser.add_argument("--trajectory-case", choices=CASE_NAMES, default="Re54/Ma0.2")
    parser.add_argument("--no-trajectory", action="store_true")
    parser.add_argument("--validation-per-case", type=int, default=1)
    parser.add_argument("--times", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--k-bins", type=int, default=32)
    parser.add_argument("--max-lag", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
