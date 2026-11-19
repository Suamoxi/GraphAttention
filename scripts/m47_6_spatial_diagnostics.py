"""M47.6: diagnose bulk spatial physics, periodic FFT, and Flow Matching sampling.

Uses immutable M47.4 epoch10 checkpoints, original M47.1 validation fields,
and original M47.2 Gaussian sampling keys. No new training, no graph edits,
no physical constraints or projection imposed on generated fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, DEFAULT_M43_CASES
from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m47_4_evaluate import validate_checkpoint_metadata, validate_provenance
from scripts.m47_generate_ensembles import CASE_NAMES, load_contract
from scripts.m47_generation_metrics import CHANNELS, relative_spectrum_l1, unique_node_mask
from scripts.m47_reference_conditioning import CONDITION_NAMES
from scripts.m47_5_regime_diagnostics import ANCHOR_CASE, REGIME_CASES
from scripts.m47_6_spatial_metrics import (
    helmholtz_spectrum, interior_divcurl, physical_velocity,
    standardized_momentum_diagnostics,
)
from scripts.train_generative import _load_standardizers
from scripts.train_m43_b_multires import FIELDS, build_recipe

PROTOCOL = "m47_6_spatial_physics_flow_trajectory_v1"
TRIMS = (0, 2, 4, 8)
TRAJECTORY_TIMES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def spatial_sample(q: np.ndarray, grid: Any, u_ref: float, bins: int) -> dict[str, Any]:
    velocity = physical_velocity(q, grid, u_ref)
    return {
        "interior_divcurl": interior_divcurl(velocity, grid.spacing, trims=TRIMS),
        "helmholtz_raw": helmholtz_spectrum(velocity, grid, k_bins=bins),
        "helmholtz_hann": helmholtz_spectrum(velocity, grid, k_bins=bins, window="hann"),
    }


def mean_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("empty sample population")
    result: dict[str, Any] = {"count": len(records), "mean_interior_divcurl": {}}
    for trim in map(str, TRIMS):
        result["mean_interior_divcurl"][trim] = {
            name: (float(np.mean([r["interior_divcurl"][trim][name]
                                  for r in records]))
                   if all(r["interior_divcurl"][trim][name] is not None
                          for r in records) else None)
            for name in (
                "rms_divergence_s_minus_1", "rms_vorticity_s_minus_1",
                "divergence_to_vorticity_rms", "divergence_gradient_fraction",
            )
        }
    for key in ("helmholtz_raw", "helmholtz_hann"):
        spectra = [r[key] for r in records]
        k = spectra[0]["k_rad_per_m"]
        if any(s["k_rad_per_m"] != k for s in spectra):
            raise ValueError("sample spectral wavenumbers differ")
        result[key] = {
            "k_rad_per_m": k,
            "mean_fraction_dilatational_in_k_range": float(np.mean([
                s["fraction_dilatational_in_k_range"] for s in spectra
            ])),
            "mean_fraction_dilatational_all_modes": float(np.mean([
                s["fraction_dilatational_all_modes"] for s in spectra
            ])),
            "max_relative_shell_closure_error": float(max(
                s["max_relative_shell_closure_error"] for s in spectra
            )),
        }
        for name in (
            "E_total_m3_per_s2", "E_solenoidal_m3_per_s2", "E_dilatational_m3_per_s2",
        ):
            result[key][name] = np.mean(
                [s[name] for s in spectra], axis=0,
            ).tolist()
    return result


def compare_final_states(
    q50: np.ndarray, q100: np.ndarray, grid: Any, u_ref: float,
    reference_std: dict[str, float], bins: int,
) -> dict[str, Any]:
    selected = unique_node_mask(grid)
    a, b = q50[selected], q100[selected]
    if a.shape != b.shape:
        raise ValueError("comparing different final mesh sizes")
    delta = a - b
    spatial50 = spatial_sample(q50, grid, u_ref, bins)
    spatial100 = spatial_sample(q100, grid, u_ref, bins)
    sa, sb = spatial50["helmholtz_raw"], spatial100["helmholtz_raw"]
    energy_key = "E_total_m3_per_s2"
    return {
        "rms_difference_nondimensional": {
            ch: float(np.sqrt(np.mean(delta[:, i]**2)))
            for i, ch in enumerate(CHANNELS)
        },
        "rms_difference_over_dns_reference_std": {
            ch: (float(np.sqrt(np.mean(delta[:, i]**2)) / reference_std[ch])
                 if reference_std[ch] > 0 else None)
            for i, ch in enumerate(CHANNELS)
        },
        "relative_spectrum_l1_50_vs_100": relative_spectrum_l1(
            {"k_rad_per_m": sa["k_rad_per_m"], "E_m3_per_s2": sa[energy_key]},
            {"k_rad_per_m": sb["k_rad_per_m"], "E_m3_per_s2": sb[energy_key]},
        ),
        "dilatational_fraction_50": sa["fraction_dilatational_all_modes"],
        "dilatational_fraction_100": sb["fraction_dilatational_all_modes"],
        "interior_divcurl_trim4_50": spatial50["interior_divcurl"]["4"],
        "interior_divcurl_trim4_100": spatial100["interior_divcurl"]["4"],
    }


@torch.inference_mode()
def heun_trajectory(
    task: FlowMatchingTask, model: torch.nn.Module, batch: Any,
    *, steps: int, seed: int, key: str, grid: Any, k_bins: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Capture states from the canonical sampler without duplicating Heun.

    A root-module forward pre-hook records the state at the FIRST velocity
    evaluation (k1) in every selected Heun step. This cannot perturb the
    forward calculation: the hook returns None and never edits its arguments.
    Capture is removed even if sampling fails.

    Calling the sampler twice and requiring torch.allclose between runs is
    not a valid strict parity gate on CUDA sparse-attention reductions,
    which may be nondeterministic. Observing the actual canonical trajectory
    avoids that independent-execution comparison entirely.
    """
    if steps < 5 or steps % 5:
        raise ValueError("trajectory requires steps >= 5 divisible by five")
    if batch.num_graphs != 1:
        raise ValueError("trajectory requires one graph")

    calls = 0
    observations: list[dict[str, Any]] = []
    stride = steps // 5

    def observe_velocity_call(
        _module: torch.nn.Module,
        positional: tuple[Any, ...],
        _keyword: dict[str, Any],
    ) -> None:
        nonlocal calls
        # Canonical Heun evaluates k1 then k2 for each step. Observe k1
        # exclusively; the k2 predictor is not a trajectory state.
        if calls % 2 == 0:
            step = calls // 2
            if step < steps and step % stride == 0:
                if not positional or not isinstance(positional[0], torch.Tensor):
                    raise ValueError("expected canonical model(state, ...)")
                source_state = positional[0]
                row = standardized_momentum_diagnostics(
                    source_state.detach().cpu().numpy(), grid, k_bins=k_bins,
                )
                row.update({"step": step, "flow_time": step / steps})
                observations.append(row)
        calls += 1

    handle = model.register_forward_pre_hook(
        observe_velocity_call, with_kwargs=True,
    )
    try:
        state = task.sample_standardized(
            model, batch, steps=steps, method="flow_ode", solver="heun",
            sampling_eps=0.0, final_denoise=False,
            sampling_seed=seed, sampling_keys=(key,),
        )
    finally:
        handle.remove()

    if calls != 2 * steps:
        raise ValueError(
            f"expected {2 * steps} canonical Heun model calls, got {calls}"
        )
    if len(observations) != 5:
        raise ValueError(
            f"expected five observed intermediate stages, got {len(observations)}"
        )
    if not torch.isfinite(state).all():
        raise ValueError("canonical trajectory produced nonfinite state")
    final = standardized_momentum_diagnostics(
        state.detach().cpu().numpy(), grid, k_bins=k_bins,
    )
    final.update({"step": steps, "flow_time": 1.0})
    observations.append(final)
    return state, {
        "sampling_key": key,
        "seed": seed,
        "steps": steps,
        "sampler": "FlowMatchingTask.sample_standardized (unchanged)",
        "observation_method": "read-only root forward pre-hook on canonical k1",
        "observed_model_calls": calls,
        "independent_trajectory_parity_test": "not_applicable_single_canonical_run",
        "observations": observations,
        "interpretation": (
            "Intermediate states are standardized Gaussian-to-data states, "
            "NOT physically interpretable velocity fields. Spectra use "
            "standardized momentum only."
        ),
    }


@torch.inference_mode()
def execute(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("M47.6 requires an available CUDA GPU")
    if args.run_dir.name != "epoch10" or args.condition not in CONDITION_NAMES:
        raise ValueError("M47.6 requires immutable M47.4 epoch10 checkpoint")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    if args.samples < 2 or not (1 <= args.compare_count <= args.samples):
        raise ValueError("samples>=2, compare_count in [1,samples] required")
    if args.steps != 50 or args.compare_steps != 100 or args.seed < 0:
        raise ValueError("M47.6 frozen comparison requires 50 vs 100 Heun, nonnegative seed")
    if args.k_bins < 1:
        raise ValueError("k_bins must be positive")
    provenance = validate_provenance(args.run_dir, args.condition, 10)
    signature, _, split = load_contract(
        run_dir=args.run_dir, ref_dir=args.reference_dir, mode=args.condition,
        root=args.archive_root, expected_epochs=10,
    )
    dataset = AVBPMultiCaseHITDataset(
        root=args.archive_root, cases=DEFAULT_M43_CASES,
        reference_dir=args.reference_dir, require_references=True,
        expected_snapshots_per_case=100,
        selected_sample_ids=tuple(split["validation_ids"]),
    )
    if set(dataset.sample_ids) != set(split["validation_ids"]):
        raise ValueError("M47 validation split mismatch")
    by_case = {
        case: [sid for sid in dataset.sample_ids if dataset.case_by_sample_id[sid] == case]
        for case in CASE_NAMES
    }
    if any(len(ids) != 5 for ids in by_case.values()):
        raise ValueError("expected five reference validation snapshots per case")
    device = torch.device("cuda:0")
    scalers = _load_standardizers(args.run_dir / "standardizers.pt")
    if (not scalers.physical_nondimensionalization
        or scalers.weighting != "sample_balanced"
        or tuple(scalers.train_sample_ids) != tuple(split["train_ids"])
        or scalers.inputs.channel_names != scalers.targets.channel_names
        or tuple(name.split(".")[0] for name in scalers.inputs.channel_names) != CHANNELS):
        raise ValueError("standardizers differ from M47.1 train-only scaler")
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
    by_id = {sid: i for i, sid in enumerate(dataset.sample_ids)}
    probe_batch = task_batch_to_device(
        collator([dataset[by_id[by_case[REGIME_CASES[0]][0]]]]),
        device=device, dtype=torch.float32,
    )
    probe = task.make_validation_problem(scalers.transform(probe_batch))
    model, _ = instantiate_controlled_model(build_recipe(), probe, seed=42)
    model = model.to(device=device, dtype=torch.float32)
    checkpoint = torch.load(args.run_dir / "last.pt", map_location="cpu", weights_only=True)
    validate_checkpoint_metadata(checkpoint, signature_sha=signature["sha256"], epochs=10)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    del checkpoint, probe, probe_batch

    args.output_dir.mkdir(parents=True, exist_ok=False)
    trajectory = None
    convergence_rows = []
    for case in REGIME_CASES:
        ids = by_case[case]
        anchor = dataset[by_id[ids[0]]]
        grid = infer_cartesian_grid_3d(
            anchor.mesh.coords.detach().cpu().numpy(),
            periodic_endpoint_mode="drop_max",
        )
        if grid.shape != (64, 64, 64):
            raise ValueError("M47.6 currently restricted to 65-cube Re54 source")
        uref = float(dataset.case_definitions[case].reference_scales["U_ref"].value)
        references = []
        reference_std = {}
        for sid in ids:
            sample = collator([dataset[by_id[sid]]])
            q = sample.inputs.detach().cpu().numpy()
            if tuple(x.split(".")[0] for x in sample.input_channels) != CHANNELS:
                raise ValueError("unexpected field order in original DNS")
            references.append(spatial_sample(q, grid, uref, args.k_bins))
            uq = q[unique_node_mask(grid)]
            for i, ch in enumerate(CHANNELS):
                reference_std.setdefault(ch, []).append(float(np.std(uq[:, i])))
        reference_std = {ch: float(np.mean(v)) for ch, v in reference_std.items()}
        baseline = collator([anchor])
        scaled = scalers.transform(task_batch_to_device(
            baseline, device=device, dtype=torch.float32,
        ))
        if scaled.num_graphs != 1 or tuple(scaled.conditioning_names) != CONDITION_NAMES[args.condition]:
            raise ValueError("conditioning differs from frozen checkpoint")
        generated = []
        for j in range(args.samples):
            key = f"M47.2/{case}/replicate_{j:04d}"
            if case == ANCHOR_CASE and j == 0:
                q50_standard, trajectory = heun_trajectory(
                    task, model, scaled, steps=args.steps, seed=args.seed,
                    key=key, grid=grid, k_bins=args.k_bins,
                )
            else:
                q50_standard = task.sample_standardized(
                    model, scaled, steps=args.steps, method="flow_ode",
                    solver="heun", sampling_eps=0.0, final_denoise=False,
                    sampling_seed=args.seed, sampling_keys=(key,),
                )
            q50 = scalers.inputs.inverse(
                q50_standard, scaled.input_channels,
            ).detach().cpu().numpy()
            generated.append(spatial_sample(q50, grid, uref, args.k_bins))
            if case == ANCHOR_CASE and j < args.compare_count:
                q100_standard = task.sample_standardized(
                    model, scaled, steps=args.compare_steps, method="flow_ode",
                    solver="heun", sampling_eps=0.0, final_denoise=False,
                    sampling_seed=args.seed, sampling_keys=(key,),
                )
                q100 = scalers.inputs.inverse(
                    q100_standard, scaled.input_channels,
                ).detach().cpu().numpy()
                convergence_rows.append({
                    "sampling_key": key,
                    **compare_final_states(q50, q100, grid, uref,
                                           reference_std, args.k_bins),
                })
                del q100_standard, q100
            print(f"[M47.6] {args.condition} {case} {j+1}/{args.samples}", flush=True)
            del q50_standard, q50
        report = {
            "case": case, "source": "original M47.1 validation, 5 DNS frames",
            "reference_sample_ids": ids,
            "generated_sampling_keys": [
                f"M47.2/{case}/replicate_{j:04d}" for j in range(args.samples)
            ],
            "reference_velocity_scale_m_per_s": uref,
            "spacing_m": list(grid.spacing),
            "unique_fft_grid_shape": list(grid.shape),
            "boundary_exclusion_cells": list(TRIMS),
            "reference": mean_records(references),
            "generated": mean_records(generated),
            "reference_per_sample": references,
            "generated_per_sample": generated,
        }
        _json(args.output_dir / f"case_{case.replace('/', '_')}_divcurl.json", report)
        del scaled, baseline, anchor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if trajectory is None or len(convergence_rows) != args.compare_count:
        raise ValueError("trajectory or convergence comparison missing")
    _json(args.output_dir / "flow_trajectory_Re54_Ma0.2.json", trajectory)
    _json(args.output_dir / "heun_step_convergence.json", {
        "case": ANCHOR_CASE, "steps": [args.steps, args.compare_steps],
        "sampling_seed": args.seed, "matched_latent": True,
        "number_of_realizations": len(convergence_rows),
        "per_sample": convergence_rows,
        "purpose": "Assess ODE discretization sensitivity, not numerical convergence proof",
    })
    summary = {
        "protocol": PROTOCOL, "status": "complete",
        "mode": args.condition, "checkpoint": str(args.run_dir / "last.pt"),
        "original_training_revision": provenance["original_training_revision"],
        "signature_sha256": signature["sha256"],
        "evaluation_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True,
            text=True, check=True,
        ).stdout.strip(),
        "source_cases": list(REGIME_CASES),
        "source_split": "original M47.1 5 validation samples per case, no held-out test data",
        "sampling_seed": args.seed, "sampling_steps": args.steps,
        "heun_comparison_steps": args.compare_steps,
        "generation_replicates_per_case": args.samples,
        "heun_comparison_replicates": args.compare_count,
        "time_observations": list(TRAJECTORY_TIMES),
        "spatial_boundary_trims_cells": list(TRIMS),
        "notes": [
            "Raw FFT Helmholtz decomposition assumes periodicity and is seam-sensitive.",
            "Hann-window FFT projection is boundary sensitivity only, NOT physical dilatational energy.",
            "Local finite-difference divergence/curl excludes wrap stencils for trims >=2.",
            "Intermediate trajectory observations are standardized momentum, not physical states.",
            "No model weights, graph connectivity, training or data normalization changed.",
        ],
    }
    _json(args.output_dir / "summary.json", summary)
    print(f"[M47.6] COMPLETE {args.output_dir}", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--condition", choices=tuple(CONDITION_NAMES), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references"
    ))
    p.add_argument("--archive-root", type=Path,
                   default=Path("/archivenfs/phlusim/HIT_DNS_COMP"))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--compare-steps", type=int, default=100)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--compare-count", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261116)
    p.add_argument("--k-bins", type=int, default=32)
    execute(p.parse_args())


if __name__ == "__main__":
    main()
