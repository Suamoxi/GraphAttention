"""M47.5: epoch-10 regime diagnostics and controlled conditioning interventions.

No training, topology modification or checkpoint rewriting. Original M47.1
validation fields are DNS references; generated Gaussian fields remain
independent of those snapshots. Native cases use their own reference scales.
The controlled interventions retain Re54/Ma0.2 geometry and scales and change
only a model conditioning vector. They are sensitivity tests, NOT independent
physical-regime generation with appropriate case-specific units.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch

from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, DEFAULT_M43_CASES
from graph_attention.evaluation.spectra_3d import CartesianGrid3D, infer_cartesian_grid_3d
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m47_4_evaluate import validate_checkpoint_metadata, validate_provenance
from scripts.m47_generate_ensembles import CASE_NAMES, analyze_case, load_contract
from scripts.m47_generation_metrics import (
    CHANNELS, field_diagnostics, finite_mean, mean_curves, relative_spectrum_l1,
    unique_node_mask,
)
from scripts.m47_reference_conditioning import CONDITION_NAMES
from scripts.train_generative import _load_standardizers
from scripts.train_m43_b_multires import FIELDS, build_recipe


REGIME_CASES = ("Re54/Ma0.05", "Re54/Ma0.2", "Re54/Ma0.3")
ANCHOR_CASE = "Re54/Ma0.2"
RE_PROBE_CASE = "Re94/Ma0.2"
PROTOCOL = "m47_5_regime_structure_condition_sensitivity_v1"


def unique_grid_wrap_ratio(state: np.ndarray, grid: CartesianGrid3D) -> dict[str, float | None]:
    """RMS unique-FFT-grid wrap jump divided by RMS interior neighbour jump.

    The duplicated maximum-coordinate planes are deliberately excluded, as
    in the existing FFT spectra. This is NOT the existing duplicate-endpoint
    mismatch: it tests the discontinuity directly seen by periodic FFTs.
    """
    if grid.periodic_endpoint_mode != "drop_max":
        raise ValueError("unique wrap diagnostic requires drop_max periodic grid")
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (int(np.prod(grid.source_shape)), len(CHANNELS)):
        raise ValueError("unexpected conservative field shape")
    full = np.empty((*grid.source_shape, len(CHANNELS)), dtype=np.float64)
    full.reshape(-1, len(CHANNELS))[grid.linear_indices] = values
    unique = full[tuple(slice(0, side) for side in grid.shape)]
    wrap_square = np.zeros(len(CHANNELS), dtype=np.float64)
    interior_square = np.zeros(len(CHANNELS), dtype=np.float64)
    wrap_count = interior_count = 0
    for axis in range(3):
        jump = np.take(unique, 0, axis=axis) - np.take(unique, -1, axis=axis)
        interior = np.diff(unique, axis=axis)
        wrap_square += np.sum(jump * jump, axis=(0, 1))
        interior_square += np.sum(interior * interior, axis=(0, 1, 2))
        wrap_count += int(np.prod(jump.shape[:-1]))
        interior_count += int(np.prod(interior.shape[:-1]))
    wrms = np.sqrt(wrap_square / wrap_count)
    irms = np.sqrt(interior_square / interior_count)
    return {
        name: (float(wrms[i] / irms[i]) if irms[i] > 0 else None)
        for i, name in enumerate(CHANNELS)
    }


def cross_domain_edge_audit(
    coordinates: np.ndarray, topologies: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Count explicit local/dilated graph edges crossing opposite physical faces.

    A zero count is evidence that existing edges do not directly connect
    opposite coordinate planes, not a proof of a defective periodic solver.
    """
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("expected physical coordinates [N,3]")
    span = np.ptp(xyz, axis=0)
    if np.any(span <= 0):
        raise ValueError("degenerate mesh extent")
    result = {}
    for name, edge_tensor in topologies.items():
        edges = np.asarray(edge_tensor, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("expected directed edges [2,E]")
        if edges.size and (edges.min() < 0 or edges.max() >= len(xyz)):
            raise ValueError("graph edges refer to missing nodes")
        counts = np.zeros(3, dtype=np.int64)
        for start in range(0, edges.shape[1], 262144):
            chunk = edges[:, start:start + 262144]
            jumps = np.abs(xyz[chunk[0]] - xyz[chunk[1]])
            counts += np.sum(jumps > (0.75 * span)[None, :], axis=0)
        result[name] = {
            "directed_edges": int(edges.shape[1]),
            "full_span_crossing_counts_xyz": counts.tolist(),
            "criterion": "absolute coordinate jump > 0.75 * domain extent",
        }
    return result


def intervention_plan(
    mode: str, references: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Change one audited conditioning variable at a time, never geometry."""
    names = CONDITION_NAMES[mode]
    if not names:
        return []
    anchor = dict(references[ANCHOR_CASE])
    if set(anchor) != set(names):
        raise ValueError("anchor conditioning does not match model condition names")
    plan = []
    for case, suffix in (("Re54/Ma0.05", "low"), ("Re54/Ma0.3", "high")):
        modified = dict(anchor)
        modified["Ma_condition"] = references[case]["Ma_condition"]
        plan.append({
            "id": f"mach_{suffix}",
            "reference_values_taken_from": case,
            "changed_names": ["Ma_condition"],
            "conditioning": modified,
        })
    if mode == "mach_re":
        modified = dict(anchor)
        modified["log_Re_condition"] = references[RE_PROBE_CASE]["log_Re_condition"]
        plan.append({
            "id": "re_high_fixed_ma",
            "reference_values_taken_from": RE_PROBE_CASE,
            "changed_names": ["log_Re_condition"],
            "conditioning": modified,
            "out_of_training_support_note": (
                "Uses the Re94 Reynolds descriptor on the Re54 65-cube "
                "geometry; cannot assess physical generalization in isolation."
            ),
        })
    return plan


def _reference_condition_values(
    dataset: AVBPMultiCaseHITDataset, mode: str,
) -> dict[str, dict[str, float]]:
    names = CONDITION_NAMES[mode]
    return {
        case: {
            name: float(dataset.case_definitions[case].regime_parameters[name].value)
            for name in names
        }
        for case in (*REGIME_CASES, RE_PROBE_CASE)
    }


def _case_periodicity(
    *, dataset: AVBPMultiCaseHITDataset, case: str, ids: list[str],
    collator: GraphTaskCollator, saved_fields: Path,
    samples: int,
) -> dict[str, Any]:
    by_id = {sid: i for i, sid in enumerate(dataset.sample_ids)}
    anchor = dataset[by_id[ids[0]]]
    grid = infer_cartesian_grid_3d(
        anchor.mesh.coords.detach().cpu().numpy(),
        periodic_endpoint_mode="drop_max",
    )
    reference = [
        unique_grid_wrap_ratio(
            collator([dataset[by_id[sid]]]).inputs.detach().cpu().numpy(), grid,
        )
        for sid in ids
    ]
    generated = []
    for replicate in range(samples):
        file = saved_fields / case.replace("/", "_") / f"sample_{replicate:04d}.npz"
        with np.load(file) as data:
            generated.append(unique_grid_wrap_ratio(
                data["conservative_nondimensional"], grid,
            ))
    host = collator([anchor])
    topologies = {
        "one_hop": host.edge_index.detach().cpu().numpy(),
        "exact_two_hop": host.attention_edge_indices["dilated"].detach().cpu().numpy(),
    }
    topology = cross_domain_edge_audit(
        anchor.mesh.coords.detach().cpu().numpy(), topologies,
    )
    return {
        "metric_definition": (
            "FFT unique-grid RMS first/last-face jump divided by "
            "RMS interior nearest-neighbour jump; separate from duplicate "
            "stored-endpoint mismatch."
        ),
        "reference_mean": {ch: finite_mean([row[ch] for row in reference])
                           for ch in CHANNELS},
        "generated_mean": {ch: finite_mean([row[ch] for row in generated])
                           for ch in CHANNELS},
        "reference_per_sample": reference,
        "generated_per_sample": generated,
        "original_graph_topology": topology,
        "topology_modified": False,
    }


def _paired_change(
    field: np.ndarray, baseline: np.ndarray, grid: CartesianGrid3D,
    reference_stds: dict[str, float],
) -> dict[str, float | None]:
    selected = unique_node_mask(grid)
    diff = np.asarray(field, dtype=np.float64)[selected] - np.asarray(
        baseline, dtype=np.float64,
    )[selected]
    rms = np.sqrt(np.mean(diff * diff, axis=0))
    return {
        ch: (float(rms[i] / reference_stds[ch]) if reference_stds[ch] > 0 else None)
        for i, ch in enumerate(CHANNELS)
    }


@torch.inference_mode()
def _controlled_sensitivity(
    *, model: torch.nn.Module, task: FlowMatchingTask, scalers: Any,
    dataset: AVBPMultiCaseHITDataset, collator: GraphTaskCollator,
    mode: str, native: dict[str, Any], saved_fields: Path,
    device: torch.device, samples: int, steps: int, seed: int,
    k_bins: int, max_lag: int,
) -> dict[str, Any]:
    references = _reference_condition_values(dataset, mode)
    plan = intervention_plan(mode, references)
    if not plan:
        return {"status": "not_applicable", "reason": "unconditioned model"}
    by_id = {sid: i for i, sid in enumerate(dataset.sample_ids)}
    anchor_id = native[ANCHOR_CASE]["reference_sample_ids"][0]
    anchor = dataset[by_id[anchor_id]]
    physical_grid = infer_cartesian_grid_3d(
        anchor.mesh.coords.detach().cpu().numpy(),
        periodic_endpoint_mode="drop_max",
    )
    host = collator([anchor])
    prepared = task_batch_to_device(host, device=device, dtype=torch.float32)
    scaled = scalers.transform(prepared)
    if scaled.num_graphs != 1 or tuple(scaled.conditioning_names) != CONDITION_NAMES[mode]:
        raise ValueError("baseline scaled conditioning differs from checkpoint mode")
    anchor_values = references[ANCHOR_CASE]
    for i, name in enumerate(scaled.conditioning_names):
        if not np.isclose(
            float(scaled.conditioning[0, i].item()),
            anchor_values[name], rtol=2e-6, atol=2e-6,
        ):
            raise ValueError(f"baseline conditioning value {name} differs from YAML")
    if scaled.input_channels != scalers.inputs.channel_names:
        raise ValueError("mismatched standardizer input channels")
    velocity_scale = dataset.case_definitions[ANCHOR_CASE].reference_scales["U_ref"].value
    baseline_fields = []
    for i in range(samples):
        path = saved_fields / ANCHOR_CASE.replace("/", "_") / f"sample_{i:04d}.npz"
        with np.load(path) as data:
            baseline_fields.append(data["conservative_nondimensional"].copy())
    reference_stds = {
        ch: float(np.mean([row["field_std"][ch]
                           for row in native[ANCHOR_CASE]["reference_per_sample"]]))
        for ch in CHANNELS
    }
    baseline_spectrum = native[ANCHOR_CASE]["generated_mean_velocity_spectrum"]
    results = []
    for variant in plan:
        vector = [variant["conditioning"][name] for name in CONDITION_NAMES[mode]]
        modified = replace(
            scaled,
            conditioning=torch.tensor(
                [vector], dtype=scaled.conditioning.dtype, device=device,
            ),
        )
        records, changes, wrap = [], [], []
        for replicate in range(samples):
            # Identical key to native Re54/Ma0.2: same Gaussian x0 and geometry.
            key = f"M47.2/{ANCHOR_CASE}/replicate_{replicate:04d}"
            standardized = task.sample_standardized(
                model, modified, steps=steps, method="flow_ode", solver="heun",
                sampling_eps=0.0, final_denoise=False,
                sampling_seed=seed, sampling_keys=(key,),
            )
            state = scalers.inputs.inverse(
                standardized, modified.input_channels,
            ).detach().cpu().numpy()
            records.append(field_diagnostics(
                state, physical_grid, modified.input_channels,
                k_bins=k_bins, max_lag=max_lag,
                velocity_scale_m_per_s=velocity_scale,
            ))
            changes.append(_paired_change(
                state, baseline_fields[replicate], physical_grid, reference_stds,
            ))
            wrap.append(unique_grid_wrap_ratio(state, physical_grid))
            print(
                f"[M47.5] mode={mode} intervention={variant['id']} "
                f"replicate={replicate + 1}/{samples}", flush=True,
            )
            del standardized, state
        spectrum = mean_curves(
            records, "velocity_energy_spectrum", "k_rad_per_m", "E_m3_per_s2",
        )
        report = dict(variant)
        report.update({
            "samples": samples,
            "paired_normalized_rms_change": {
                ch: finite_mean([row[ch] for row in changes]) for ch in CHANNELS
            },
            "paired_normalized_rms_change_per_sample": changes,
            "mean_field_mean": {
                ch: finite_mean([row["field_mean"][ch] for row in records])
                for ch in CHANNELS
            },
            "mean_field_std": {
                ch: finite_mean([row["field_std"][ch] for row in records])
                for ch in CHANNELS
            },
            "mean_invalid_thermodynamic_fraction": finite_mean([
                row["density_or_internal_energy_invalid_fraction"] for row in records
            ]),
            "density_valid_count": sum(row["density_positive_everywhere"]
                                       for row in records),
            "thermodynamically_valid_count": sum(
                row["thermodynamically_admissible_everywhere"] for row in records
            ),
            "mean_dilatational_energy_fraction": finite_mean([
                row["dilatational_energy_fraction"] for row in records
            ]),
            "mean_unique_wrap_ratio": {
                ch: finite_mean([row[ch] for row in wrap]) for ch in CHANNELS
            },
            "mean_duplicate_endpoint_jump_over_std": {
                ch: finite_mean([
                    row["periodic_endpoint_jump_over_field_std"][ch]
                    for row in records
                ]) for ch in CHANNELS
            },
            "velocity_spectrum_l1_to_fixed_geometry_baseline":
                relative_spectrum_l1(baseline_spectrum, spectrum),
            "mean_velocity_spectrum": spectrum,
            "generated_per_sample": records,
        })
        results.append(report)
    return {
        "status": "complete",
        "design": (
            "Counterfactual interventions: fixed Re54/Ma0.2 geometry, "
            "convective scales, standardizers and Gaussian latent per replicate; "
            "change one audited model-conditioning input at a time."
        ),
        "physical_consistency_warning": (
            "These are sensitivity probes, NOT DNS-equivalent distributions "
            "for other physical regimes. In particular re_high_fixed_ma "
            "uses a Re94 descriptor with the Re54 65-cube mesh."
        ),
        "baseline_case": ANCHOR_CASE,
        "baseline_conditioning": references[ANCHOR_CASE],
        "baseline_sampling_keys": native[ANCHOR_CASE]["generated_sampling_keys"],
        "baseline_mean_field_mean": {
            ch: finite_mean([
                row["field_mean"][ch] for row in
                native[ANCHOR_CASE]["generated_per_sample"]
            ]) for ch in CHANNELS
        },
        "reference_condition_vectors": references,
        "interventions": results,
    }


@torch.inference_mode()
def execute(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("M47.5 requires a CUDA device")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing overwrite: {args.output_dir}")
    if args.steps < 1 or args.samples < 1 or args.seed < 0:
        raise ValueError("invalid sample count, step count or sampling seed")
    if args.node_subsample < 1 or args.k_bins < 1 or not 1 <= args.max_lag <= 32:
        raise ValueError("invalid physical diagnostic settings")
    if args.condition not in CONDITION_NAMES:
        raise ValueError("unknown conditioning mode")
    if args.run_dir.name != "epoch10":
        raise ValueError("M47.5 only evaluates the frozen epoch10 last.pt")
    provenance = validate_provenance(args.run_dir, args.condition, 10)
    signature, _, split = load_contract(
        run_dir=args.run_dir, ref_dir=args.reference_dir,
        mode=args.condition, root=args.archive_root, expected_epochs=10,
    )
    dataset = AVBPMultiCaseHITDataset(
        root=args.archive_root, cases=DEFAULT_M43_CASES,
        reference_dir=args.reference_dir, require_references=True,
        expected_snapshots_per_case=100,
        selected_sample_ids=tuple(split["validation_ids"]),
    )
    if set(dataset.sample_ids) != set(split["validation_ids"]):
        raise ValueError("original M47 validation split mismatch")
    ids = {
        case: [sid for sid in dataset.sample_ids
               if dataset.case_by_sample_id[sid] == case]
        for case in CASE_NAMES
    }
    if any(len(items) != 5 for items in ids.values()):
        raise ValueError("expected five M47.1 validation snapshots in each regime")
    device = torch.device("cuda:0")
    scalers = _load_standardizers(args.run_dir / "standardizers.pt")
    if (not scalers.physical_nondimensionalization
        or scalers.weighting != "sample_balanced"
        or tuple(scalers.train_sample_ids) != tuple(split["train_ids"])
        or scalers.inputs.channel_names != scalers.targets.channel_names
        or tuple(ch.split(".")[0] for ch in scalers.inputs.channel_names) != CHANNELS):
        raise ValueError("M47.5 requires the original train-only M47 standardizers")
    scalers = scalers.to(device=device, dtype=torch.float32)
    task = FlowMatchingTask(
        state_fields=FIELDS,
        conditioning_parameters=CONDITION_NAMES[args.condition],
        physical_nondimensionalization=True, validation_seed=1234,
        time_embedding_scale=1000.0,
    )
    collator = GraphTaskCollator(
        task, dataset.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    by_id = {sid: i for i, sid in enumerate(dataset.sample_ids)}
    probe_batch = task_batch_to_device(
        collator([dataset[by_id[ids[REGIME_CASES[0]][0]]]]),
        device=device, dtype=torch.float32,
    )
    probe = task.make_validation_problem(scalers.transform(probe_batch))
    model, _ = instantiate_controlled_model(build_recipe(), probe, seed=42)
    model = model.to(device=device, dtype=torch.float32)
    checkpoint = torch.load(
        args.run_dir / "last.pt", map_location="cpu", weights_only=True,
    )
    validate_checkpoint_metadata(
        checkpoint, signature_sha=signature["sha256"], epochs=10,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    del checkpoint, probe, probe_batch

    args.output_dir.mkdir(parents=True, exist_ok=False)
    native = {}
    with TemporaryDirectory(prefix="native_fields_", dir=args.output_dir) as tmp:
        saved_fields = Path(tmp)
        for case in REGIME_CASES:
            print(
                f"[M47.5] mode={args.condition} native_case={case} "
                f"steps={args.steps} samples={args.samples}", flush=True,
            )
            report = analyze_case(
                model=model, task=task, scalers=scalers, dataset=dataset,
                case_name=case, sample_ids=ids[case], device=device,
                steps=args.steps, samples=args.samples, seed=args.seed,
                subsample_nodes=args.node_subsample, k_bins=args.k_bins,
                max_lag=args.max_lag, field_dir=saved_fields,
            )
            report["periodicity_audit"] = _case_periodicity(
                dataset=dataset, case=case, ids=ids[case], collator=collator,
                saved_fields=saved_fields, samples=args.samples,
            )
            native[case] = report
            (args.output_dir / f"case_{case.replace('/', '_')}.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
        sensitivity = _controlled_sensitivity(
            model=model, task=task, scalers=scalers, dataset=dataset,
            collator=collator, mode=args.condition, native=native,
            saved_fields=saved_fields, device=device, samples=args.samples,
            steps=args.steps, seed=args.seed, k_bins=args.k_bins,
            max_lag=args.max_lag,
        )
        (args.output_dir / "controlled_conditioning.json").write_text(
            json.dumps(sensitivity, indent=2, allow_nan=False) + "\n"
        )
        if args.store_fields:
            # Explicit opt-in only; full fields are not required for JSON review.
            import shutil
            shutil.copytree(saved_fields, args.output_dir / "fields")

    summary = {
        "protocol": PROTOCOL,
        "status": "complete",
        "conditioning_mode": args.condition,
        "checkpoint": str(args.run_dir / "last.pt"),
        "checkpoint_kind": "fixed epoch10 last.pt",
        "signature_sha256": signature["sha256"],
        "original_training_revision": provenance["original_training_revision"],
        "evaluation_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "population": "original M47.1 validation IDs; five per Re54 Mach case",
        "test_snapshot_state_used": False,
        "native_cases": list(REGIME_CASES),
        "native_reference_policy": "case-specific audited physical reference and mesh",
        "native_generation_sampling_key": "M47.2/{case}/replicate_{index:04d}",
        "seed": args.seed, "heun_steps": args.steps,
        "samples_per_case": args.samples,
        "node_subsample": args.node_subsample,
        "k_bins": args.k_bins, "max_lag": args.max_lag,
        "controlled_sensitivity_status": sensitivity["status"],
        "topology_modified": False,
        "training_modified": False,
        "native_generation": {
            case: {k: v for k, v in report.items()
                   if k not in ("reference_per_sample", "generated_per_sample")}
            for case, report in native.items()
        },
        "caveats": [
            "Re54 is a case family; actual Ma_ref and log_Re_ref are read from audited YAML.",
            "Conditioning interventions retain anchor geometry/scales; they are not physical generalization tests.",
            "Three Mach cases are in the original training population; this is interpolation of observed regimes, not unseen-regime generalization.",
            "FFT diagnostics discard stored maximum-coordinate planes; duplicate endpoint discontinuities are reported separately.",
            "The original model did not receive periodic SOLUTBOUND edges during training.",
            "No training, physical scales, checkpoint tensors or graph connectivity were changed.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(f"[M47.5] COMPLETE: {args.output_dir / 'summary.json'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITION_NAMES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir", type=Path,
        default=Path("/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references"),
    )
    parser.add_argument(
        "--archive-root", type=Path,
        default=Path("/archivenfs/phlusim/HIT_DNS_COMP"),
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--node-subsample", type=int, default=8192)
    parser.add_argument("--k-bins", type=int, default=32)
    parser.add_argument("--max-lag", type=int, default=8)
    parser.add_argument("--store-fields", action="store_true")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
