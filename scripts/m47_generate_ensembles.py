"""M47.2 matched independent 3-D HIT ensembles and physics diagnostics.

Runs ONE of the three frozen M47.1 checkpoints. Identical Gaussian keys and
sampling settings across arms; held-out snapshots are distributional references,
NOT paired targets for generated fields. Full spatial fields are optional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import torch

from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, DEFAULT_M43_CASES
from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m47_generation_metrics import (
    CHANNELS, field_diagnostics, finite_mean, mean_curves,
    population_wasserstein, relative_spectrum_l1, sampled_nodes,
)
from scripts.m47_qualified_setup import verify_generated_references
from scripts.m47_reference_conditioning import CONDITION_NAMES
from scripts.train_generative import _load_standardizers
from scripts.train_m43_b_multires import FIELDS, M37_RECIPE, build_recipe

CASE_NAMES = tuple(case.case_id for case in DEFAULT_M43_CASES)


def load_contract(
    *, run_dir: Path, ref_dir: Path, mode: str, root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Strict model/dataset/scaler provenance before expensive GPU sampling."""
    if mode not in CONDITION_NAMES:
        raise ValueError(f"unknown condition mode: {mode}")
    signature = json.loads((run_dir / "run_signature.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    split = json.loads((run_dir / "split.json").read_text())
    if summary.get("status") != "train_complete" or summary.get("conditioning_mode") != mode:
        raise ValueError("M47.2 requires completed train_s42 checkpoint in selected arm")
    if (summary.get("conditioning_parameters") != list(CONDITION_NAMES[mode])
        or summary.get("signature_sha256") != signature.get("sha256")):
        raise ValueError("summary, conditioning and run signature disagree")
    content = signature.get("content", {})
    physics = content.get("m47_physical_conditions", {})
    if (content.get("mode") != "train" or content.get("seed") != 42
        or physics.get("condition_mode") != mode
        or physics.get("conditioning_parameters") != list(CONDITION_NAMES[mode])
        or content.get("model") != M37_RECIPE
        or content.get("time_embedding_scale") != 1000.0):
        raise ValueError("frozen M47.1 model/physical-conditioning contract changed")
    if Path(content.get("root", "")).resolve() != root.resolve():
        raise ValueError("generation dataset root differs from frozen M47.1 run")
    counts = {"train_ids": 240, "validation_ids": 30, "test_ids": 42}
    if any(len(split.get(k, [])) != count for k, count in counts.items()):
        raise ValueError("expected reviewed M47.1 chronological data partition")
    for key in counts:
        if split[key] != content.get(key):
            raise ValueError(f"{key} differs from the checkpoint source signature")
    if len(split.get("excluded_ids", [])) != 48 or split.get("temporal_embargo") != 2:
        raise ValueError("unexpected M47.1 embargo protocol")
    if len(summary.get("history", [])) != 2 or summary.get("best_epoch") != 1:
        raise ValueError("M47.2 is restricted to the validated two-epoch seed-42 pilot")
    evidence = verify_generated_references(ref_dir, run_dir / "physical_reference_audit.json")
    if evidence["m47_reference_audit_sha256"] != physics.get("m47_reference_audit_sha256"):
        raise ValueError("physical reference audit does not match frozen run signature")
    if evidence["reference_hashes"] != physics.get("reference_hashes"):
        raise ValueError("physical reference hashes differ from original training")
    if not (run_dir / "best.pt").is_file() or not (run_dir / "standardizers.pt").is_file():
        raise FileNotFoundError("M47.2 requires original best.pt and standardizers.pt")
    # Guard against inadvertent modification of source test snapshots/meshes.
    sources = {row["id"]: row for row in content.get("samples", [])}
    if len(sources) != 360:
        raise ValueError("frozen M47 source inventory must have exactly 360 snapshots")
    for sample_id in split["test_ids"]:
        source = sources[sample_id]
        for path_key, size_key, mtime_key in (
            ("path", "snapshot_size", "snapshot_mtime_ns"),
            ("mesh", "mesh_size", "mesh_mtime_ns"),
        ):
            path = Path(source[path_key]).resolve(strict=True)
            meta = path.stat()
            if meta.st_size != source[size_key] or meta.st_mtime_ns != source[mtime_key]:
                raise ValueError(f"source data changed since training: {sample_id}")
    return signature, summary, split


def _curve_report(records: list[dict], name: str, x_key: str, y_key: str) -> dict | None:
    return mean_curves(records, name, x_key, y_key)


def analyze_case(
    *, model: torch.nn.Module, task: FlowMatchingTask, scalers: Any,
    dataset: AVBPMultiCaseHITDataset, case_name: str, sample_ids: list[str],
    device: torch.device, steps: int, samples: int, seed: int,
    subsample_nodes: int, k_bins: int, max_lag: int,
    field_dir: Path | None,
) -> dict[str, Any]:
    indices = {name: i for i, name in enumerate(dataset.sample_ids)}
    selected = [indices[name] for name in sample_ids]
    collator = GraphTaskCollator(
        task, dataset.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    # Physical coordinates retained for k in rad/m; the model uses scaled
    # coordinates internally, matching training.
    anchor = dataset[selected[0]]
    grid = infer_cartesian_grid_3d(
        anchor.mesh.coords.detach().cpu().numpy(),
        periodic_endpoint_mode="drop_max",
    )
    if any(n not in (64, 128) for n in grid.shape):
        raise ValueError(f"M47.2 requires 64^3 or 128^3 unique FFT grid: {grid.shape}")
    velocity_scale = dataset.case_definitions[case_name].reference_scales["U_ref"].value
    channels = None
    reference_records = []
    reference_nodes = []
    for index, sample_id in zip(selected, sample_ids, strict=True):
        source = collator([dataset[index]])
        if channels is None:
            channels = tuple(source.input_channels)
        elif channels != tuple(source.input_channels):
            raise ValueError("reference channels changed")
        field = source.inputs.detach().cpu().numpy()
        reference_records.append(field_diagnostics(
            field, grid, channels, k_bins=k_bins, max_lag=max_lag,
            velocity_scale_m_per_s=velocity_scale,
        ))
        reference_nodes.append(sampled_nodes(
            field, grid, key=f"M47.2/reference/{sample_id}", count=subsample_nodes,
        ))
        del source

    assert channels is not None
    # One physical test snapshot provides the topology and global conditioning;
    # its actual state is NOT supplied to the generative model.
    host = collator([anchor])
    prepared = task_batch_to_device(host, device=device, dtype=torch.float32)
    scaled = scalers.transform(prepared)
    if scaled.num_graphs != 1 or tuple(scaled.conditioning_names) != tuple(CONDITION_NAMES[task_mode(task)]):
        raise ValueError("single-case conditioning did not match frozen configuration")
    if tuple(scaled.input_channels) != channels:
        raise ValueError("generated channel order does not match reference")
    generated_records = []
    generated_nodes = []
    sampling_keys = []
    for replicate in range(samples):
        key = f"M47.2/{case_name}/replicate_{replicate:04d}"
        sampling_keys.append(key)
        with torch.inference_mode():
            standardized = task.sample_standardized(
                model, scaled, steps=steps, method="flow_ode", solver="heun",
                sampling_eps=0.0, final_denoise=False,
                sampling_seed=seed, sampling_keys=(key,),
            )
            nondimensional = scalers.inputs.inverse(standardized, scaled.input_channels)
        field = nondimensional.detach().cpu().numpy()
        record = field_diagnostics(
            field, grid, channels, k_bins=k_bins, max_lag=max_lag,
            velocity_scale_m_per_s=velocity_scale,
        )
        generated_records.append(record)
        generated_nodes.append(sampled_nodes(
            field, grid, key=f"M47.2/generated/{key}", count=subsample_nodes,
        ))
        if field_dir is not None:
            folder = field_dir / case_name.replace("/", "_")
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"sample_{replicate:04d}.npz",
                                conservative_nondimensional=field.astype(np.float32))
        print(f"[M47.2] {case_name} replicate={replicate + 1}/{samples} "
              f"min_rho={record['min_density']:.6g} "
              f"rho_bad={record['nonpositive_density_fraction']:.3g}", flush=True)
        del standardized, nondimensional
    del scaled, prepared, host, anchor, collator
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ref_spectrum = _curve_report(reference_records, "velocity_energy_spectrum",
                                 "k_rad_per_m", "E_m3_per_s2")
    gen_spectrum = _curve_report(generated_records, "velocity_energy_spectrum",
                                 "k_rad_per_m", "E_m3_per_s2")
    ref_correlation = _curve_report(reference_records, "velocity_spatial_correlation",
                                    "lag_m", "vector_correlation")
    gen_correlation = _curve_report(generated_records, "velocity_spatial_correlation",
                                    "lag_m", "vector_correlation")
    correlation_l1 = None
    if ref_correlation is not None and gen_correlation is not None:
        if ref_correlation["lag_m"] != gen_correlation["lag_m"]:
            raise ValueError("inconsistent spatial-correlation grid")
        correlation_l1 = float(np.mean(np.abs(
            np.asarray(ref_correlation["vector_correlation"][1:])
            - np.asarray(gen_correlation["vector_correlation"][1:])
        )))
    return {
        "case": case_name,
        "reference_sample_ids": sample_ids,
        "generated_sampling_keys": sampling_keys,
        "unique_grid_shape": list(grid.shape),
        "source_grid_shape": list(grid.source_shape),
        "physical_grid_spacing_m": list(grid.spacing),
        "reference_velocity_scale_m_per_s": velocity_scale,
        "reference_count": len(reference_records),
        "generated_count": len(generated_records),
        "reference_density_valid_count": sum(r["density_positive_everywhere"] for r in reference_records),
        "generated_density_valid_count": sum(r["density_positive_everywhere"] for r in generated_records),
        "reference_thermodynamically_valid_count": sum(r["thermodynamically_admissible_everywhere"] for r in reference_records),
        "generated_thermodynamically_valid_count": sum(r["thermodynamically_admissible_everywhere"] for r in generated_records),
        "reference_mean_bad_density_fraction": finite_mean([r["nonpositive_density_fraction"] for r in reference_records]),
        "reference_seam_jump_over_std": {name: finite_mean([r["periodic_endpoint_jump_over_field_std"][name] for r in reference_records]) for name in CHANNELS},
        "generated_seam_jump_over_std": {name: finite_mean([r["periodic_endpoint_jump_over_field_std"][name] for r in generated_records]) for name in CHANNELS},
        "generated_mean_bad_density_fraction": finite_mean([r["nonpositive_density_fraction"] for r in generated_records]),
        "reference_mean_bad_thermo_fraction": finite_mean([r["density_or_internal_energy_invalid_fraction"] for r in reference_records]),
        "generated_mean_bad_thermo_fraction": finite_mean([r["density_or_internal_energy_invalid_fraction"] for r in generated_records]),
        "one_point_wasserstein": population_wasserstein(reference_nodes, generated_nodes),
        "reference_mean_velocity_spectrum": ref_spectrum,
        "generated_mean_velocity_spectrum": gen_spectrum,
        "relative_velocity_spectrum_l1": relative_spectrum_l1(ref_spectrum, gen_spectrum),
        "reference_mean_dilatational_energy_fraction": finite_mean([r["dilatational_energy_fraction"] for r in reference_records]),
        "generated_mean_dilatational_energy_fraction": finite_mean([r["dilatational_energy_fraction"] for r in generated_records]),
        "reference_mean_velocity_correlation": ref_correlation,
        "generated_mean_velocity_correlation": gen_correlation,
        "vector_correlation_mean_absolute_error": correlation_l1,
        "reference_per_sample": reference_records,
        "generated_per_sample": generated_records,
    }


def task_mode(task: FlowMatchingTask) -> str:
    names = tuple(task.conditioning_parameters)
    for mode, expected in CONDITION_NAMES.items():
        if names == tuple(expected):
            return mode
    raise ValueError("unexpected conditioning parameters")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.training_root / args.condition / "train_s42"
    root = args.archive_root
    _, summary, split = load_contract(
        run_dir=run_dir, ref_dir=args.reference_dir, mode=args.condition, root=root,
    )
    # Comparator certifies exactly matched model-training source inputs.
    comparison = json.loads((args.training_root / "comparison_s42.json").read_text())
    if (comparison.get("status") != "matched_three_arm_m47_comparison"
        or comparison.get("same_signature_except_condition") is not True
        or comparison.get("same_split") is not True):
        raise ValueError("M47.1 matched three-arm comparison is missing or invalid")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing M47.2 overwrite: {args.output_dir}")
    requested = tuple(args.case) if args.case else CASE_NAMES
    if not requested or any(c not in CASE_NAMES for c in requested) or len(set(requested)) != len(requested):
        raise ValueError("case selection must be unique and restricted to six approved cases")
    if args.samples_per_case < 1 or args.steps < 1 or args.k_bins < 1 or args.node_subsample < 1:
        raise ValueError("samples, steps, k-bins and node_subsample must be positive")
    if args.max_lag < 1 or args.max_lag > 32:
        raise ValueError("max_lag must lie in [1,32] for M47 meshes")
    if args.sampling_seed < 0:
        raise ValueError("sampling seed must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("M47.2 ensemble generation requires a CUDA GPU")
    dataset = AVBPMultiCaseHITDataset(
        root=root, cases=DEFAULT_M43_CASES,
        reference_dir=args.reference_dir,
        require_references=True, expected_snapshots_per_case=100,
        selected_sample_ids=tuple(split["test_ids"]),
    )
    test_ids = tuple(split["test_ids"])
    if set(test_ids) != set(dataset.sample_ids):
        raise RuntimeError("source test IDs differ from reconstructed M47.1 dataset")
    source_case_ids = {name: dataset.case_by_sample_id[name] for name in dataset.sample_ids}
    for name in test_ids:
        if source_case_ids[name] not in CASE_NAMES:
            raise ValueError("unknown test case")
    device = torch.device("cuda:0")
    scalers = _load_standardizers(run_dir / "standardizers.pt")
    if (not scalers.physical_nondimensionalization or scalers.weighting != "sample_balanced"
        or tuple(scalers.train_sample_ids) != tuple(split["train_ids"])):
        raise ValueError("frozen training scaler does not match M47.1 train partition")
    if scalers.inputs.channel_names != scalers.targets.channel_names:
        raise ValueError("FM input/target standardizer channel semantics differ")
    if tuple(name.split(".")[0] for name in scalers.inputs.channel_names) != CHANNELS:
        raise ValueError("unexpected conservative fields")
    task = FlowMatchingTask(
        state_fields=FIELDS, conditioning_parameters=CONDITION_NAMES[args.condition],
        physical_nondimensionalization=True, validation_seed=1234,
        time_embedding_scale=1000.0,
    )
    probe_index = dataset.sample_ids.index(next(
        name for name in test_ids if source_case_ids[name] == requested[0]
    ))
    collator = GraphTaskCollator(
        task, dataset.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    probe = task.make_validation_problem(scalers.transform(collator([dataset[probe_index]])))
    model, _ = instantiate_controlled_model(build_recipe(), probe, seed=42)
    del probe, collator
    model = model.to(device=device, dtype=torch.float32)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    signature = json.loads((run_dir / "run_signature.json").read_text())
    if (checkpoint.get("signature_sha256") != signature["sha256"]
        or checkpoint.get("epoch") != summary["best_epoch"]):
        raise RuntimeError("selected checkpoint is inconsistent with M47.1 provenance")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.eval()
    scalers = scalers.to(device=device, dtype=torch.float32)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected_ids_by_case = {
        case: [name for name in test_ids if source_case_ids[name] == case]
        for case in requested
    }
    result = {
        "version": "m47_independent_ensemble_v1",
        "status": "complete",
        "conditioning_mode": args.condition,
        "training_checkpoint": str(run_dir / "best.pt"),
        "training_signature_sha256": signature["sha256"],
        "best_epoch": int(summary["best_epoch"]),
        "generator_code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True,
        ).stdout.strip(),
        "held_out_reference_pairing": False,
        "gaussian_sampling_seed": args.sampling_seed,
        "sampling_steps": args.steps,
        "heun_model_evaluations_per_generated_field": 2 * args.steps,
        "samples_per_case": args.samples_per_case,
        "node_subsample_per_field": args.node_subsample,
        "energy_spectral_bins": args.k_bins,
        "spatial_correlation_max_lag": args.max_lag,
        "periodic_fft_endpoint_mode": "drop_max",
        "reference_population": "original 7 held-out test snapshots per physical regime",
        "condition_scope": "case-global Ma and/or log Re via shared adaLN-Zero",
        "physical_admissibility": "density>0 and rhoE-|rho*u|^2/(2*rho)>0 on every unique point",
        "generation_independence": "Gaussian latent independent of held-out snapshot state; model receives geometry and case reference only",
        "caveats": [
            "Only six trained regimes; no extrapolation or interpolation established.",
            "Reynolds number is confounded with 64^3/128^3 unique resolution.",
            "Periodic SOLUTBOUND graph edges not implemented at training time; explicit seam jumps are measured.",
            "Positive density/internal energy is a necessary ideal-gas proxy, not a complete AVBP EOS audit.",
            "Held-out snapshot times may be correlated; statistical confidence intervals not established.",
            "Two training epochs; model convergence has not been demonstrated.",
        ],
        "cases": {},
    }
    for case_name in requested:
        print(f"[M47.2] mode={args.condition}, case={case_name}, "
              f"steps={args.steps}, samples={args.samples_per_case}", flush=True)
        report = analyze_case(
            model=model, task=task, scalers=scalers, dataset=dataset,
            case_name=case_name, sample_ids=selected_ids_by_case[case_name],
            device=device, steps=args.steps, samples=args.samples_per_case,
            seed=args.sampling_seed, subsample_nodes=args.node_subsample,
            k_bins=args.k_bins, max_lag=args.max_lag,
            field_dir=args.output_dir / "fields" if args.store_fields else None,
        )
        result["cases"][case_name] = report
        (args.output_dir / f"case_{case_name.replace('/', '_')}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    # No giant repeated arrays in the final metadata; detailed curves live in
    # per-case JSON for subsequent paired analysis.
    concise = dict(result)
    concise["cases"] = {
        case: {k: v for k, v in report.items()
               if k not in ("reference_per_sample", "generated_per_sample")}
        for case, report in result["cases"].items()
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(concise, indent=2, allow_nan=False) + "\n"
    )
    print(f"[M47.2] COMPLETE: {args.output_dir / 'summary.json'}", flush=True)
    return concise


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
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples-per-case", type=int, default=8)
    parser.add_argument("--sampling-seed", type=int, default=20261116)
    parser.add_argument("--node-subsample", type=int, default=8192)
    parser.add_argument("--k-bins", type=int, default=32)
    parser.add_argument("--max-lag", type=int, default=8)
    parser.add_argument("--store-fields", action="store_true")
    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
