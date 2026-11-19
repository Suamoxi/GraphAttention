"""M47.7A: read-only directional/cross-component covariance audit.

Reconstruct frozen M47.4 epoch10 DiNAT-DiT using its original M47.1 split,
train-only scaler and audited physical case references. Compare five validation
DNS fields with four independent canonical Flow Matching realizations in each
Re54 Mach regime. No training, graph edits, projection or parameter changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, DEFAULT_M43_CASES
from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m47_4_evaluate import validate_checkpoint_metadata, validate_provenance
from scripts.m47_5_regime_diagnostics import REGIME_CASES
from scripts.m47_6_spatial_metrics import physical_velocity
from scripts.m47_7_covariance_metrics import spectral_tensor_audit, aggregate_spectral_tensor
from scripts.m47_generate_ensembles import CASE_NAMES, load_contract
from scripts.m47_generation_metrics import CHANNELS
from scripts.m47_reference_conditioning import CONDITION_NAMES
from scripts.train_generative import _load_standardizers
from scripts.train_m43_b_multires import FIELDS, build_recipe

PROTOCOL = "m47_7a_cross_component_spectral_covariance_v1"


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _case_ids(dataset, split, case: str) -> list[str]:
    ids = [
        sid for sid in split["validation_ids"]
        if dataset.case_by_sample_id[sid] == case
    ]
    if len(ids) != 5:
        raise ValueError(f"{case}: expected five original validation IDs, got {len(ids)}")
    return ids


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("M47.7A requires CUDA")
    if args.condition not in CONDITION_NAMES:
        raise ValueError("invalid conditioning mode")
    if args.run_dir.name != "epoch10" or args.output_dir.exists():
        raise ValueError("M47.7A requires frozen epoch10 and a new output directory")
    if args.steps != 50 or args.samples != 4 or args.seed != 20261116 or args.bins != 32:
        raise ValueError("use canonical M47.7A 50-step/4-generation/32-bin protocol")

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
        raise ValueError("validation split differs from frozen M47.1 manifest")
    for case in CASE_NAMES:
        _case_ids(dataset, split, case)
    scalers = _load_standardizers(args.run_dir / "standardizers.pt")
    if (
        not scalers.physical_nondimensionalization or
        scalers.weighting != "sample_balanced" or
        tuple(scalers.train_sample_ids) != tuple(split["train_ids"]) or
        tuple(s.split(".")[0] for s in scalers.inputs.channel_names) != CHANNELS
    ):
        raise ValueError("original M47 scaler provenance mismatch")
    device = torch.device("cuda:0")
    scalers = scalers.to(device=device, dtype=torch.float32)
    task = FlowMatchingTask(
        state_fields=FIELDS, conditioning_parameters=CONDITION_NAMES[args.condition],
        physical_nondimensionalization=True,
        validation_seed=1234, time_embedding_scale=1000.0,
    )
    collator = GraphTaskCollator(
        task, dataset.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    index = {sid: i for i, sid in enumerate(dataset.sample_ids)}
    probe = task.make_validation_problem(scalers.transform(
        task_batch_to_device(
            collator([dataset[index[_case_ids(dataset, split, REGIME_CASES[0])[0]]]]),
            device=device, dtype=torch.float32,
        )
    ))
    model, _ = instantiate_controlled_model(build_recipe(), probe, seed=42)
    model = model.to(device=device, dtype=torch.float32)
    checkpoint = torch.load(args.run_dir / "last.pt", map_location="cpu", weights_only=True)
    validate_checkpoint_metadata(checkpoint, signature_sha=signature["sha256"], epochs=10)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    del checkpoint, probe

    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = {}
    for case in REGIME_CASES:
        ids = _case_ids(dataset, split, case)
        anchor = dataset[index[ids[0]]]
        grid = infer_cartesian_grid_3d(
            anchor.mesh.coords.detach().cpu().numpy(),
            periodic_endpoint_mode="drop_max",
        )
        uref = float(dataset.case_definitions[case].reference_scales["U_ref"].value)
        reference = []
        for sid in ids:
            host = collator([dataset[index[sid]]])
            if tuple(s.split(".")[0] for s in host.input_channels) != CHANNELS:
                raise ValueError("unexpected conservative channel ordering")
            ref_vel = physical_velocity(host.inputs.detach().cpu().numpy(), grid, uref)
            reference.append(spectral_tensor_audit(ref_vel, grid, bins=args.bins))
        host = collator([anchor])
        prepared = scalers.transform(task_batch_to_device(
            host, device=device, dtype=torch.float32,
        ))
        if prepared.num_graphs != 1 or (
            tuple(prepared.conditioning_names) != tuple(CONDITION_NAMES[args.condition])
        ):
            raise ValueError("wrong conditioning for frozen checkpoint")
        generated = []
        keys = []
        for j in range(args.samples):
            key = f"M47.2/{case}/replicate_{j:04d}"
            keys.append(key)
            result = task.sample_standardized(
                model, prepared, steps=args.steps, method="flow_ode", solver="heun",
                sampling_eps=0.0, final_denoise=False,
                sampling_seed=args.seed, sampling_keys=(key,),
            )
            q = scalers.inputs.inverse(result, prepared.input_channels)
            generated.append(spectral_tensor_audit(
                physical_velocity(q.detach().cpu().numpy(), grid, uref),
                grid, bins=args.bins,
            ))
            print(f"[M47.7A] {args.condition} {case} {j+1}/{args.samples}", flush=True)
            del q, result
        refmean, genmean = (
            aggregate_spectral_tensor(reference), aggregate_spectral_tensor(generated)
        )
        rE = np.asarray(refmean["E_total_m3_per_s2"])
        gE = np.asarray(genmean["E_total_m3_per_s2"])
        cases[case] = {
            "case": case,
            "source": "M47.1 original five validation DNS frames",
            "reference_sample_ids": ids, "generation_sampling_keys": keys,
            "mesh_source_shape": list(grid.source_shape),
            "mesh_unique_fft_shape": list(grid.shape),
            "spacing_m": list(grid.spacing),
            "reference_velocity_scale_m_per_s": uref,
            "reference": refmean, "generated": genmean,
            "relative_total_spectrum_l1": float(
                np.sum(np.abs(gE-rE))/max(np.sum(np.abs(rE)),1e-30)
            ),
            "reference_per_sample": reference,
            "generated_per_sample": generated,
        }
        _write(args.output_dir / f"case_{case.replace('/', '_')}_covariance.json", cases[case])
        del prepared, host, anchor
        torch.cuda.empty_cache()

    summary = {
        "protocol": PROTOCOL, "status": "complete",
        "source": "M47.4 epoch10", "conditioning_mode": args.condition,
        "training_revision": provenance["original_training_revision"],
        "evaluation_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "checkpoint": str(args.run_dir / "last.pt"),
        "signature_sha256": signature["sha256"],
        "data_partition": "frozen original M47.1 validation, no test data",
        "samples_per_case": args.samples,
        "dns_reference_snapshots_per_case": 5,
        "heun_steps": args.steps, "sampling_seed": args.seed, "k_bins": args.bins,
        "cases": {
            case: {
                "dns_longitudinal_fraction":
                    row["reference"]["mean_longitudinal_energy_fraction_integrated"],
                "generated_longitudinal_fraction":
                    row["generated"]["mean_longitudinal_energy_fraction_integrated"],
                "dns_cross_cancellation":
                    row["reference"]["mean_cross_cancellation_integrated"],
                "generated_cross_cancellation":
                    row["generated"]["mean_cross_cancellation_integrated"],
                "relative_total_spectrum_l1": row["relative_total_spectrum_l1"],
            } for case, row in cases.items()
        },
        "limitations": [
            "Training regimes all present; this is not held-out regime generalization.",
            "Periodic Fourier decomposition is seam-sensitive.",
            "Real off-diagonal cospectra can cancel under isotropy; weighted cross "
            "terms and directional longitudinal energy are the primary tests.",
            "No weights, normalization or topology were modified.",
        ],
    }
    _write(args.output_dir / "summary.json", summary)
    print(f"[M47.7A] COMPLETE {args.output_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITION_NAMES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references"
    ))
    parser.add_argument("--archive-root", type=Path, default=Path(
        "/archivenfs/phlusim/HIT_DNS_COMP"
    ))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--bins", type=int, default=32)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
