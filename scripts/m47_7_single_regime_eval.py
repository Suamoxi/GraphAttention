"""M47.7B: held-validation covariance/physics generation at epoch 10 or 30.

Generates from the independent Gaussian source using identical M47.2 keys and
the frozen single-regime train-only scaler; no test data, retraining or
checkpoint selection by generative metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, HITCaseSpec
from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m47_5_regime_diagnostics import ANCHOR_CASE
from scripts.m47_6_spatial_diagnostics import spatial_sample, mean_records
from scripts.m47_6_spatial_metrics import physical_velocity
from scripts.m47_7_covariance_metrics import spectral_tensor_audit, aggregate_spectral_tensor
from scripts.m47_7_covariance_audit import _write
from scripts.m47_generation_metrics import CHANNELS, unique_node_mask
from scripts.train_generative import _load_standardizers
from scripts.train_m43_b_multires import FIELDS, build_recipe

PROTOCOL = "m47_7b_single_regime_covariance_evaluation_v1"


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("M47.7B evaluation requires CUDA")
    if args.epoch not in (10, 30) or args.output_dir.exists():
        raise ValueError("evaluate only epoch10 or epoch30 into new directories")
    if args.steps != 50 or args.samples != 4 or args.seed != 20261116:
        raise ValueError("M47.7B requires fixed 50-step/4-sample M47.2 keys")
    origin = json.loads((args.train_dir / "run_signature.json").read_text())
    split = json.loads((args.train_dir / "split.json").read_text())
    if split["case"] != ANCHOR_CASE:
        raise ValueError("single case ID mismatch")
    saved = torch.load(
        args.train_dir / f"epoch{args.epoch:02d}.pt",
        map_location="cpu", weights_only=True,
    )
    if (saved["signature_sha256"] != origin["sha256"]
        or int(saved["epoch"]) != args.epoch - 1
        or len(saved["history"]) != args.epoch):
        raise ValueError("checkpoint does not match frozen single-regime training stage")
    if origin["content"]["case"] != ANCHOR_CASE or origin["content"]["source_distribution"] != "independent_unit_Gaussian_standardized":
        raise ValueError("unexpected trained model provenance")
    if tuple(split["train_ids"]) != tuple(origin["content"]["train_ids"]) or (
        tuple(split["validation_ids"]) != tuple(origin["content"]["validation_ids"])
    ):
        raise ValueError("evaluation split mismatch")
    val_ids = list(split["validation_ids"])
    if len(val_ids) != 5:
        raise ValueError("require original five M47.1 validation DNS snapshots")
    data = AVBPMultiCaseHITDataset(
        root=args.archive_root, cases=(HITCaseSpec(ANCHOR_CASE, 65),),
        reference_dir=args.reference_dir, require_references=True,
        expected_snapshots_per_case=100,
        selected_sample_ids=val_ids,
    )
    if set(data.sample_ids) != set(val_ids):
        raise ValueError("missing required validation snapshots")
    idx = {sid: i for i, sid in enumerate(data.sample_ids)}
    scalers = _load_standardizers(args.train_dir / "standardizers.pt")
    if (
        not scalers.physical_nondimensionalization or
        tuple(scalers.train_sample_ids) != tuple(split["train_ids"]) or
        scalers.weighting != "sample_balanced" or
        tuple(s.split(".")[0] for s in scalers.inputs.channel_names) != CHANNELS
    ):
        raise ValueError("single-case scaler trained on wrong samples")
    device = torch.device("cuda:0")
    scalers = scalers.to(device=device, dtype=torch.float32)
    task = FlowMatchingTask(
        state_fields=FIELDS, conditioning_parameters=(),
        physical_nondimensionalization=True,
        validation_seed=1234, time_embedding_scale=1000.0,
    )
    collator = GraphTaskCollator(
        task, data.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    anchor = data[idx[val_ids[0]]]
    grid = infer_cartesian_grid_3d(
        anchor.mesh.coords.detach().cpu().numpy(),
        periodic_endpoint_mode="drop_max",
    )
    uref = float(data.case_definitions[ANCHOR_CASE].reference_scales["U_ref"].value)
    prepared = scalers.transform(task_batch_to_device(
        collator([anchor]), device=device, dtype=torch.float32,
    ))
    probe = task.make_validation_problem(prepared)
    model, _ = instantiate_controlled_model(
        build_recipe(), probe, seed=origin["content"]["model_initialization_seed"],
    )
    model = model.to(device=device, dtype=torch.float32)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    ref_cov, gen_cov, ref_phy, gen_phy = [], [], [], []
    for sid in val_ids:
        host = collator([data[idx[sid]]])
        q = host.inputs.detach().cpu().numpy()
        ref_cov.append(spectral_tensor_audit(physical_velocity(q, grid, uref), grid))
        ref_phy.append(spatial_sample(q, grid, uref, 32))
    invalid = []
    keys = []
    for j in range(args.samples):
        key = f"M47.2/{ANCHOR_CASE}/replicate_{j:04d}"
        keys.append(key)
        std = task.sample_standardized(
            model, prepared, steps=args.steps,
            method="flow_ode", solver="heun",
            sampling_eps=0.0, final_denoise=False,
            sampling_seed=args.seed, sampling_keys=(key,),
        )
        q = scalers.inputs.inverse(
            std, prepared.input_channels,
        ).detach().cpu().numpy()
        # No clipping of invalid density. Preserve the failure as a
        # scientific result; skip spectra that would divide by nonpositive rho.
        q_unique = q[unique_node_mask(grid)]
        invalid_fraction = float(np.mean(~np.isfinite(q_unique[:, 0]) | (q_unique[:, 0] <= 0)))
        if invalid_fraction > 0 or not np.isfinite(q_unique).all():
            invalid.append({"sampling_key": key, "density_invalid_fraction": invalid_fraction})
        else:
            gen_cov.append(spectral_tensor_audit(
                physical_velocity(q, grid, uref), grid,
            ))
            gen_phy.append(spatial_sample(q, grid, uref, 32))
        print(
            f"[M47.7B eval] epoch={args.epoch} sample={j+1}/{args.samples} "
            f"invalid_density_fraction={invalid_fraction:.6g}",
            flush=True,
        )
        del std, q

    ref_report = aggregate_spectral_tensor(ref_cov)
    gen_report = aggregate_spectral_tensor(gen_cov) if gen_cov else None
    result = {
        "protocol": PROTOCOL, "status": "complete",
        "case": ANCHOR_CASE, "evaluated_epoch": args.epoch,
        "train_signature_sha256": origin["sha256"],
        "checkpoint": str(args.train_dir / f"epoch{args.epoch:02d}.pt"),
        "training_validation_mse": saved["history"][-1]["validation"]["mean"],
        "reference_sample_ids": val_ids, "generated_sampling_keys": keys,
        "original_train_sample_count": len(split["train_ids"]),
        "generation_samples_requested": args.samples,
        "generation_samples_valid": len(gen_cov),
        "generation_samples_invalid": invalid,
        "source": "independent Gaussian, canonical Heun sampler",
        "training_execution": origin["content"].get("execution", {"sparse_attention_backend": "torch_sparse"}),
        "reference": ref_report, "generated": gen_report,
        "reference_physics": mean_records(ref_phy),
        "generated_physics": mean_records(gen_phy) if gen_phy else None,
        "reference_per_sample": ref_cov,
        "generated_per_sample": gen_cov,
        "heun_steps": args.steps, "sampling_seed": args.seed,
        "spacing_m": list(grid.spacing), "reference_velocity_scale_m_per_s": uref,
        "interpretation_warning": (
            "Single-regime train-only scaler and one-case training population "
            "differ from the six-regime M47.4 reference. Use physical velocity "
            "statistics for comparison; do not compare regression MSE across scalers."
        ),
    }
    _write(args.output_dir / "case_Re54_Ma0.2_covariance.json", result)
    summary = {
        "protocol": PROTOCOL, "status": "complete", "case": ANCHOR_CASE,
        "epoch": args.epoch,
        "training_revision": origin["content"]["code_revision"],
        "evaluation_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "train_run": str(args.train_dir),
        "training_execution": origin["content"].get("execution", {"sparse_attention_backend": "torch_sparse"}),
        "evaluation_forward_backend": "torch_sparse",
        "checkpoint": str(args.train_dir / f"epoch{args.epoch:02d}.pt"),
        "sample_count": len(gen_cov),
        "invalid_count": len(invalid),
        "dns_longitudinal_fraction": ref_report["mean_longitudinal_energy_fraction_integrated"],
        "generated_longitudinal_fraction": (
            None if gen_report is None else gen_report["mean_longitudinal_energy_fraction_integrated"]
        ),
        "dns_cross_cancellation": ref_report["mean_cross_cancellation_integrated"],
        "generated_cross_cancellation": (
            None if gen_report is None else gen_report["mean_cross_cancellation_integrated"]
        ),
        "dns_divcurl_trim4": result["reference_physics"]["mean_interior_divcurl"]["4"]["divergence_to_vorticity_rms"],
        "generated_divcurl_trim4": (
            None if result["generated_physics"] is None else
            result["generated_physics"]["mean_interior_divcurl"]["4"]["divergence_to_vorticity_rms"]
        ),
    }
    _write(args.output_dir / "summary.json", summary)
    print(f"[M47.7B eval] COMPLETE {args.output_dir}", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--epoch", type=int, choices=(10, 30), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references"
    ))
    p.add_argument("--archive-root", type=Path, default=Path(
        "/archivenfs/phlusim/HIT_DNS_COMP"
    ))
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261116)
    run(p.parse_args())


if __name__ == "__main__":
    main()
