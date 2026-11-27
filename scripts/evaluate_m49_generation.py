"""M49 unpaired validation populations at immutable training milestones.

Uses the canonical generic generator and existing M47 population diagnostics.
No test fields are read, no clipping, no normalization refit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from graph_attention.data.case_definition import load_case_definition
from graph_attention.data.periodic_endpoints import reduce_cartesian_endpoints, lift_periodic_endpoints
from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from graph_attention.utils.provenance import collect_runtime_provenance
from scripts.generate_generative import run_generative_generation
from scripts.m47_generation_metrics import (
    CHANNELS, field_diagnostics, finite_mean, mean_curves,
    population_wasserstein, relative_spectrum_l1, sampled_nodes, unique_node_mask,
)
from scripts.m47_6_spatial_metrics import physical_velocity, helmholtz_spectrum
from scripts.m47_7_covariance_metrics import spectral_tensor_audit, aggregate_spectral_tensor

EPOCHS = (25, 100, 250, 500, 750, 1000)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(json_safe(data), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def population(values, ids, grid, channels, u_ref):
    records, nodes, cov, helm = [], [], [], []
    for q, sid in zip(values, ids, strict=True):
        q = np.asarray(q, dtype=np.float64)
        finite = np.isfinite(q).all()
        record = {"sample_id": sid, "finite_all_stored_nodes": bool(finite)}
        if finite:
            rho = q[:, 0]
            positive = rho > 0
            internal = q[positive, 4] - np.sum(q[positive, 1:4]**2, axis=1) / (2*rho[positive])
            record["all_stored_nodes_nonpositive_density_fraction"] = float(np.mean(~positive))
            record["all_stored_nodes_thermo_invalid_fraction"] = float(1 - np.count_nonzero(internal > 0) / len(rho))
            record.update(field_diagnostics(q, grid, channels, k_bins=16, max_lag=16,
                                            velocity_scale_m_per_s=u_ref))
            nodes.append(sampled_nodes(q, grid, key=sid, count=4096))
            if positive.all() and np.all(internal > 0):
                v = physical_velocity(q, grid, u_ref)
                fluct = v - v.mean(axis=(0, 1, 2), keepdims=True)
                flat = fluct.reshape(-1, 3)
                record["tke_m2_s2"] = float(.5*np.mean(np.sum(flat**2, axis=1)))
                record["velocity_covariance_m2_s2"] = (flat.T @ flat / len(flat)).tolist()
                cov.append(spectral_tensor_audit(v, grid))
                helm.append(helmholtz_spectrum(v, grid, k_bins=16))
        records.append(record)
    finite_records = [r for r in records if r["finite_all_stored_nodes"]]
    valid_records = [r for r in finite_records if r["all_stored_nodes_thermo_invalid_fraction"] == 0]
    def mean(key):
        return finite_mean([r.get(key) for r in finite_records])
    covariance = [r["velocity_covariance_m2_s2"] for r in valid_records]
    mean_helm = None
    if helm:
        k = helm[0]["k_rad_per_m"]
        if any(x["k_rad_per_m"] != k for x in helm):
            raise ValueError("different Helmholtz bins")
        mean_helm = {"k_rad_per_m": k, "count": len(helm)}
        for key in ("E_total_m3_per_s2", "E_solenoidal_m3_per_s2", "E_dilatational_m3_per_s2"):
            mean_helm[key] = np.mean([x[key] for x in helm], axis=0).tolist()
    result = {
        "count": len(records), "finite_count": len(finite_records),
        "all_stored_nodes_admissible_count": len(valid_records),
        "mean_density_invalid_fraction": mean("all_stored_nodes_nonpositive_density_fraction"),
        "mean_thermo_invalid_fraction": mean("all_stored_nodes_thermo_invalid_fraction"),
        "mean_tke_m2_s2": mean("tke_m2_s2"),
        "mean_velocity_covariance_m2_s2": np.mean(covariance, axis=0).tolist() if covariance else None,
        "mean_dilatational_energy_fraction": mean("dilatational_energy_fraction"),
        "mean_periodic_endpoint_jump_over_std": {
            ch: finite_mean([r["periodic_endpoint_jump_over_field_std"][ch] for r in finite_records])
            for ch in CHANNELS},
        "velocity_energy_spectrum": mean_curves(valid_records, "velocity_energy_spectrum", "k_rad_per_m", "E_m3_per_s2"),
        "velocity_spatial_correlation": mean_curves(valid_records, "velocity_spatial_correlation", "lag_m", "vector_correlation"),
        "helmholtz_spectrum": mean_helm,
        "spectral_covariance": aggregate_spectral_tensor(cov) if cov else None,
        "per_sample": records,
    }
    return result, nodes


def validate_checkpoint(checkpoint, source, epoch, *, sampling_study=False):
    if checkpoint["epoch"] != epoch-1 or checkpoint["global_step"] != epoch*50:
        raise ValueError("checkpoint epoch/update mismatch")
    model_class = checkpoint["model_class"]
    if model_class not in ("AlternatingDilatedGeometricDiT", "FullDiTGraphTransformer"):
        raise ValueError("expected M49 DiNAT or Full DiT")
    if str(source.model._target_).rsplit(".", 1)[-1] != model_class:
        raise ValueError("checkpoint model differs from frozen source configuration")
    endpoint_policy = str(source.data.get("periodic_endpoint_policy", "keep"))
    if endpoint_policy not in ("keep", "drop_max", "periodic_quotient"):
        raise ValueError("unsupported M49 endpoint policy")
    if model_class == "FullDiTGraphTransformer" and endpoint_policy != "keep":
        raise ValueError("Full DiT evaluation is qualified only for the original stored-node baseline")
    if sampling_study and (epoch != 1000 or endpoint_policy != "keep"
                           or model_class != "AlternatingDilatedGeometricDiT"):
        raise ValueError("sampling study requires original endpoint-inclusive DiNAT epoch 1000")
    return model_class, endpoint_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--steps", type=int, choices=(50,100,200), default=50)
    p.add_argument("--sampling-study", action="store_true")
    p.add_argument("--boundary", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch size must be positive")
    if not 1 <= args.epoch <= 1000:
        p.error("epoch must be in [1,1000] and have an immutable epoch_NNNN.pt checkpoint")
    run = args.run_dir.resolve()
    manifest = json.loads((run/"dataset_split_manifest.json").read_text())
    if len(manifest["validation_ids"]) != 180 or len(manifest["train_ids"]) != 800:
        raise ValueError("expected frozen M49 blocked split")
    source = OmegaConf.load(run/"resolved_config.yaml")
    ckpt_name = f"epoch_{args.epoch:04d}.pt"
    checkpoint = torch.load(run/ckpt_name, map_location="cpu", weights_only=True)
    model_class, endpoint_policy = validate_checkpoint(
        checkpoint, source, args.epoch, sampling_study=args.sampling_study,
    )
    del checkpoint
    unique = endpoint_policy in ("drop_max", "periodic_quotient")
    prefix = "m49_sampling" if args.sampling_study else "m49_validation"
    name = f"{prefix}_epoch_{args.epoch:04d}_heun{args.steps}_seed5678"
    dest = run/"generations"/name
    generation_seconds = None
    if not dest.exists():
        started = perf_counter()
        cfg = OmegaConf.create(dict(run_dir=str(run), checkpoint=ckpt_name, device="cuda",
            batch_size=args.batch_size, num_workers=4, overwrite=False, output_name=name,
            reference_split="validation",
            sampling=dict(steps=args.steps, method="flow_ode", solver="heun",
                          sampling_eps=0.0, final_denoise=False, seed=5678)))
        run_generative_generation(cfg)
        generation_seconds = perf_counter()-started
    # If metrics fail after generation, reuse completed generation only.
    generation = json.loads((dest/"summary.json").read_text())
    if int(OmegaConf.load(dest/"generation_config.yaml").batch_size) != args.batch_size:
        raise ValueError("existing generation batch size differs")
    if generation["source_checkpoint"] != ckpt_name or generation["source_checkpoint_epoch"] != args.epoch-1 or generation["model"] != model_class or generation["reference_population"] != "validation" or generation["sampling_seed"] != 5678 or generation["sampling_steps"] != args.steps or generation["sampling_solver"] != "heun":
        raise ValueError("generation artifact protocol mismatch")
    artifact = torch.load(dest/"generated_validation.pt", map_location="cpu", weights_only=True)
    if tuple(artifact["reference_ids"]) != tuple(manifest["validation_ids"]):
        raise ValueError("reference sample IDs differ")
    if tuple(artifact["generated_ids"]) != tuple(f"gen_{i:06d}" for i in range(180)):
        raise ValueError("unexpected Gaussian sampling keys")
    if artifact["sampling_steps"] != args.steps or artifact["sampling_seed"] != 5678 or artifact["sampling_solver"] != "heun":
        raise ValueError("generation tensor protocol mismatch")
    channels = tuple(artifact["channel_names"])
    if tuple(c.split(".")[0] for c in channels) != CHANNELS:
        raise ValueError("unexpected field semantics")
    dataset = instantiate(source.data)
    idx = {sid:i for i,sid in enumerate(dataset.sample_ids)}
    sid = artifact["reference_ids"][0]
    mesh = dataset[idx[sid]].mesh
    n = 32768 if unique else 35937
    if mesh.num_nodes != n or tuple(artifact["node_counts"]) != (n,)*180:
        raise ValueError("unexpected fixed HIT mesh")
    reference = artifact["target_nondimensional"].reshape(180,n,5)
    generated = artifact["generated_nondimensional"].reshape(180,n,5)
    if unique:
        # Use original native mesh ordering and the same FFT/diagnostics.
        # Lifting duplicates is evaluation only, not a model constraint.
        original_cfg = OmegaConf.create(OmegaConf.to_container(source.data,resolve=True))
        original_cfg.periodic_endpoint_policy = "keep"
        original = instantiate(original_cfg)
        if tuple(original.sample_ids) != tuple(dataset.sample_ids):
            raise ValueError("source inventory differs after endpoint reduction")
        full_mesh = original[idx[sid]].mesh
        reduction = reduce_cartesian_endpoints(full_mesh,side=33)
        if not torch.equal(reduction.mesh.coords,mesh.coords):
            raise ValueError("retained node order differs from native map")
        reference = lift_periodic_endpoints(reference,reduction)
        generated = lift_periodic_endpoints(generated,reduction)
        mesh = full_mesh
    grid = infer_cartesian_grid_3d(mesh.coords.cpu().numpy(), periodic_endpoint_mode="drop_max")
    if grid.source_shape != (33,33,33):
        raise ValueError("unexpected original HIT mesh")
    case = load_case_definition(source.data.case_file)
    u_ref = float(case.reference_scales["U_ref"].value)
    reference = reference.numpy()
    generated = generated.numpy()
    ref, rn = population(reference, artifact["reference_ids"], grid, channels, u_ref)
    gen, gn = population(generated, artifact["generated_ids"], grid, channels, u_ref)
    result = dict(protocol="m49_validation_checkpoint_v1", epoch=args.epoch,
                  reference_split="validation", comparison_mode="unpaired_population",
                  provenance=collect_runtime_provenance(), sampling_seed=5678,
                  sampling_steps=args.steps, source_checkpoint=ckpt_name,
                  generation_wall_seconds=generation_seconds,
                  model_node_count=n, generation_batch_size=args.batch_size,
                  periodic_endpoint_policy=endpoint_policy,
                  periodic_wrap_edges=bool(endpoint_policy == "periodic_quotient"),
                  model_class=model_class,
                  attention_hop_schedule=(list(source.model.get("attention_hop_schedule",(1,2)))
                                          if model_class == "AlternatingDilatedGeometricDiT" else None),
                  model_layers=int(source.model.num_layers),
                  endpoint_values_lifted_for_diagnostics=unique,
                  endpoint_mismatch_interpretation=("zero by construction, not a learned property" if unique else "independent stored-node predictions"),
                  reference=ref, generated=gen,
                  one_point_wasserstein=population_wasserstein(rn,gn) if rn and gn else None,
                  relative_spectrum_l1=relative_spectrum_l1(ref["velocity_energy_spectrum"],gen["velocity_energy_spectrum"]))
    if args.boundary or args.sampling_study or unique:
        from scripts.m47_9b_periodic_boundary_audit import _population as boundary_population
        for label, values, ids in (
            ("reference",reference,artifact["reference_ids"]),
            ("generated",generated,artifact["generated_ids"]),
        ):
            # Existing audit separates identical-location endpoint jumps
            # from true one-cell wrap increments and interior divergence.
            audit = boundary_population(values,ids,channels,grid,u_ref,4)
            result.setdefault("boundary_diagnostics",{})[label] = {
                k:v for k,v in audit.items() if k != "per_sample"
            }
    write(dest/"physical_metrics.json",result)
    print(f"Metrics={dest/'physical_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
