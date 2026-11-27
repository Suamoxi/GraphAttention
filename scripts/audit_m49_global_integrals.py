"""Read-only M49 global integral audit on the fixed periodic HIT cube.

Native reference fields are integrated in FP64. Saved generations are
convectively nondimensionalized FP32 fields; never integrate standardized
states or infer the channel order. No optimization, correction, or test data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from graph_attention.data.nondimensionalization import ConvectiveNondimensionalizer
from graph_attention.data.periodic_endpoints import reduce_cartesian_endpoints, reduce_sample
from graph_attention.utils.provenance import collect_runtime_provenance
from scripts.m47_generation_metrics import CHANNELS, wasserstein_1d

STATE_CHANNELS = ("rho.value", "rhou.x", "rhov.y", "rhow.z", "rhoE.value")
INTEGRALS = ("mass_kg", "Px_kg_m_s", "Py_kg_m_s", "Pz_kg_m_s", "energy_J")
PROTOCOL = "m49_global_integrals_periodic_unique_v1"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def native_quadrature(mesh, reduction):
    """Equal-volume Cartesian hex nodal quadrature; quotient all duplicates.

    A native cell contributes its volume/8 to each of its eight nodes.
    Normalized weights sum to one. On the qualified periodic uniform mesh
    their pushforward is exactly the uniform measure on unique nodes.
    """
    cells = mesh.cell_connectivity
    if cells is None or cells.shape != (reduction.mesh.num_nodes, 8):
        raise ValueError("require every original uniform hex")
    counts = np.bincount(cells.numpy().reshape(-1), minlength=mesh.num_nodes)
    weights = counts.astype(np.float64) / (8 * len(cells))
    quotient_weights = np.bincount(
        reduction.full_to_unique.numpy(), weights=weights,
        minlength=reduction.mesh.num_nodes,
    )
    if not np.allclose(quotient_weights, 1/reduction.mesh.num_nodes, rtol=0, atol=1e-15):
        raise ValueError("native volume quadrature is inconsistent with periodic quotient")
    volume = float(np.prod(reduction.mesh.metadata["periodic_box_lengths"]))
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("invalid dimensional volume")
    return volume, weights


def snapshot_record(q, sid, *, scales, volume, weights=None, full=None):
    """Integrate one *unique* nondimensional conservative state."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(CHANNELS) or not len(q):
        raise ValueError("expected [unique_nodes,5] ordered conservative fields")
    record = {"sample_id": str(sid), "finite_unique": bool(np.isfinite(q).all()),
              "finite_all_stored": bool(np.isfinite(full).all()) if full is not None
              else bool(np.isfinite(q).all())}
    if not record["finite_unique"]:
        return record
    mean = q.mean(axis=0, dtype=np.float64)
    rho = q[:, 0]
    positive = rho > 0
    internal = q[positive, 4] - np.sum(q[positive, 1:4]**2, axis=1)/(2*rho[positive])
    record.update(
        mean_nondimensional=mean.tolist(),
        spatial_std_nondimensional=q.std(axis=0).tolist(),
        integrals_si=(mean*scales*volume).tolist(),
        nonpositive_density_fraction=float(np.mean(~positive)),
        thermodynamic_invalid_fraction=float(1 - np.count_nonzero(internal > 0)/len(q)),
        bulk_velocity_m_per_s=((mean[1:4]*scales[1:4])/(mean[0]*scales[0])).tolist()
        if mean[0] > 0 else None,
    )
    if full is not None and weights is not None and record["finite_all_stored"]:
        weighted = weights @ np.asarray(full, dtype=np.float64)
        record["native_hex_integrals_si"] = (weighted*scales*volume).tolist()
        record["native_minus_unique_mean_nondimensional"] = (weighted-mean).tolist()
    return record


def statistics(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(x.mean()), "std": float(x.std(ddof=0)),
        "min": float(x.min()), "max": float(x.max()),
        "q05": float(np.quantile(x,.05)), "median": float(np.median(x)),
        "q95": float(np.quantile(x,.95)),
    }


def population(records):
    finite = [r for r in records if r["finite_unique"]]
    if not finite:
        raise ValueError("no finite unique-node snapshots")
    m = np.asarray([r["mean_nondimensional"] for r in finite])
    totals = np.asarray([r["integrals_si"] for r in finite])
    delta = np.asarray([r["native_minus_unique_mean_nondimensional"] for r in finite
                        if "native_minus_unique_mean_nondimensional" in r])
    return {
        "count": len(records), "finite_unique_count": len(finite),
        "finite_all_stored_count": sum(r["finite_all_stored"] for r in records),
        "admissible_unique_count": sum(r["thermodynamic_invalid_fraction"] == 0 for r in finite),
        "mean_nondimensional": {name: statistics(m[:,i]) for i,name in enumerate(CHANNELS)},
        "integrals_si": {name: statistics(totals[:,i]) for i,name in enumerate(INTEGRALS)},
        "mean_spatial_std_nondimensional": np.mean(
            [r["spatial_std_nondimensional"] for r in finite], axis=0).tolist(),
        "native_minus_unique_mean_rms_nondimensional": np.sqrt((delta**2).mean(0)).tolist()
        if len(delta) else None,
        "per_snapshot": records,
    }


def compare(reference, candidate, scales, volume, train_spatial_std):
    """Unpaired empirical global-integral distributions, not samplewise errors."""
    a = np.asarray([r["mean_nondimensional"] for r in reference["per_snapshot"] if r["finite_unique"]])
    b = np.asarray([r["mean_nondimensional"] for r in candidate["per_snapshot"] if r["finite_unique"]])
    result = {}
    for i,name in enumerate(CHANNELS):
        bias = b[:,i].mean()-a[:,i].mean()
        w1 = wasserstein_1d(a[:,i], b[:,i])
        result[name] = {
            "reference_std_nondimensional": float(a[:,i].std()),
            "candidate_std_nondimensional": float(b[:,i].std()),
            "mean_bias_nondimensional": float(bias),
            "mean_bias_integral_si": float(bias*scales[i]*volume),
            "wasserstein_integral_si": float(w1*scales[i]*volume),
            "wasserstein_nondimensional": w1,
            "wasserstein_over_train_spatial_std": w1/train_spatial_std[i]
            if train_spatial_std[i] > 0 else None,
        }
    return result


def check_split(manifest, expected):
    for key in ("train_ids", "validation_ids", "test_ids"):
        if tuple(manifest[key]) != tuple(expected[key]):
            raise ValueError(f"frozen M49 split differs: {key}")
    if len(expected["train_ids"]) != 800 or len(expected["validation_ids"]) != 180:
        raise ValueError("unexpected M49 reference population sizes")


def original_data_config(cfg):
    data = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    if str(data.case_id) != "HIT_LES_FORCED" or tuple(data.field_names) != CHANNELS:
        raise ValueError("audit qualified only for ordered HIT_LES_FORCED fields")
    if not cfg.task.physical_nondimensionalization:
        raise ValueError("convective nondimensionalization required")
    data.periodic_endpoint_policy = "keep"
    return data


def audit_generation(dest, *, split, expected_data, expected_val, reduction, weights,
                     scales, volume, reference, train_spatial_std):
    summary = json.loads((dest/"summary.json").read_text())
    manifest = json.loads((dest/"dataset_split_manifest.json").read_text())
    check_split(manifest, split)
    cfg = OmegaConf.load(dest/"resolved_config.yaml")
    data = original_data_config(cfg)
    # Mesh ordering and field semantics must match the native source. Paths
    # may differ by /scratch vs /gpfs aliases, so compare resolved paths.
    for key in ("mesh_file", "snapshot_dir", "case_file"):
        left, right = Path(str(data[key])), Path(str(expected_data[key]))
        if key == "case_file":
            if sha256(left) != sha256(right):
                raise ValueError("source case references differ")
        elif left.resolve() != right.resolve():
            raise ValueError(f"source data path differs: {key}")
    for key in ("coord_paths", "connectivity_path", "connectivity_indexing"):
        if data[key] != expected_data[key]:
            raise ValueError(f"source mesh convention differs: {key}")
    policy = str(cfg.data.get("periodic_endpoint_policy", "keep"))
    if policy not in ("keep", "drop_max", "periodic_quotient"):
        raise ValueError("unknown endpoint policy")
    artifact_path = dest/"generated_validation.pt"
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if (summary["reference_population"] != "validation"
        or summary["sampling_steps"] != 50 or summary["sampling_seed"] != 5678
        or summary["sampling_solver"] != "heun" or summary["sampling_method"] != "flow_ode"
        or artifact["sampling_steps"] != 50 or artifact["sampling_seed"] != 5678
        or artifact["sampling_solver"] != "heun" or artifact["sampling_method"] != "flow_ode"
        or artifact["generated_reference_pairing"] is not False):
        raise ValueError("expected canonical unpaired M49 Heun50 seed5678")
    if tuple(artifact["channel_names"]) != STATE_CHANNELS:
        raise ValueError("unexpected state channel semantics/order")
    ids = tuple(artifact["reference_ids"])
    gen_ids = tuple(artifact["generated_ids"])
    if ids != tuple(split["validation_ids"]) or gen_ids != tuple(f"gen_{i:06d}" for i in range(180)):
        raise ValueError("population IDs differ from frozen validation protocol")
    retained = reduction.retained.numpy()
    n = len(retained) if policy != "keep" else len(reduction.full_to_unique)
    if tuple(artifact["node_counts"]) != (n,)*180:
        raise ValueError("unexpected fixed node count")
    ref = artifact["target_nondimensional"]
    gen = artifact["generated_nondimensional"]
    if ref.shape != (180*n,5) or gen.shape != ref.shape or ref.dtype != torch.float32 or gen.dtype != torch.float32:
        raise ValueError("unexpected canonical nondimensional tensor shape/dtype")
    ref = ref.numpy().reshape(180,n,5)
    gen = gen.numpy().reshape(180,n,5)
    records = []
    max_ref_diff = np.zeros(5)
    for i, sid in enumerate(ids):
        expected = expected_val[i] if policy == "keep" else expected_val[i][retained]
        np.testing.assert_allclose(ref[i], expected, rtol=1e-6, atol=1e-7,
                                   err_msg=f"stored reference mismatch: {sid}")
        max_ref_diff = np.maximum(max_ref_diff, np.max(np.abs(
            ref[i].astype(np.float64)-expected.astype(np.float64)),axis=0))
        full = gen[i] if policy == "keep" else None
        q = gen[i][retained] if policy == "keep" else gen[i]
        records.append(snapshot_record(q, gen_ids[i], scales=scales, volume=volume,
                                       weights=weights, full=full))
    pop = population(records)
    return {
        "path": str(dest), "model": summary["model"],
        "source_checkpoint": summary["source_checkpoint"],
        "completed_epoch": int(summary["source_checkpoint_epoch"])+1,
        "endpoint_policy": policy, "artifact_sha256": sha256(artifact_path),
        "reference_max_abs_difference_from_native_fp32": max_ref_diff.tolist(),
        "generated": pop,
        "comparison_to_raw_validation": compare(reference,pop,scales,volume,train_spatial_std),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run", type=Path, required=True)
    p.add_argument("--generation-dir", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.baseline_run.resolve()
    cfg = OmegaConf.load(run/"resolved_config.yaml")
    manifest = json.loads((run/"dataset_split_manifest.json").read_text())
    split_path = Path(__file__).resolve().parents[1]/"configs/splits/m49_hit_les_forced_blocked50.json"
    split = json.loads(split_path.read_text())
    check_split(manifest, split)
    data = original_data_config(cfg)
    dataset = instantiate(data)
    lookup = {sid:i for i,sid in enumerate(dataset.sample_ids)}
    first = dataset[lookup[split["train_ids"][0]]]
    reduction = reduce_cartesian_endpoints(first.mesh, side=33, policy="periodic_quotient")
    volume, weights = native_quadrature(first.mesh, reduction)
    transform = ConvectiveNondimensionalizer(first.reference_scales)
    scales = np.array([transform.field_scale(name) for name in CHANNELS], dtype=np.float64)
    expected_val = []
    populations = {}
    for label, key in (("train","train_ids"), ("validation","validation_ids")):
        records = []
        for i,sid in enumerate(split[key]):
            sample = dataset[lookup[sid]]
            if sample.mesh is not first.mesh:
                raise ValueError("expected fixed shared native mesh")
            if sample.reference_scales != first.reference_scales:
                raise ValueError("reference scales vary within fixed HIT case")
            unique = reduce_sample(sample,reduction)
            # FP64 accumulation on original physical values. Transform using
            # float64 before accumulation, not the archived FP32 scalers.
            full = torch.stack([sample.fields[k].double() for k in CHANNELS],dim=1).numpy()/scales
            q = torch.stack([unique.fields[k].double() for k in CHANNELS],dim=1).numpy()/scales
            records.append(snapshot_record(q,sid,scales=scales,volume=volume,
                                           full=full,weights=weights))
            if label == "validation":
                # Reproduce the canonical task conversion BEFORE the FP32
                # sampling cast to qualify stored reference arrays.
                nd = transform.nondimensionalize(sample.fields)
                expected_val.append(torch.stack([nd[k] for k in CHANNELS],dim=1).float().numpy())
            if (i+1)%100 == 0:
                print(f"{label}: {i+1}/{len(split[key])}",flush=True)
        populations[label] = population(records)
    train_std = populations["train"]["mean_spatial_std_nondimensional"]
    reports = []
    for path in args.generation_dir:
        report = audit_generation(path.resolve(),split=split,expected_data=data,
            expected_val=expected_val,reduction=reduction,weights=weights,scales=scales,
            volume=volume,reference=populations["validation"],train_spatial_std=train_std)
        reports.append(report)
        print(f"audited {path}",flush=True)
    result = {
        "protocol": PROTOCOL, "provenance": collect_runtime_provenance(),
        "split_sha256": sha256(split_path),
        "source_manifest_sha256": sha256(run/"dataset_split_manifest.json"),
        "mesh_sha256": sha256(Path(str(data.mesh_file))),
        "case_sha256": sha256(Path(str(data.case_file))),
        "channels": list(CHANNELS), "integral_keys_si": list(INTEGRALS),
        "dimensional_field_scales": scales.tolist(),
        "box_lengths_m": list(reduction.mesh.metadata["periodic_box_lengths"]),
        "volume_m3": volume, "unique_node_count": reduction.mesh.num_nodes,
        "node_volume_m3": volume/reduction.mesh.num_nodes,
        "reference_precision": "native physical fields, FP64 nondimensionalization and accumulation",
        "generation_precision": "archived FP32 nondimensional state, FP64 accumulation",
        "time_basis": "ordered snapshot identifiers, physical times unverified",
        "quadrature": "uniform unique periodic nodes; native hex volume/8 alternative for stored endpoints",
        "interpretation": "mass candidate invariant; forced momentum and energy not assumed invariant; no projection",
        "reference": populations,
        "validation_vs_train": compare(populations["train"],populations["validation"],
                                      scales,volume,train_std),
        "generations": reports,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    tmp.replace(args.output)
    print(f"Audit={args.output}",flush=True)


if __name__ == "__main__":
    main()
