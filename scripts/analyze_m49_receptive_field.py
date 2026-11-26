"""Exact query-node Jacobian audit of trained M49 open-grid DiNAT checkpoints.

Validation fields only; original saved config and train-only standardizers.
Probe straight Gaussian/data paths at fixed times, not generated ODE paths.
The 33^3 stored nodes remain separate variables exactly as during training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from graph_attention.evaluation.receptive_field import (
    qualify_open_cartesian_edges, query_jacobian, shell_statistics,
)
from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT
from graph_attention.tasks import FlowMatchingTask
from graph_attention.tasks.flow_matching import _deterministic_sampling_source, _flow_problem
from graph_attention.training.data_pipeline import GraphTaskCollator, dataset_sample_ids, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from graph_attention.utils.provenance import collect_runtime_provenance
from scripts.evaluate_m49_generation import write
from scripts.generate_generative import _manifest_reference_ids
from scripts.m47_generation_metrics import CHANNELS
from scripts.train_generative import _forward_model, _load_standardizers, _validate_generative_standardizers

TIMES = (0., .5, .95)
SOURCE_SEED = 5678
SHAPE = (33, 33, 33)
QUERY_POINTS = {"interior": (16, 16, 16), "face": (0, 16, 16), "corner": (0, 0, 0)}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = p.parse_args()
    if args.epoch <= 0:
        p.error("epoch must be positive")
    run = args.run_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable diagnostic destination exists: {output}")
    paths = {name: run / name for name in (
        "resolved_config.yaml", "dataset_split_manifest.json", "standardizers.pt",
        f"epoch_{args.epoch:04d}.pt",
    )}
    source = OmegaConf.load(paths["resolved_config.yaml"])
    if source.data.get("periodic_endpoint_policy", "keep") != "keep":
        raise ValueError("first ERF protocol is qualified for endpoint-inclusive open HIT only")
    manifest = json.loads(paths["dataset_split_manifest.json"].read_text())
    ids = _manifest_reference_ids(manifest, "validation")
    if len(ids) != 180 or len(manifest["train_ids"]) != 800:
        raise ValueError("expected original blocked M49 membership")
    selected = (ids[len(ids)//4], ids[3*len(ids)//4])
    checkpoint_name = f"epoch_{args.epoch:04d}.pt"
    ckpt = torch.load(paths[checkpoint_name], map_location="cpu", weights_only=True)
    if (ckpt["model_class"] != "AlternatingDilatedGeometricDiT"
            or ckpt["epoch"] != args.epoch - 1
            or ckpt["global_step"] != 50 * args.epoch):
        raise ValueError("checkpoint architecture, epoch, or optimizer updates disagree")
    checkpoint_git = ckpt.get("runtime_provenance", {}).get("git")
    provenance = collect_runtime_provenance()
    if not provenance["git"]["sha"] or provenance["git"]["dirty"] is not False:
        raise ValueError("run from a clean checkout with accessible Git metadata")

    dataset = instantiate(source.data)
    task = instantiate(source.task)
    if not isinstance(task, FlowMatchingTask) or not task.physical_nondimensionalization:
        raise ValueError("expected convectively nondimensionalized Flow Matching")
    standardizers = _load_standardizers(paths["standardizers.pt"])
    _validate_generative_standardizers(standardizers)
    if (tuple(standardizers.train_sample_ids) != tuple(manifest["train_ids"])
            or not standardizers.physical_nondimensionalization):
        raise ValueError("saved standardizers disagree with the source training manifest/task")
    inventory = {sid: i for i, sid in enumerate(dataset_sample_ids(dataset))}
    collator = GraphTaskCollator(task, dataset.field_catalog, source.geometry)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostic requested without CUDA")
    device_scalers = standardizers.to(device=device, dtype=torch.float32)

    # Build the same controlled model as training and strictly restore weights.
    sample = dataset[inventory[selected[0]]]
    host = collator([sample])
    probe = task.make_model_probe(task.make_validation_problem(standardizers.transform(host)))
    model, _ = instantiate_controlled_model(source.model, probe, seed=int(source.seed))
    if not isinstance(model, AlternatingDilatedGeometricDiT):
        raise TypeError("expected DiNAT-DiT")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    model.requires_grad_(False)
    del ckpt, probe
    schedule = tuple(model.attention_hop_schedule)
    depth = len(model.blocks)
    radius_bound = sum(schedule[l % len(schedule)] for l in range(depth))
    if depth != int(source.model.num_layers):
        raise ValueError("restored depth disagrees with config")

    # Qualify the actual local graph before using Manhattan distances.
    native_coords = sample.mesh.coords.detach().cpu().numpy()
    grid = infer_cartesian_grid_3d(native_coords)
    if grid.source_shape != SHAPE:
        raise ValueError("expected the original complete 33^3 stored grid")
    ijk = np.column_stack(np.unravel_index(grid.linear_indices, SHAPE))
    qualify_open_cartesian_edges(ijk, SHAPE, host.edge_index.numpy())
    queries = {}
    for name, point in QUERY_POINTS.items():
        indices = np.flatnonzero(np.all(ijk == point, axis=1))
        if len(indices) != 1:
            raise ValueError("query Cartesian location not unique")
        node = int(indices[0])
        queries[name] = (node, np.abs(ijk - np.asarray(point)).sum(axis=1))
    task_coords = host.coords.detach().cpu().double().numpy()
    spacing = (task_coords.max(axis=0) - task_coords.min(axis=0)) / (np.asarray(SHAPE)-1)
    if (np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=1e-5, atol=1e-8)
            or not np.allclose(task_coords, task_coords.min(axis=0) + ijk * spacing,
                               rtol=1e-5, atol=1e-6)):
        raise ValueError("expected uniform cubic coordinates in model units")
    dx = float(spacing[0])

    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        protocol="m49_open_grid_exact_input_vjp_v1", status="running",
        source_run=str(run), source_checkpoint=checkpoint_name, epoch=args.epoch,
        artifact_sha256={name: sha256(path) for name,path in paths.items()},
        checkpoint_git=checkpoint_git, diagnostic_provenance=provenance,
        source_model_config=OmegaConf.to_container(source.model, resolve=True),
        layers=depth, hop_schedule=list(schedule), theoretical_radius_hops=radius_bound,
        absolute_coordinate_embedding=model.use_absolute_coordinate_embedding,
        sparse_attention_backend=model.sparse_attention_backend,
        reference_split="validation", reference_ids=list(selected),
        reference_selection="frozen validation positions N//4 and 3*N//4, zero-based",
        source_seed=SOURCE_SEED, noise_keys="validation sample IDs, sampling-purpose Gaussian",
        times=list(TIMES), channel_names=list(CHANNELS),
        time_embedding_scale=task.time_embedding_scale,
        state_space="standardized convectively nondimensional conservative fields",
        input_support="all stored nodes, duplicated endpoints independent",
        distance="open axial-neighbor shortest path; nondimensional path length = hops * dx",
        source_shape=list(SHAPE), raw_mesh_spacing=list(grid.spacing),
        nondimensional_spacing=dx,
        queries={name: {"node_index": node, "cartesian_index": list(QUERY_POINTS[name])}
                 for name,(node,_) in queries.items()},
        records=[], forward_evaluations=0, backward_evaluations=0,
    )
    write(output/"summary.json", report)
    maps = []
    channels = tuple(CHANNELS)
    radial_header = ["sample_id", "time", "query", "distance_hops", "path_length_nondim",
                     "node_count", "squared_sensitivity_sum", "squared_sensitivity_mean",
                     "fraction", "cumulative_fraction", "momentum_diagonal_sum", "momentum_cross_sum"]
    coupling_header = ["sample_id", "time", "query", "distance_hops"] + [
        f"d_v_{out}_d_x_{inp}_squared_sum" for out in channels for inp in channels
    ]
    started = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with (output/"radial_sensitivity.csv").open("x", newline="") as radial_stream, \
         (output/"channel_coupling.csv").open("x", newline="") as coupling_stream:
        radial = csv.writer(radial_stream)
        coupling = csv.writer(coupling_stream)
        radial.writerow(radial_header)
        coupling.writerow(coupling_header)
        for sid in selected:
            sample = dataset[inventory[sid]]
            host = collator([sample])
            if (not np.array_equal(sample.mesh.coords.numpy(), native_coords)
                    or not np.array_equal(host.coords.double().numpy(), task_coords)):
                raise ValueError("reference mesh differs from qualified fixed mesh")
            qualify_open_cartesian_edges(ijk, SHAPE, host.edge_index.numpy())
            batch = device_scalers.transform(task_batch_to_device(host, device=device, dtype=torch.float32))
            if tuple(batch.input_channels) != channels or tuple(batch.target_channels) != channels:
                raise ValueError("unexpected state channel semantics/order")
            if not torch.isfinite(batch.inputs).all():
                raise ValueError("nonfinite reference state")
            z = _deterministic_sampling_source(batch, SOURCE_SEED, (sid,))
            for t in TIMES:
                tick = perf_counter()
                times = torch.tensor([t], device=device, dtype=torch.float32)
                problem = _flow_problem(batch, times, z, time_embedding_scale=task.time_embedding_scale)
                state = problem.inputs.detach().requires_grad_(True)
                # Same model call as training, including segmented packed graph_sizes.
                with torch.enable_grad():
                    v = _forward_model(model, replace(problem, inputs=state))
                    if not torch.isfinite(v).all():
                        raise ValueError("nonfinite predicted Flow Matching velocity")
                    report["forward_evaluations"] += 1
                    for qindex,(name,(node,distances)) in enumerate(queries.items()):
                        jac = query_jacobian(v, state, node, retain_graph=qindex+1 < len(queries))
                        report["backward_evaluations"] += len(channels)
                        stats = shell_statistics(jac, distances)
                        del jac
                        total = stats["total"]
                        pairs = stats["pairs"]
                        momentum = pairs[:, 1:4, 1:4]
                        diag = np.trace(momentum, axis1=1, axis2=2)
                        cross = momentum[:, ~np.eye(3, dtype=bool)].sum(axis=1)
                        mom_total = float(diag.sum() + cross.sum())
                        outside = float(stats["per_node"][distances > radius_bound].sum())
                        radii = stats["radii"]
                        rec = dict(
                            sample_id=sid, time=t, query=name,
                            total_squared_sensitivity=total,
                            self_fraction=float(stats["shell"][0]/total) if total else None,
                            outside_theoretical_radius_squared_sensitivity=outside,
                            outside_theoretical_radius_fraction=outside/total if total else None,
                            momentum_cross_fraction=float(cross.sum()/mom_total) if mom_total else None,
                            radii_hops=radii,
                            radii_nondim_path_length={k: r*dx if r is not None else None for k,r in radii.items()},
                        )
                        report["records"].append(rec)
                        for r,count in enumerate(stats["counts"]):
                            radial.writerow([
                                sid,t,name,r,r*dx,int(count),stats["shell"][r],
                                stats["mean"][r] if count else None,
                                stats["fractions"][r] if total else None,
                                stats["cumulative"][r] if total else None,diag[r],cross[r],
                            ])
                            coupling.writerow([sid,t,name,r,*pairs[r].reshape(-1).tolist()])
                        maps.append(dict(sample_id=sid,time=t,query=name,
                                         squared_frobenius=torch.from_numpy(stats["per_node"].copy())))
                del v, state, problem
                sync(device)
                radial_stream.flush()
                coupling_stream.flush()
                write(output/"summary.json", report)
                print(f"{sid} t={t}: three queries in {perf_counter()-tick:.3f} s", flush=True)
    torch.save(dict(cartesian_indices=torch.from_numpy(ijk.copy()),
                    coords_nondimensional=torch.from_numpy(task_coords.copy()),
                    records=maps), output/"sensitivity_maps.pt")
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("diagnostic unexpectedly accumulated parameter gradients")
    # For a local model, support beyond the theoretical radius should be zero.
    # Preserve the measured values rather than silently mask them.
    report["max_outside_theoretical_radius_fraction"] = max(
        (r["outside_theoretical_radius_fraction"] or 0.) for r in report["records"]
    )
    report["nonzero_outside_theoretical_radius_count"] = sum(
        r["outside_theoretical_radius_squared_sensitivity"] > 0 for r in report["records"]
    )
    report["zero_sensitivity_count"] = sum(r["total_squared_sensitivity"] == 0 for r in report["records"])
    report["elapsed_seconds"] = perf_counter()-started
    if device.type == "cuda":
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device)/1024**3
        report["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device)/1024**3
    report["status"] = "complete"
    write(output/"summary.json", report)
    print(f"Report={output/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
