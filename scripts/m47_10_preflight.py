"""M47.10 real-mesh periodic qualification and one-step A3 GPU preflight.

Runs only on allocated Grace jobs. Never opens test physical fields, fits a
scaler, trains a scientific checkpoint, or changes the original archive.

Stages:
  mesh    Qualify all six native Cartesian meshes/cells/periodic graphs and
          check duplicate endpoints on frozen training/validation samples.
  gpu     Run a full 128^3 A3 graph through the production Triton DiNAT
          forward, backward and one AdamW update using synthetic standardized
          Flow Matching states and a real frozen training mesh/condition.
  verify  Check the real-mesh gate (A1/A2) or mesh+GPU feasibility (A3)
          at the exact Git revision and with unchanged qualified inputs.
          Numerical GPU parity tests remain separate diagnostics.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter

import torch

from graph_attention.data.m47_variants import M47VariantDataset
from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, DEFAULT_M43_CASES
from graph_attention.data.multicase_split import temporal_case_split
from graph_attention.data.periodic_endpoints import reduce_cartesian_endpoints
from graph_attention.geometry.relative import edge_relative_displacement
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m47_qualified_setup import (
    eligible_sample_ids, validate_dataset_paths, verify_generated_references,
)
from scripts.m47_reference_conditioning import CONDITION_NAMES
from scripts.train_m43_b_multires import FIELDS, _forward, build_recipe

PROTOCOL = "m47_10_real_mesh_a3_gpu_preflight_v1"
EXPECTED_SIDES = {spec.case_id: spec.side for spec in DEFAULT_M43_CASES}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(repo: Path) -> str:
    return subprocess.check_output(
        ("git", "-C", str(repo), "rev-parse", "HEAD"), text=True,
    ).strip()


def identity(args: argparse.Namespace) -> dict:
    verify_generated_references(args.reference_dir, args.reference_audit)
    reference_hashes = {
        case.replace("/", "_"): sha256(args.reference_dir / (case.replace("/", "_") + ".yaml"))
        for case in EXPECTED_SIDES
    }
    return {
        "protocol": PROTOCOL,
        "git_sha": git_sha(args.repo),
        "qualified_manifest_sha256": sha256(args.qualified_manifest),
        "source_inventory_sha256": sha256(args.source_inventory),
        "selection_policy_sha256": sha256(args.selection_policy),
        "reference_audit_sha256": sha256(args.reference_audit),
        "case_reference_sha256": reference_hashes,
        "archive_root": str(args.archive_root.resolve(strict=True)),
        "reference_dir": str(args.reference_dir.resolve(strict=True)),
    }


def atomic_write(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def require_report(path: Path, expected: dict, stage: str) -> dict:
    data = json.loads(path.read_text())
    if data.get("stage") != stage or data.get("status") != "passed":
        raise RuntimeError(f"{path}: {stage} preflight did not pass")
    if data.get("identity") != expected:
        raise RuntimeError(f"{path}: preflight input/revision identity differs")
    return data


def dataset_and_split(args: argparse.Namespace):
    selected_ids, _ = eligible_sample_ids(
        args.qualified_manifest, args.source_inventory, args.selection_policy,
    )
    data = AVBPMultiCaseHITDataset(
        root=args.archive_root,
        cases=DEFAULT_M43_CASES,
        reference_dir=args.reference_dir,
        require_references=True,
        expected_snapshots_per_case=100,
        selected_sample_ids=selected_ids,
    )
    validate_dataset_paths(data, args.qualified_manifest)
    ids_by_case = {
        case: tuple(sid for sid in data.sample_ids if data.case_by_sample_id[sid] == case)
        for case in EXPECTED_SIDES
    }
    split = temporal_case_split(
        ids_by_case, train_ratio=0.7, validation_ratio=0.15,
        temporal_embargo=2,
    ).manifest
    if len(split.train_ids) != 240 or len(split.validation_ids) != 30:
        raise RuntimeError("unexpected frozen M47 240-train/30-validation membership")
    return data, split


def inspect_one_hop(mesh, *, unique_side: int, length_ref: float, chunk: int = 250_000) -> dict:
    """Check all six signed axial neighbors at every node; O(N) host memory."""
    n = unique_side ** 3
    edges = mesh.edge_index
    if mesh.num_nodes != n or edges.shape != (2, 6 * n):
        raise AssertionError(f"A3 one-hop shape is not (2, {6 * n})")
    box = float(mesh.metadata["periodic_length_nondimensional"])
    spacing = box / unique_side
    xyz = mesh.coords.to(dtype=torch.float64) / length_ref
    counts = torch.zeros(n * 6, dtype=torch.int32)
    wraps = 0
    for begin in range(0, edges.shape[1], chunk):
        pairs = edges[:, begin:begin + chunk]
        src, dst = pairs[0], pairs[1]
        if torch.any(src == dst):
            raise AssertionError("unexpected periodic self loop")
        raw = xyz[src] - xyz[dst]
        delta = edge_relative_displacement(xyz, pairs, periodic_length=box)
        steps = delta / spacing
        discrete = steps.round()
        if not bool(torch.allclose(steps, discrete, rtol=0, atol=5e-4)):
            raise AssertionError("minimum-image edge is not integer grid displacement")
        if not bool(torch.all(discrete.abs().sum(1) == 1)):
            raise AssertionError("periodic edge is not an axial nearest neighbor")
        axis = discrete.abs().argmax(dim=1)
        sign = (discrete.gather(1, axis[:, None]).flatten() > 0).long()
        counts.index_add_(0, dst * 6 + axis * 2 + sign,
                          torch.ones(len(dst), dtype=torch.int32))
        wraps += int((raw.abs() > box / 2).any(dim=1).sum())
    if not bool(torch.all(counts == 1)):
        raise AssertionError("missing or repeated signed periodic neighbor")
    if wraps != 6 * unique_side ** 2:
        raise AssertionError(f"periodic wrap edges {wraps}, expected {6 * unique_side ** 2}")
    expected_box = float(mesh.metadata["periodic_box_lengths"][0]) / length_ref
    if not math.isclose(box, expected_box, rel_tol=1e-6):
        raise AssertionError("periodic length disagrees with physical L_ref")
    return {
        "nodes": n,
        "one_hop_directed_edges": int(edges.shape[1]),
        "wrap_directed_edges": wraps,
        "one_hop_signed_neighbor_degree": 6,
        "normalized_box_length": box,
        "normalized_grid_spacing": spacing,
    }


def _index(source):
    return {sid: i for i, sid in enumerate(source.sample_ids)}


def run_mesh(args: argparse.Namespace, expected: dict) -> None:
    root = args.output_root / "preflight"
    data, split = dataset_and_split(args)
    a2 = M47VariantDataset(data, variant="a2")
    a3 = M47VariantDataset(data, variant="a3")
    indices = _index(data)
    report = {}
    start = perf_counter()
    for case, side in EXPECTED_SIDES.items():
        train_ids = [sid for sid in split.train_ids if data.case_by_sample_id[sid] == case]
        val_ids = [sid for sid in split.validation_ids if data.case_by_sample_id[sid] == case]
        # Structural qualification only: never read the final test set.
        chosen = tuple(dict.fromkeys((
            train_ids[0], train_ids[len(train_ids)//2], train_ids[-1], val_ids[0],
        )))
        first = data[indices[chosen[0]]]
        original = first.mesh
        reduced = a2[indices[chosen[0]]]
        periodic = a3[indices[chosen[0]]]
        # Independently validate *every original native hex cell* via the
        # M49 periodic quotient implementation, without changing A3's graph.
        quotient = reduce_cartesian_endpoints(original, side=side, policy="periodic_quotient")
        if (not torch.equal(reduced.mesh.coords, quotient.mesh.coords)
                or not torch.equal(reduced.mesh.coords, periodic.mesh.coords)
                or quotient.mesh.cell_connectivity.shape != ((side-1)**3, 8)):
            raise AssertionError(f"{case}: unique-node mapping or periodic hex-cell qualification failed")
        length_ref = float(data.case_definitions[case].reference_scales["L_ref"].value)
        topology = inspect_one_hop(periodic.mesh, unique_side=side-1, length_ref=length_ref)
        field_records = []
        for sid in chosen:
            row = a3[indices[sid]]  # strict equality for every stored conserved field
            if row.mesh is not periodic.mesh or tuple(row.fields) != FIELDS:
                raise AssertionError(f"{case}: inconsistent cached A3 geometry or fields")
            if any(v.shape[0] != (side-1)**3 for v in row.fields.values()):
                raise AssertionError(f"{case}: inconsistent endpoint reduction")
            field_records.append(sid)
        report[case] = {
            "original_side": side,
            "unique_side": side - 1,
            "native_periodic_hex_cells": int(quotient.mesh.cell_connectivity.shape[0]),
            "endpoint_checked_samples": field_records,
            "topology": topology,
        }
        print(f"[M47.10] real mesh passed case={case} side={side} ",
              f"nodes={topology['nodes']} edges={topology['one_hop_directed_edges']} ",
              f"wrap={topology['wrap_directed_edges']}", flush=True)
    atomic_write(root / "mesh.json", {
        "stage": "mesh", "status": "passed", "identity": expected,
        "cases": report, "wall_seconds": perf_counter()-start,
        "caveat": "Endpoint values checked on three frozen train and one validation sample per case; all training and validation snapshots are checked again by reduction during training.",
    })
    print(f"[M47.10] mesh preflight PASSED: {root/'mesh.json'}", flush=True)


def run_gpu(args: argparse.Namespace, expected: dict) -> None:
    root = args.output_root / "preflight"
    require_report(root / "mesh.json", expected, "mesh")
    if not torch.cuda.is_available():
        raise RuntimeError("full-size A3 optimizer-step gate requires Grace CUDA")
    data, split = dataset_and_split(args)
    a3 = M47VariantDataset(data, variant="a3")
    case = "Re94/Ma0.2"
    sid = next(s for s in split.train_ids if data.case_by_sample_id[s] == case)
    sample = a3[_index(data)[sid]]
    task = FlowMatchingTask(
        state_fields=FIELDS, conditioning_parameters=CONDITION_NAMES["mach_re"],
        physical_nondimensionalization=True, validation_seed=1234,
        time_embedding_scale=1000.0,
    )
    collator = GraphTaskCollator(task, data.field_catalog, {
        "attention_edge_indices": {"dilated": "exact_two_hop"},
    })
    start = perf_counter()
    # This construction uses the SAME 1- and 2-hop topology path as training.
    host = collator([sample])
    n = 128 ** 3
    if host.inputs.shape != (n, 5) or host.edge_index.shape != (2, 6*n):
        raise AssertionError("A3 128^3 packed graph has the wrong support")
    dilated = host.attention_edge_indices["dilated"]
    if dilated.shape != (2, 18*n):
        raise AssertionError("A3 exact-two-hop shell is not 18 neighbors per node")
    if not bool(torch.all(torch.bincount(dilated[1], minlength=n) == 18)):
        raise AssertionError("A3 exact-two-hop degree must be 18 for every node")
    topology_seconds = perf_counter() - start
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    prepared = task_batch_to_device(host, device=device, dtype=torch.float32)
    # This is a resource/kernel correctness test, not an optimization study:
    # synthetic STANDARDIZED Gaussian targets avoid fitting a scientific scaler.
    rng = torch.Generator(device=device).manual_seed(104710)
    standardized = replace(prepared, inputs=torch.randn(
        (n, 5), generator=rng, device=device, dtype=torch.float32,
    ), targets=torch.randn((n, 5), generator=rng, device=device, dtype=torch.float32))
    problem = task.make_training_problem(standardized, generator=rng)
    recipe = build_recipe()
    recipe.use_absolute_coordinate_embedding = False
    recipe.sparse_attention_backend = "torch_sparse_triton"
    model, init = instantiate_controlled_model(
        recipe, task.make_model_probe(problem), seed=42,
    )
    model = model.to(device=device, dtype=torch.float32)
    model.train()
    # DiT's adaLN-Zero initialization initially blocks gradients through
    # attention. Deliberately open the residual gates in this throwaway
    # preflight model so every Triton sparse-attention backward runs.
    # Scientific training retains the untouched normal initialization.
    with torch.no_grad():
        model.final_layer.projection.weight.normal_(std=0.03)
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=0.02)
            block.adaLN_modulation[-1].bias.normal_(std=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    compute_start = perf_counter()
    loss = task.training_loss(_forward(model, problem), problem)
    if loss.sample_count != 1 or not bool(torch.isfinite(loss.mean)):
        raise AssertionError("nonfinite A3 Flow Matching loss")
    loss.loss_sum.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
        raise AssertionError("missing or nonfinite A3 gradients")
    if not any(bool((g != 0).any()) for g in grads):
        raise AssertionError("all A3 gradients are identically zero")
    active_geometric_blocks = sum(
        any(p.grad is not None and bool((p.grad != 0).any())
            for p in block.attention.geometry_mlp.parameters())
        for block in model.blocks
    )
    if active_geometric_blocks != len(model.blocks):
        raise AssertionError(
            f"only {active_geometric_blocks}/{len(model.blocks)} blocks have nonzero geometric-attention gradients"
        )
    optimizer.step()
    torch.cuda.synchronize(device)
    for p in model.parameters():
        if not bool(torch.isfinite(p).all()):
            raise AssertionError("A3 optimizer step produced nonfinite parameters")
    elapsed = perf_counter()-compute_start
    result = {
        "stage": "gpu", "status": "passed", "identity": expected,
        "case": case, "sample_id": sid,
        "unique_nodes": n, "one_hop_edges": 6*n, "exact_two_hop_edges": 18*n,
        "loss_value": float(loss.mean.detach().cpu()),
        "finite_gradient_tensor_count": len(grads),
        "nonzero_geometric_attention_gradient_blocks": active_geometric_blocks,
        "preflight_only_adaln_gates_activated": True,
        "topology_construction_seconds": topology_seconds,
        "forward_backward_adamw_seconds": elapsed,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device)/1024**3,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved(device)/1024**3,
        "gpu_name": torch.cuda.get_device_name(device),
        "optimizer_step_passed": True,
        "check": "actual Re94 A3 mesh and mach_re condition, synthetic iid standardized Flow Matching state, original production 10-layer h128/h4 Triton model, backward and AdamW",
    }
    atomic_write(root / "gpu.json", result)
    atomic_write(root / "ready.json", {
        "stage": "ready", "status": "passed", "identity": expected,
        "mesh_report_sha256": sha256(root / "mesh.json"),
        "gpu_report_sha256": sha256(root / "gpu.json"),
    })
    print(f"[M47.10] GPU preflight PASSED: {root/'gpu.json'}", flush=True)
    print(f"[M47.10] READY: {root/'ready.json'}", flush=True)


def verify(args: argparse.Namespace, expected: dict) -> None:
    root = args.output_root / "preflight"
    mesh_report = root / "mesh.json"
    require_report(mesh_report, expected, "mesh")
    if args.verify_scope == "full":
        gpu_report = root / "gpu.json"
        require_report(gpu_report, expected, "gpu")
        gate = require_report(root / "ready.json", expected, "ready")
        if (gate.get("mesh_report_sha256") != sha256(mesh_report)
                or gate.get("gpu_report_sha256") != sha256(gpu_report)):
            raise RuntimeError("preflight reports changed since ready gate")
    print(f"[M47.10] PASSED {args.verify_scope} preflight gate at git={expected['git_sha']}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("mesh", "gpu", "verify"), required=True)
    p.add_argument("--verify-scope", choices=("mesh", "full"), default="full",
                   help="For verify: A1/A2 require mesh; A3 requires full GPU feasibility")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--reference-audit", type=Path, required=True)
    p.add_argument("--qualified-manifest", type=Path, required=True)
    p.add_argument("--source-inventory", type=Path, required=True)
    p.add_argument("--selection-policy", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    signature = identity(args)
    if args.stage == "mesh":
        run_mesh(args, signature)
    elif args.stage == "gpu":
        run_gpu(args, signature)
    else:
        verify(args, signature)


if __name__ == "__main__":
    main()
