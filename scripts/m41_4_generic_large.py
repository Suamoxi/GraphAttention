"""Generic M41.4 unstructured-graph preparation and GH200 large-mesh inference.

PREPARE: HDF5 cells -> disk one-hop CSR -> spatial owners -> partition-local
exact-two-hop edges. No Cartesian shortcut or global A^2 materialization.

INFER: same trained M37 weights and M41.3 CPU-streamed executor. Supports a
single measured forward and resumable 50-step Heun in standardized space.
The checkpoint was trained on a different 33^3 HIT dataset: successful
high-resolution inference does NOT establish statistical generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import h5py
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from graph_attention.geometry.external_csr import (
    DiskCSR, build_hdf5_disk_csr,
)
from graph_attention.geometry.partition_csr import (
    spatial_owners, build_partition_files,
)
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT
from graph_attention.models.disk_streamed_dinat import build_disk_streamed_dinat_plan
from graph_attention.models.streamed_dinat import forward_streamed_dinat
from scripts.generate_generative import _load_standardizers


FIELD_NAMES = ("rho", "rhou", "rhov", "rhow", "rhoE")


def _host_memory_stats() -> dict[str, float | None]:
    """Linux process RSS plus peak RSS, including memory-mapped graph pages.

    ru_maxrss is KiB on Linux and bytes on macOS. RSS measures the full
    resident CPU process (graph plan, hidden buffers, Python and caches),
    not Slurm node-wide RAM consumption.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gib = float(peak) / (1024**2 if sys.platform != "darwin" else 1024**3)
    current_gib = None
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                current_gib = int(line.split()[1]) / (1024**2)
                break
    return {"rss_current_gib": current_gib, "rss_peak_gib": peak_gib}


def _report(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _read_coords(mesh: Path, output: Path, n: int, divisor: float) -> np.ndarray:
    with h5py.File(mesh, "r") as h5:
        dsets = [h5[f"Coordinates/{name}"] for name in "xyz"]
        if any(d.shape != (n,) for d in dsets):
            raise ValueError(f"all coordinate datasets must be [N={n}]")
        result = np.lib.format.open_memmap(
            output, mode="w+", dtype=np.float32, shape=(n, 3)
        )
        for start in range(0, n, 262144):
            stop = min(n, start+262144)
            slab = np.column_stack([d[start:stop] for d in dsets])
            if not np.isfinite(slab).all():
                raise ValueError("nonfinite coordinates")
            result[start:stop] = (slab/divisor).astype(np.float32)
        result.flush()
        del result
    return np.load(output, mmap_mode="r")


def _prepare(args) -> dict:
    start = time.perf_counter()
    root = args.output_dir.resolve()
    if root.exists():
        raise FileExistsError(f"output already exists: {root}; never silently overwrite")
    root.mkdir(parents=True)
    mesh = args.mesh_file.expanduser().resolve(strict=True)
    snapshot = args.snapshot_file.expanduser().resolve(strict=True)
    if args.coordinate_divisor <= 0 or not np.isfinite(args.coordinate_divisor):
        raise ValueError("coordinate divisor must be finite and strictly positive")
    with h5py.File(mesh, "r") as mf, h5py.File(snapshot, "r") as sf:
        n = int(mf["Coordinates/x"].size)
        if n != args.expected_nodes:
            raise ValueError(f"expected {args.expected_nodes} nodes, got {n}")
        for field in FIELD_NAMES:
            if sf[f"GaseousPhase/{field}"].shape != (n,):
                raise ValueError(f"snapshot field {field} does not match mesh nodes")
        connectivity = mf[args.connectivity_path]
        if not np.issubdtype(connectivity.dtype, np.integer):
            raise TypeError("connectivity must have integer dtype")
    print(f"[prepare] mesh={mesh}, nodes={n}, element={args.element_type}", flush=True)
    coords = _read_coords(mesh, root/"coords.npy", n, args.coordinate_divisor)
    owners = spatial_owners(coords, blocks=tuple(args.blocks))
    np.save(root/"owners.npy", owners)
    print(f"[prepare] {int(owners.max())+1} occupied coordinate-based partitions", flush=True)
    t1 = time.perf_counter()
    csr_meta = build_hdf5_disk_csr(
        mesh, num_nodes=n, output_dir=root/"csr",
        element_type=args.element_type, connectivity_path=args.connectivity_path,
        one_based=(args.indexing=="one"), bucket_nodes=args.bucket_nodes,
        chunk_cells=args.chunk_cells,
    )
    print(f"[prepare] unique one-hop directed edges={csr_meta['directed_edges']:,}; "
          f"CSR seconds={time.perf_counter()-t1:.1f}", flush=True)
    t2 = time.perf_counter()
    csr = DiskCSR.open(root/"csr")
    part_meta = build_partition_files(
        csr, owners, output_dir=root/"partitions",
        max_owned_nodes=args.max_owned_nodes, chunk_targets=args.chunk_targets,
        include_self_attention=args.include_self_attention,
        include_local_self_attention=args.include_local_self_attention,
    )
    print(f"[prepare] exact local partition topologies generated "
          f"in {time.perf_counter()-t2:.1f}s", flush=True)
    output = {
        "status": "generic_topology_ready",
        "num_nodes": n, "num_partitions": part_meta["num_partitions"],
        "num_onehop_edges": csr_meta["directed_edges"],
        "num_partition_local_edges": sum(p["local_edges"] for p in part_meta["parts"]),
        "num_partition_twohop_edges": sum(p["dilated_edges"] for p in part_meta["parts"]),
        "max_partition_read_nodes": max(p["read_nodes"] for p in part_meta["parts"]),
        "max_partition_owned_nodes": max(p["owned_nodes"] for p in part_meta["parts"]),
        "mesh": str(mesh), "snapshot": str(snapshot),
        "element_type": args.element_type, "indexing": args.indexing,
        "connectivity_path": args.connectivity_path,
        "blocks": args.blocks, "coordinate_divisor": args.coordinate_divisor,
        "include_self_attention": args.include_self_attention,
        "include_local_self_attention": args.include_local_self_attention,
        "seconds": time.perf_counter()-start,
        "host_memory_at_end": _host_memory_stats(),
        "caveats": [
            "Coordinates must use the same nondimensional convention as the trained M37 model.",
            "The existing checkpoint was trained on a different, smaller HIT dataset.",
            "One-hop graph is global on disk; no global exact-two-hop graph is constructed.",
        ],
    }
    _report(root/"prepare.json", output)
    print(json.dumps(output, indent=2), flush=True)
    return output


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_model(run_dir: Path, checkpoint: str, channels: int):
    config = OmegaConf.load(run_dir/"resolved_config.yaml")
    if str(config.model.get("_target_")) != (
        "graph_attention.models.dinat_dit.AlternatingDilatedGeometricDiT"
    ):
        raise ValueError("large-inference path is for original M37 DiNAT-DiT only")
    if "FlowMatchingTask" not in str(config.task.get("_target_")):
        raise ValueError("M41.4 generation supports FlowMatchingTask only")
    if str(config.model.get("sparse_attention_backend")) != "torch_sparse":
        raise ValueError("M41.4 executor requires original torch_sparse backend")
    if channels != len(config.task.state_fields):
        raise ValueError("state-channel count does not match task state field count")
    conditioning_channels = len(config.task.get("conditioning_parameters", [])) + 1
    model = instantiate(
        config.model, in_channels=channels, out_channels=channels,
        spatial_dim=3, conditioning_channels=conditioning_channels,
    )
    if not isinstance(model, AlternatingDilatedGeometricDiT):
        raise TypeError("instantiated an unexpected architecture")
    ckpt_path = run_dir/checkpoint
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if payload.get("model_class") not in (None, type(model).__name__):
        raise ValueError("checkpoint architecture differs")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, config, ckpt_path


def _inference(args) -> dict:
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("M41.4 inference must run on a Grace GPU")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.prepared_dir.resolve(strict=True)
    prep = json.loads((root/"prepare.json").read_text())
    if prep["status"] != "generic_topology_ready":
        raise ValueError("mesh preparation did not complete")
    run_dir = args.run_dir.resolve(strict=True)
    standardizers = _load_standardizers(run_dir/"standardizers.pt")
    names = tuple(standardizers.inputs.channel_names)
    if len(names) != 5 or tuple(configured.split(".")[0] for configured in names) != FIELD_NAMES:
        raise ValueError(f"unexpected training channel semantics: {names}")
    if not standardizers.physical_nondimensionalization:
        raise ValueError("expected M37 physical nondimensionalization; inspect saved config")
    model, cfg, ckpt = _load_model(run_dir, args.checkpoint, len(names))
    if len(cfg.task.get("conditioning_parameters", [])) != 0:
        raise ValueError("only unconditionally trained M37 supported by high-res driver")
    if bool(cfg.task.get("physical_nondimensionalization", False)) != bool(standardizers.physical_nondimensionalization):
        raise ValueError("training task and standardizers mismatch")
    for field in ("include_self_attention", "include_local_self_attention"):
        if bool(prep[field]) != bool(getattr(model, field)):
            raise ValueError(f"prepared graph self-loop policy mismatch: {field}")
    model = model.to("cuda", dtype=torch.float32).eval()
    raw_coords = np.load(root/"coords.npy", mmap_mode="r")
    coords = torch.from_numpy(np.array(raw_coords, dtype=np.float32, copy=True))
    host_memory_before_plan = _host_memory_stats()
    plan_started = time.perf_counter()
    plan = build_disk_streamed_dinat_plan(
        model, coords=coords, partition_dir=root/"partitions",
        preload=args.preload_partitions, pin_memory=False,
    )
    plan_seconds = time.perf_counter()-plan_started
    host_memory_after_plan = _host_memory_stats()
    print(f"[infer] partition plan ready in {plan_seconds:.1f}s; "
          f"preload={args.preload_partitions}, "
          f"host RSS={host_memory_after_plan['rss_current_gib']:.2f}GiB, "
          f"host peak={host_memory_after_plan['rss_peak_gib']:.2f}GiB", flush=True)
    n = prep["num_nodes"]
    ckpt_digest = _sha256(ckpt)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    if args.mode == "forward":
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        noise = torch.randn((n, len(names)), generator=generator, dtype=torch.float32)
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = forward_streamed_dinat(
                model, noise, plan=plan,
                conditioning=torch.tensor([[0.0]], dtype=torch.float32),
            )
        if not torch.isfinite(output).all():
            raise ValueError("nonfinite full high-resolution forward result")
        report = {
            "status": "full_highres_forward_passed", "flow_time": 0.0,
            "nodes": n, "output_shape": list(output.shape),
            "forward_seconds": time.perf_counter()-t0,
            "output_mean": output.mean(0).tolist(),
            "output_std": output.std(0).tolist(),
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device)/1024**3,
            "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device)/1024**3,
            "checkpoint_sha256": ckpt_digest,
            "preload_partitions": args.preload_partitions,
            "plan_load_seconds": plan_seconds,
            "host_memory_before_plan": host_memory_before_plan,
            "host_memory_after_plan": host_memory_after_plan,
            "host_memory_after_forward": _host_memory_stats(),
            "partition_count": prep["num_partitions"],
            "max_partition_owned_nodes": prep["max_partition_owned_nodes"],
            "max_partition_read_nodes": prep["max_partition_read_nodes"],
            "total_wall_seconds": time.perf_counter()-started,
        }
        _report(args.output_dir/"forward.json", report)
        if args.save_forward:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.output_dir/"forward.npy", output.numpy())
        print(json.dumps(report, indent=2), flush=True)
        return report

    if args.mode != "heun":
        raise ValueError("unsupported inference mode")
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    progress_file = folder/"progress.json"
    signature = {
        "nodes": n, "channels": list(names),
        "checkpoint_sha256": ckpt_digest,
        "prepared_dir": str(root), "seed": args.seed,
        "steps": args.steps, "time_embedding_scale": float(cfg.task.time_embedding_scale),
    }
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        if progress["signature"] != signature:
            raise ValueError("resume metadata differs from current run/checkpoint")
        index = int(progress["steps_complete"])
        filename = folder/progress["state_filename"]
        if not filename.is_file():
            raise FileNotFoundError("checkpoint state for Heun resume is missing")
        state = torch.from_numpy(np.load(filename))
        if state.shape != (n, len(names)) or state.dtype != torch.float32:
            raise ValueError("saved generation state has incorrect shape or dtype")
        print(f"[infer] resuming Heun at completed step {index}", flush=True)
    else:
        if any(folder.iterdir()):
            raise FileExistsError("generation directory must be empty for a new run")
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        state = torch.randn((n, len(names)), generator=generator, dtype=torch.float32)
        index = 0
        progress = {"signature": signature, "steps_complete": 0,
                    "state_filename": None, "nfe": 0, "history": []}

    upper = min(args.steps, index + args.max_steps_this_run)
    with torch.inference_mode():
        for step in range(index, upper):
            t0 = time.perf_counter()
            t = torch.tensor([[step/args.steps*float(cfg.task.time_embedding_scale)]],
                             dtype=torch.float32)
            k1 = forward_streamed_dinat(model, state, plan=plan, conditioning=t)
            predictor = state + k1/args.steps
            t_next = torch.tensor([[(step+1)/args.steps*float(cfg.task.time_embedding_scale)]],
                                  dtype=torch.float32)
            k2 = forward_streamed_dinat(model, predictor, plan=plan, conditioning=t_next)
            state = state + (k1+k2)*(0.5/args.steps)
            if not torch.isfinite(state).all():
                raise ValueError(f"nonfinite generated state at step {step}")
            elapsed = time.perf_counter()-t0
            filename = f"state_step_{step+1:04d}.npy"
            np.save(folder/filename, state.numpy())
            prior = progress.get("state_filename")
            progress = {
                "signature": signature, "steps_complete": step+1,
                "state_filename": filename, "nfe": 2*(step+1),
                "history": progress["history"]+[{"step": step+1, "seconds": elapsed}],
            }
            _report(progress_file, progress)
            if prior and prior != filename:
                (folder/prior).unlink(missing_ok=True)
            print(f"[infer] Heun {step+1}/{args.steps}, NFE {2*(step+1)}, "
                  f"elapsed={elapsed:.1f}s, GPU allocated peak="
                  f"{torch.cuda.max_memory_allocated(device)/1024**3:.2f}GiB", flush=True)
    if progress["steps_complete"] == args.steps:
        # Output in node order and training's nondimensional convention.
        # Do NOT label the inverse-statistically-scaled fields as SI units.
        output_h5 = folder/"generated_nondimensional.h5"
        with h5py.File(output_h5, "w") as dst:
            s = dst.create_group("standardized")
            u = dst.create_group("training_nondimensional")
            avg = standardizers.inputs.mean.cpu().numpy()
            scale = standardizers.inputs.scale.cpu().numpy()
            for j, name in enumerate(FIELD_NAMES):
                values = state.numpy()[:, j]
                s.create_dataset(name, data=values)
                u.create_dataset(name, data=values*float(scale[j])+float(avg[j]))
            dst.attrs["checkpoint_sha256"] = ckpt_digest
            dst.attrs["physical_units"] = "not assigned: training nondimensionalization"
        progress["generated_file"] = str(output_h5)
        progress["status"] = "heun_generation_complete"
        _report(progress_file, progress)
    else:
        progress["status"] = "partially_generated_resume_with_same_command"
        _report(progress_file, progress)
    report = {
        **progress,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device)/1024**3,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device)/1024**3,
        "session_wall_seconds": time.perf_counter()-started,
        "plan_load_seconds": plan_seconds,
        "host_memory_before_plan": host_memory_before_plan,
        "host_memory_after_plan": host_memory_after_plan,
        "host_memory_at_end": _host_memory_stats(),
        "partition_count": prep["num_partitions"],
    }
    _report(folder/"run_report.json", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--mesh-file", type=Path, required=True)
    prep.add_argument("--snapshot-file", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--expected-nodes", type=int, default=257**3)
    prep.add_argument("--blocks", nargs=3, type=int, default=[4,4,4])
    prep.add_argument("--element-type", choices=["hex","tet"], default="hex")
    prep.add_argument("--connectivity-path", default="Connectivity/hex->node")
    prep.add_argument("--indexing", choices=["zero","one"], default="one")
    prep.add_argument("--bucket-nodes", type=int, default=131072)
    prep.add_argument("--chunk-cells", type=int, default=65536)
    prep.add_argument("--chunk-targets", type=int, default=8192)
    prep.add_argument("--max-owned-nodes", type=int, default=400000)
    prep.add_argument("--coordinate-divisor", type=float, default=1.0)
    prep.add_argument("--include-self-attention", action="store_true")
    prep.add_argument("--include-local-self-attention", action="store_true")
    infer = sub.add_parser("infer")
    infer.add_argument("--prepared-dir", type=Path, required=True)
    infer.add_argument("--run-dir", type=Path, required=True)
    infer.add_argument("--checkpoint", default="best.pt")
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--mode", choices=["forward", "heun"], default="forward")
    infer.add_argument("--preload-partitions", action="store_true")
    infer.add_argument("--seed", type=int, default=5678)
    infer.add_argument("--save-forward", action="store_true")
    infer.add_argument("--steps", type=int, default=50)
    infer.add_argument("--max-steps-this-run", type=int, default=1)
    args = p.parse_args()
    if args.command == "prepare":
        _prepare(args)
    else:
        if args.steps < 1 or args.max_steps_this_run < 1:
            raise ValueError("steps and max_steps_this_run must be positive")
        _inference(args)


if __name__ == "__main__":
    main()
