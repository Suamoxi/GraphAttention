"""M48.4f: matched real-CEDAR SR patch execution ablation on one GPU.

Compare 4 sequential single-graph forward/backward passes against a disjoint
packed batch with either original indexed or opt-in segmented conditioning.
All modes have identical full-domain-projected FM samples, exact halos,
owned-only equal-patch loss, one gradient clip, and one AdamW update.
No scientific M48 trainer or checkpoint defaults are modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import torch

from graph_attention.data.ms_sr_adapter import MANIFEST_SHA256
from graph_attention.data.ms_sr_training import record_identity, select_role_records
from graph_attention.models.two_scale_dinat import TwoScaleDiNAT
from scripts.benchmark_m47_segmented import audit_parity
from scripts.m43_b1_memory_gate import atomic_json
from scripts.m48_4b_frozen_train import (
    DEFAULT_NORMALIZATION, _grid_representatives, _read_frozen, _step_draw,
)
from scripts.m48_4c_patch_batch import make_patch
from scripts.m48_4f_patch_execution import (
    MODES, execute_patch_update, sample_sha256, tensor_sha256,
)
from scripts.m48_ms_sr_train import _prepare_pair


def authenticated_sources(root: Path, normalization: Path):
    """Reuse the locked 44-teacher read-only M48.4b source and scaler gate."""
    records, selection_sha = select_role_records(
        root, role="teacher", filter_id="w1p25_l1",
        case_prefix="", max_sources=0, manifest_sha256=MANIFEST_SHA256,
    )
    frozen, scales, frozen_sha = _read_frozen(
        normalization, records, selection_sha, manifest_sha256=MANIFEST_SHA256,
    )
    representatives = _grid_representatives(records)
    selected = [record_identity(r) for r in representatives]
    return representatives, scales, {
        "manifest_sha256": MANIFEST_SHA256,
        "selection_sha256": selection_sha,
        "normalization_sha256": frozen_sha,
        "normalization_audit_sha256": frozen["audit_sha256"],
        "teacher_snapshots": len(records),
        "representatives": selected,
    }


def _git_revision() -> str:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("M48.4f benchmark requires a clean Git checkout")
    return revision


def worker(manifest_file: Path, execution: str, report_path: Path, audit_dir: Path):
    manifest = json.loads(manifest_file.read_text())
    if execution not in MODES:
        raise ValueError("unknown M48.4f execution mode")
    if _git_revision() != manifest["revision"]:
        raise RuntimeError("code revision changed between driver and worker")
    setup = manifest["configuration"]
    root = Path(manifest["ms_sr_root"])
    normalization = Path(manifest["normalization_file"])
    representatives, scales, provenance = authenticated_sources(root, normalization)
    if provenance != manifest["provenance"]:
        raise RuntimeError("M48.4f authenticated teacher provenance changed")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("M48.4f performance runs require a real CUDA GPU")
    torch.manual_seed(setup["seed"])
    torch.cuda.manual_seed_all(setup["seed"])
    model = TwoScaleDiNAT(
        mode=setup["architecture"], fine_width=setup["width"],
        coarse_width=setup["width"], num_heads=setup["heads"],
        fine_pre_layers=setup["depth"], fine_post_layers=setup["depth"],
        coarse_static_layers=setup["depth"],
        coarse_dynamic_layers=setup["depth"],
    ).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=setup["learning_rate"])
    initial_sha = tensor_sha256([p for p in model.parameters()])
    options = SimpleNamespace(
        ms_sr_root=root, manifest_sha256=MANIFEST_SHA256,
        core_side=setup["core_side"], depth=setup["depth"],
        max_fine_read=setup["max_fine_read"],
        max_coarse_read=setup["max_coarse_read"],
    )
    audit_dir.mkdir(parents=True, exist_ok=False)
    measurements = {}
    for grid_id, record in enumerate(representatives):
        side = record["target_shape"][0]
        mark = time.perf_counter()
        prepared = _prepare_pair(options, record, scales)
        loaded_seconds = time.perf_counter() - mark
        records = []
        reference_sha = None
        for step in range(setup["steps_per_grid"]):
            draws = [
                _step_draw(
                    setup["seed"], grid_id * 1000000 + step * setup["effective_batch"] + j,
                    side,
                )
                for j in range(setup["effective_batch"])
            ]
            mark = time.perf_counter()
            patches = [
                make_patch(
                    prepared, options, fm_time=t, core_origin=origin,
                    source_seed=seed, mode=setup["architecture"],
                )
                for t, origin, seed in draws
            ]
            prep_seconds = time.perf_counter() - mark
            if step == 0:
                reference_sha = sample_sha256(patches)
            metrics, audit = execute_patch_update(
                model, optimizer, patches, mode=execution, device=device,
                grad_clip=setup["grad_clip"], capture_audit=(step == 0),
            )
            metrics["preparation_seconds"] = prep_seconds
            metrics["total_step_seconds"] = prep_seconds + metrics["update_seconds"]
            metrics["max_projected_source_defect"] = max(
                float(p["cedar_defect"]) for p in patches
            )
            if metrics["max_projected_source_defect"] >= 1e-9:
                raise RuntimeError("M48.4f global CEDAR projected source invalid")
            if step == 0:
                audit["input_sha256"] = reference_sha
                audit["initial_weights_sha256"] = initial_sha
                torch.save(audit, audit_dir / f"grid_{side}.pt")
            records.append(metrics)
            print(
                f"[M48.4f] mode={execution} grid={side} step={step + 1}/"
                f"{setup['steps_per_grid']} loss={metrics['loss']:.8f} "
                f"prep={prep_seconds:.4f}s gpu={metrics['forward_backward_seconds']:.4f}s "
                f"update={metrics['update_seconds']:.4f}s "
                f"peak={metrics['peak_allocated_gib']:.3f}GiB",
                flush=True,
            )
        stable = records[1:]
        wanted = (
            "preparation_seconds", "transfer_seconds", "forward_backward_seconds",
            "optimizer_seconds", "update_seconds", "total_step_seconds",
        )
        measurements[str(side)] = {
            "case": record["case"], "snapshot_id": int(record["snapshot_id"]),
            "fine_side": side, "loaded_seconds": loaded_seconds,
            "input_sha256": reference_sha,
            "first_step_audit_path": str(audit_dir / f"grid_{side}.pt"),
            "steps": records,
            "medians": {k: statistics.median(x[k] for x in stable) for k in wanted},
            "measured_steps": len(stable), "warmup_discarded": 1,
            "peak_allocated_gib": max(x["peak_allocated_gib"] for x in stable),
            "peak_reserved_gib": max(x["peak_reserved_gib"] for x in stable),
            "owned_nodes_per_update": records[0]["owned_fine_nodes"],
            "fine_read_nodes_per_update": records[0]["fine_read_nodes"],
            "coarse_read_nodes_per_update": records[0]["coarse_read_nodes"],
            "owned_nodes_per_gpu_second": (
                records[0]["owned_fine_nodes"]
                / statistics.median(x["forward_backward_seconds"] for x in stable)
            ),
            "owned_nodes_per_end_to_end_second": (
                records[0]["owned_fine_nodes"]
                / statistics.median(x["total_step_seconds"] for x in stable)
            ),
        }
    return {
        "status": "ok", "execution": execution,
        "architecture": setup["architecture"],
        "revision": manifest["revision"], "input_provenance": provenance,
        "initial_weights_sha256": initial_sha, "grids": measurements,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "cuda_device": torch.cuda.get_device_name(device),
        "gpu_total_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
    }


def compare_grid(reports: dict[str, dict], side: str, audit_dirs: dict[str, Path]):
    reference = reports["sequential_single"]
    baseline = reference["grids"][side]
    output = {}
    matched = True
    for execution in ("packed_index", "packed_segmented"):
        candidate = reports[execution]
        grid = candidate["grids"][side]
        hashes_match = (
            candidate["initial_weights_sha256"] == reference["initial_weights_sha256"]
            and grid["input_sha256"] == baseline["input_sha256"]
            and candidate["input_provenance"] == reference["input_provenance"]
        )
        audit = audit_parity(
            audit_dirs["sequential_single"] / f"grid_{side}.pt",
            audit_dirs[execution] / f"grid_{side}.pt",
        )
        ref_audit = torch.load(
            audit_dirs["sequential_single"] / f"grid_{side}.pt",
            map_location="cpu", weights_only=True,
        )
        other_audit = torch.load(
            audit_dirs[execution] / f"grid_{side}.pt",
            map_location="cpu", weights_only=True,
        )
        prediction = (other_audit["predictions"] - ref_audit["predictions"]).abs()
        prediction_max = float(prediction.max())
        predictions_match = bool(torch.allclose(
            other_audit["predictions"], ref_audit["predictions"],
            atol=1e-4, rtol=1e-3,
        ))
        max_loss_diff = max(
            abs(a["loss"] - b["loss"])
            for a, b in zip(baseline["steps"], grid["steps"], strict=True)
        )
        ok = bool(
            hashes_match and audit["status"] == "pass"
            and predictions_match and max_loss_diff <= 1e-4
        )
        matched = matched and ok
        output[execution] = {
            "pass": ok, "matched_inputs_initial_weights_and_provenance": hashes_match,
            "audit": audit, "first_step_max_absolute_prediction_difference": prediction_max,
            "first_step_predictions_within_tolerance": predictions_match,
            "max_step_loss_difference": max_loss_diff,
            "speedup_over_sequential_gpu": (
                baseline["medians"]["forward_backward_seconds"]
                / grid["medians"]["forward_backward_seconds"]
            ),
            "speedup_over_sequential_end_to_end": (
                baseline["medians"]["total_step_seconds"]
                / grid["medians"]["total_step_seconds"]
            ),
            "peak_allocated_gib": grid["peak_allocated_gib"],
        }
    index = reports["packed_index"]["grids"][side]
    segmented = reports["packed_segmented"]["grids"][side]
    direct = audit_parity(
        audit_dirs["packed_index"] / f"grid_{side}.pt",
        audit_dirs["packed_segmented"] / f"grid_{side}.pt",
    )
    packed_predictions = torch.load(
        audit_dirs["packed_index"] / f"grid_{side}.pt",
        map_location="cpu", weights_only=True,
    )["predictions"]
    segmented_predictions = torch.load(
        audit_dirs["packed_segmented"] / f"grid_{side}.pt",
        map_location="cpu", weights_only=True,
    )["predictions"]
    direct_prediction_max = float((segmented_predictions - packed_predictions).abs().max())
    direct_ok = direct["status"] == "pass" and direct_prediction_max <= 1e-4
    output["segmented_vs_indexed"] = {
        "pass": direct_ok, "audit": direct,
        "first_step_max_absolute_prediction_difference": direct_prediction_max,
        "speedup_gpu": (
            index["medians"]["forward_backward_seconds"]
            / segmented["medians"]["forward_backward_seconds"]
        ),
        "speedup_end_to_end": (
            index["medians"]["total_step_seconds"]
            / segmented["medians"]["total_step_seconds"]
        ),
    }
    return {"pass": matched and direct_ok, "comparisons": output}


def run_driver(args):
    if args.steps_per_grid < 3 or args.effective_batch != 4:
        raise ValueError("M48.4f requires >=3 steps/grid and four effective patches")
    if (args.mode not in ("static", "dynamic") or args.depth < 1
            or args.core_side < 1 or args.width < 1
            or args.heads < 1 or args.width % args.heads):
        raise ValueError("invalid architecture geometry")
    revision = _git_revision()
    reps, _scales, provenance = authenticated_sources(
        args.ms_sr_root, args.normalization_file,
    )
    config = {
        "architecture": args.mode, "width": args.width, "heads": args.heads,
        "depth": args.depth, "core_side": args.core_side,
        "max_fine_read": args.max_fine_read,
        "max_coarse_read": args.max_coarse_read,
        "steps_per_grid": args.steps_per_grid, "effective_batch": args.effective_batch,
        "learning_rate": 3e-4, "grad_clip": 1.0, "seed": args.seed,
    }
    manifest = {
        "schema": "graph_attention.m48_4f_sr_execution.v1",
        "revision": revision, "configuration": config,
        "ms_sr_root": str(args.ms_sr_root.resolve()),
        "normalization_file": str(args.normalization_file.resolve()),
        "provenance": provenance, "run_order": list(MODES),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "scope": "paired real-CEDAR computational probe, not scientific training",
        "source_count": len(reps),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_file = args.output_dir / "manifest.json"
    atomic_json(manifest_file, manifest)
    print(f"ARTIFACT_MANIFEST={manifest_file}", flush=True)
    print(f"ARTIFACT_SUMMARY={args.output_dir / 'summary.json'}", flush=True)
    result = {**manifest, "status": "running", "executions": {}, "comparisons": {}}
    reports = {}
    audits = {}
    for execution in MODES:
        folder = args.output_dir / execution
        folder.mkdir()
        reports_file = folder / "report.json"
        log_file = folder / "process.log"
        audit_dir = folder / "audits"
        audits[execution] = audit_dir
        print(f"ARTIFACT_{execution.upper()}_REPORT={reports_file}", flush=True)
        print(f"ARTIFACT_{execution.upper()}_LOG={log_file}", flush=True)
        print(f"ARTIFACT_{execution.upper()}_AUDITS={audit_dir}", flush=True)
        cmd = [
            sys.executable, "-u", "-m", "scripts.m48_4f_compare",
            "--worker", "--manifest", str(manifest_file), "--execution", execution,
            "--report", str(reports_file), "--audit-dir", str(audit_dir),
        ]
        t0 = time.perf_counter()
        with log_file.open("w") as log:
            log.write(f"revision={revision} execution={execution}\n")
            log.flush()
            code = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, check=False,
            ).returncode
        receipt = json.loads(reports_file.read_text()) if reports_file.is_file() else {
            "status": "missing_report",
        }
        result["executions"][execution] = {
            "status": receipt.get("status"), "exit_code": code,
            "process_wall_seconds": time.perf_counter() - t0,
            "report": str(reports_file), "log": str(log_file),
            "audit_dir": str(audit_dir),
            "error": receipt.get("error"),
            "grids": receipt.get("grids", {}),
        }
        if code == 0 and receipt.get("status") == "ok":
            reports[execution] = receipt
        atomic_json(args.output_dir / "progress.json", result)
        print(f"[M48.4f] execution={execution} exit={code} status={receipt.get('status')}",
              flush=True)
    if len(reports) == len(MODES):
        for side in ("48", "80"):
            result["comparisons"][side] = compare_grid(reports, side, audits)
        passed = all(x["pass"] for x in result["comparisons"].values())
    else:
        passed = False
    result["status"] = "completed_not_scientifically_accepted" if passed else "failed_or_review_required"
    atomic_json(args.output_dir / "summary.json", result)
    print(f"[M48.4f] final_status={result['status']}", flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--execution", choices=MODES)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--ms-sr-root", type=Path, default=Path("/scratch/ai/MS_SR"))
    parser.add_argument("--normalization-file", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("static", "dynamic"))
    parser.add_argument("--core-side", type=int, default=12)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--effective-batch", type=int, default=4)
    parser.add_argument("--steps-per-grid", type=int, default=4)
    parser.add_argument("--seed", type=int, default=484)
    parser.add_argument("--max-fine-read", type=int, default=100000)
    parser.add_argument("--max-coarse-read", type=int, default=50000)
    args = parser.parse_args()
    if args.worker:
        if any(v is None for v in (args.manifest, args.execution, args.report, args.audit_dir)):
            parser.error("worker needs --manifest, --execution, --report, --audit-dir")
        try:
            receipt = worker(args.manifest, args.execution, args.report, args.audit_dir)
        except Exception:
            receipt = {"status": "failed", "error": traceback.format_exc()}
            atomic_json(args.report, receipt)
            print(receipt["error"], file=sys.stderr, flush=True)
            raise
        atomic_json(args.report, receipt)
    else:
        if args.output_dir is None or args.mode is None:
            parser.error("driver needs --output-dir and --mode")
        if not run_driver(args):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
