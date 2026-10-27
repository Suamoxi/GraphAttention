"""M44 SR-0.2 read-only audit of an existing SR-0.1 best checkpoint.

Evaluates held-out conditional FM loss against zero-velocity and WRONG coarse
conditions with identical FM noise/time, plus small multi-draw diagnostics.
Does NOT optimize, choose a checkpoint, or modify the source training run.
The coarse condition is point-sampled DNS, NOT a filtered LES state.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from dataclasses import replace
from typing import Any

import torch
from omegaconf import OmegaConf

from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, HITCaseSpec
from graph_attention.tasks.sr0 import SR0FlowMatching, SR0Scales
from graph_attention.tasks.sr0_evaluation import (
    STATE_FIELDS,
    ensemble_point_proxy_metrics,
    per_field_mse,
    point_sample_anchors,
    project_point_sample,
)
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.train_m44_sr0 import (
    _build_pairs, _file_record, _forward_condition, _revision, _sha256,
    _save_json, _save_torch,
)


def _source_snapshot(sample_id: str) -> str:
    if sample_id.count("/sr0_patch_") != 1:
        raise ValueError(f"unrecognized SR-0 patch sample ID: {sample_id}")
    return sample_id.split("/sr0_patch_", 1)[0]


def _origin_key(sample_id: str) -> str:
    if sample_id.count("/sr0_patch_") != 1:
        raise ValueError(f"unrecognized SR-0 patch sample ID: {sample_id}")
    return sample_id.split("/sr0_patch_", 1)[1]


def _wrong_condition_donors(pairs: list[Any]) -> dict[int, int | None]:
    """Rotate held-out snapshots WITHIN each fixed origin; never within a snapshot.

    If the split contains only one distinct source snapshot at an origin,
    counterfactuals are unavailable rather than manufacturing an OOD rotation.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(pairs):
        groups[_origin_key(pair.fine.source.sample_ids[0])].append(i)
    donors = {}
    for positions in groups.values():
        unique = {_source_snapshot(pairs[i].fine.source.sample_ids[0]) for i in positions}
        if len(unique) != len(positions):
            raise ValueError("duplicate source snapshot at the same patch origin")
        for k, i in enumerate(positions):
            donors[i] = positions[(k + 1) % len(positions)] if len(positions) > 1 else None
    return donors


def _dict_fields(values: torch.Tensor) -> dict[str, float]:
    if values.shape != (5,):
        raise ValueError("expected five conservative channels")
    return {field: float(values[j].detach().cpu()) for j, field in enumerate(STATE_FIELDS)}


def _mean_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no sample rows")
    matched = torch.tensor([row["matched_mse_by_field"] for row in rows], dtype=torch.float64)
    zero = torch.tensor([row["zero_velocity_mse_by_field"] for row in rows], dtype=torch.float64)
    result = {
        "num_patches": len(rows),
        "matched_flow_mse": float(matched.mean()),
        "zero_velocity_flow_mse": float(zero.mean()),
        "model_improvement_over_zero_mse": float((zero - matched).mean()),
        "model_improvement_over_zero_fraction": float((zero - matched).mean() / zero.mean()),
        "matched_by_field": _dict_fields(matched.mean(dim=0)),
        "zero_velocity_by_field": _dict_fields(zero.mean(dim=0)),
    }
    paired = [row for row in rows if row["wrong_mse_by_field"] is not None]
    if paired:
        wrong = torch.tensor([row["wrong_mse_by_field"] for row in paired], dtype=torch.float64)
        matched_paired = torch.tensor([row["matched_mse_by_field"] for row in paired], dtype=torch.float64)
        result.update({
            "counterfactual_patches": len(paired),
            "wrong_coarse_flow_mse": float(wrong.mean()),
            "wrong_minus_matched_mse": float((wrong - matched_paired).mean()),
            "wrong_minus_matched_by_field": _dict_fields((wrong - matched_paired).mean(dim=0)),
        })
    else:
        result.update({
            "counterfactual_patches": 0,
            "wrong_coarse_flow_mse": None,
            "wrong_minus_matched_mse": None,
            "wrong_minus_matched_by_field": None,
        })
    return result


def _evaluate_split(
    model: torch.nn.Module,
    task: SR0FlowMatching,
    pairs: list[Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    donors = _wrong_condition_donors(pairs)
    rows = []
    # Cache transfer on a per-patch basis only, not the huge full native DNS fields.
    with torch.inference_mode():
        for i, original in enumerate(pairs):
            pair = original.to(device=device, dtype=torch.float32)
            problem = task.make_validation_problem(pair)
            target = problem.flow.targets
            matched = _forward_condition(model, problem, problem.spatial_condition)
            matched_mse = per_field_mse(matched, target)
            zero_mse = per_field_mse(torch.zeros_like(target), target)
            wrong_mse = None
            donor_i = donors[i]
            if donor_i is not None:
                donor = pairs[donor_i]
                sid = original.fine.source.sample_ids[0]
                did = donor.fine.source.sample_ids[0]
                if _origin_key(sid) != _origin_key(did) or _source_snapshot(sid) == _source_snapshot(did):
                    raise RuntimeError("counterfactual must change snapshot but preserve patch origin")
                alt_pair = donor.to(device=device, dtype=torch.float32)
                if not torch.equal(pair.fine.coords, alt_pair.fine.coords):
                    raise ValueError("counterfactual coarse fields require identical fine geometry")
                alt_condition = task.spatial_condition(alt_pair)
                if alt_condition.shape != problem.spatial_condition.shape:
                    raise ValueError("counterfactual spatial condition shape mismatch")
                wrong = _forward_condition(model, problem, alt_condition)
                wrong_mse = per_field_mse(wrong, target).detach().cpu().tolist()
            rows.append({
                "sample_id": original.fine.source.sample_ids[0],
                "source_snapshot": _source_snapshot(original.fine.source.sample_ids[0]),
                "patch_origin": _origin_key(original.fine.source.sample_ids[0]),
                "donor_sample_id": pairs[donor_i].fine.source.sample_ids[0] if donor_i is not None else None,
                "matched_mse_by_field": matched_mse.detach().cpu().tolist(),
                "zero_velocity_mse_by_field": zero_mse.detach().cpu().tolist(),
                "wrong_mse_by_field": wrong_mse,
            })
    by_snapshot = defaultdict(list)
    for row in rows:
        by_snapshot[row["source_snapshot"]].append(row)
    return ({
        **_mean_rows(rows),
        "per_snapshot": {name: _mean_rows(part) for name, part in by_snapshot.items()},
        "diagnostic_caveat": (
            "The wrong condition comes from a distinct real DNS snapshot at the SAME "
            "spatial origin; the target, residual noise, time and fine mesh are unchanged. "
            "A positive loss gap supports conditional usefulness but is not SGS validation."
        ),
    }, rows)


def _ensemble_split(
    model: torch.nn.Module,
    task: SR0FlowMatching,
    pairs: list[Any],
    device: torch.device,
    *,
    draws: int,
    sample_steps: int,
    max_patches: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    # One origin per source snapshot, avoiding oversampling one DNS snapshot.
    chosen = []
    snapshots = set()
    for pair in pairs:
        sid = pair.fine.source.sample_ids[0]
        group = _source_snapshot(sid)
        if group not in snapshots:
            snapshots.add(group)
            chosen.append(pair)
            if len(chosen) >= max_patches:
                break
    results = []
    example = None
    model.eval()
    with torch.inference_mode():
        for original in chosen:
            pair = original.to(device=device, dtype=torch.float32)
            input_pair = replace(pair, fine_target=None)
            draws_list = []
            for k in range(draws):
                sample = task.generate(
                    model, input_pair, steps=sample_steps, solver="heun",
                    sampling_seed=seed,
                    sampling_keys=(f"{pair.fine.source.sample_ids[0]}:draw:{k}",),
                )
                if not torch.isfinite(sample).all():
                    raise ValueError("generated ensemble contains NaN or Inf")
                draws_list.append(sample)
            samples = torch.stack(draws_list, dim=0)
            diag = ensemble_point_proxy_metrics(
                samples, pair.fine_baseline, pair.fine_target,
                pair.coarse_state, pair.transfer,
            )
            results.append({"sample_id": pair.fine.source.sample_ids[0], **diag})
            if example is None:
                first = samples[0]
                first_projected = project_point_sample(first, pair.coarse_state, pair.transfer)
                anchors, coarse_ids = point_sample_anchors(pair.transfer)
                example = {
                    "sample_id": pair.fine.source.sample_ids[0],
                    "raw_generations": samples.detach().cpu(),
                    "first_point_projected": first_projected.detach().cpu(),
                    "reference_fine_state": pair.fine_target.detach().cpu(),
                    "coarse_state": pair.coarse_state.detach().cpu(),
                    "baseline": pair.fine_baseline.detach().cpu(),
                    "fine_coords": pair.fine.coords.detach().cpu(),
                    "anchor_fine_ids": anchors.detach().cpu(),
                    "anchor_coarse_ids": coarse_ids.detach().cpu(),
                    "sampling_keys": [f"{pair.fine.source.sample_ids[0]}:draw:{k}" for k in range(draws)],
                }
    return results, example


def _verify_manifest_inputs(manifest: dict[str, Any], dataset: AVBPMultiCaseHITDataset) -> None:
    if manifest.get("protocol") != "M44_SR0.1_POINT_SAMPLED_DNS_NOT_LES":
        raise ValueError("refusing unsupported run protocol")
    for split, records in manifest["snapshots"].items():
        for index, record in zip(manifest["snapshot_indices"][split], records, strict=True):
            if index < 0 or index >= len(dataset.sample_specs):
                raise ValueError("snapshot index outside current source inventory")
            if record["sample_id"] != dataset.sample_ids[index]:
                raise ValueError("source snapshot identity changed")
            snapshot = Path(record["path"])
            if _file_record(snapshot) != {key: record[key] for key in ("path", "bytes", "mtime_ns")}:
                raise ValueError("source snapshot file changed since training")
    ref_record = manifest["reference_file"]
    if _sha256(Path(ref_record["path"])) != ref_record["sha256"]:
        raise ValueError("M3.3 reference file changed since training")
    mesh_record = manifest["mesh_file"]
    if _file_record(Path(mesh_record["path"])) != mesh_record:
        raise ValueError("source native mesh file changed since training")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.draws < 2 or args.sample_steps < 1 or args.max_generation_patches < 1:
        raise ValueError("draws >=2, sample_steps >=1, max_generation_patches >=1 required")
    run_dir = Path(args.run_dir).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["manifest_sha256"] != _sha256(manifest_path):
        raise ValueError("best checkpoint manifest hash mismatch")
    stats = torch.load(run_dir / "scales.pt", map_location="cpu", weights_only=True)
    for key in ("coarse_mean", "coarse_std", "residual_rms"):
        if not torch.equal(stats[key], checkpoint["train_scales"][key]):
            raise ValueError(f"best checkpoint statistics differ from scales.pt: {key}")
    if tuple(stats["train_sample_ids"]) != tuple(checkpoint["train_scales"]["train_sample_ids"]):
        raise ValueError("train scale sample IDs differ from best checkpoint")

    dataset = AVBPMultiCaseHITDataset(
        root=manifest["root"], cases=(HITCaseSpec(manifest["case"], manifest["side"]),),
        reference_dir=Path(manifest["reference_file"]["path"]).parent,
        require_references=True, expected_snapshots_per_case=None,
    )
    _verify_manifest_inputs(manifest, dataset)
    mini_args = argparse.Namespace(
        side=manifest["side"], patch_cells=manifest["patch_cells"],
        refinement=manifest["refinement"],
    )
    pairs = _build_pairs(dataset, manifest, mini_args)
    if tuple(pair.fine.source.sample_ids[0] for pair in pairs["train"]) != tuple(stats["train_sample_ids"]):
        raise ValueError("saved training scaling IDs are not the current paired training data")
    scales = SR0Scales(
        stats["coarse_mean"], stats["coarse_std"],
        stats["residual_rms"], tuple(stats["train_sample_ids"]),
    )
    task = SR0FlowMatching(
        scales, validation_seed=int(manifest["seed"]) + 2000,
        time_embedding_scale=float(manifest["time_embedding_scale"]),
        spatial_condition_mode=str(manifest["spatial_condition_mode"]),
    )
    prototype = task.make_validation_problem(pairs["validation"][0].to(device=device, dtype=torch.float32))
    model_config = checkpoint["model_config"]
    cfg = OmegaConf.create(model_config)
    model, _ = instantiate_controlled_model(
        cfg, task.model_probe(prototype), seed=int(manifest["seed"]),
    )
    if model.in_channels != model_config["in_channels"] or model.out_channels != 5:
        raise ValueError("checkpoint architecture differs from data contract")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()

    start = time.perf_counter()
    evaluated = {}
    per_patch = {}
    for split in ("train", "validation", "test"):
        evaluated[split], per_patch[split] = _evaluate_split(model, task, pairs[split], device)
        print(f"[SR-0.2] {split}: matched={evaluated[split]['matched_flow_mse']:.5f}, "
              f"zero={evaluated[split]['zero_velocity_flow_mse']:.5f}, "
              f"wrong={evaluated[split]['wrong_coarse_flow_mse']}", flush=True)
    # Scientific parity gate: this exact validation problem and checkpoint
    # must reproduce the M44 SR-0.1 best-epoch validation scalar.
    stored_val = float(checkpoint["best_validation_loss"])
    actual_val = float(evaluated["validation"]["matched_flow_mse"])
    if abs(actual_val - stored_val) > 1e-4 * max(1.0, abs(stored_val)):
        raise RuntimeError(
            f"saved best validation loss does not reproduce: "
            f"stored={stored_val:.8f}, rerun={actual_val:.8f}"
        )
    ensemble = {}
    examples = {}
    for split in ("validation", "test"):
        ensemble[split], example = _ensemble_split(
            model, task, pairs[split], device,
            draws=args.draws, sample_steps=args.sample_steps,
            max_patches=args.max_generation_patches,
            seed=int(manifest["seed"]) + 3000,
        )
        if example is not None:
            examples[split] = example
    report = {
        "status": "sr0_checkpoint_diagnostics_complete",
        "source_run_dir": str(run_dir),
        "source_training_revision": checkpoint.get("git_revision"),
        "diagnostic_revision": _revision(),
        "best_validation_loss_reproduced": True,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_loss_reported": float(checkpoint["best_validation_loss"]),
        "source_manifest_sha256": _sha256(manifest_path),
        "model_configuration": model_config,
        "coarse_observation": "point_sampled_DNS_not_filtered_LES",
        "snapshot_splits": {k: len(manifest["snapshot_indices"][k]) for k in ("train", "validation", "test")},
        "flow_matching_counterfactual": evaluated,
        "generation_ensemble": ensemble,
        "draws_per_generation_patch": args.draws,
        "generation_heun_steps": args.sample_steps,
        "wall_seconds": time.perf_counter() - start,
        "interpretation": [
            "Best checkpoint is fixed; no training or model selection was performed.",
            "Wrong condition is from a different source snapshot at same patch location.",
            "Matched and wrong use IDENTICAL fine state, noise, FM time, graph and target.",
            "One DNS target for each coarse condition cannot establish conditional PDF quality.",
            "Point-sample projection replaces observed nodes only; not a conservative LES filter.",
            "Per-patch FM loss uses equal-node weights and per-field MSE.",
            "The reported test split remains held out from checkpoint selection.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    _save_json(output / "diagnostic_summary.json", report)
    _save_json(output / "flow_matching_per_patch.json", {"splits": per_patch})
    if examples:
        _save_torch(output / "ensemble_examples.pt", examples)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--max-generation-patches", type=int, default=2)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "status": report["status"],
        "source_run_dir": report["source_run_dir"],
        "best_epoch": report["best_epoch"],
        "flow_matching_summary": {
            key: {k: v for k, v in value.items() if k in (
                "matched_flow_mse", "wrong_coarse_flow_mse", "wrong_minus_matched_mse",
                "zero_velocity_flow_mse", "model_improvement_over_zero_fraction")}
            for key, value in report["flow_matching_counterfactual"].items()
        },
        "generation_patches": {k: len(v) for k, v in report["generation_ensemble"].items()},
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
