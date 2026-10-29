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
import math
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
    ensemble_residual_moments_and_increments,
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



def _snapshot_generation_moments(
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Pool equal-node patch second moments *within* each source snapshot.

    Each snapshot, not each generated draw or spatial patch, is the
    independent reporting unit. No temporal decorrelation is claimed.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[_source_snapshot(entry["sample_id"])].append(entry)
    output: dict[str, dict[str, Any]] = {}
    for snapshot, patches in sorted(grouped.items()):
        if len(set(p["sample_id"] for p in patches)) != len(patches):
            raise ValueError("repeated source/patch ID in generation ensemble")
        fields: dict[str, Any] = {}
        for field in STATE_FIELDS:
            def average(key: str) -> float:
                values = [
                    float(p["residual_spatial_moments"]["second_moment_decomposition"][key][field])
                    for p in patches
                ]
                if not all(math.isfinite(v) and v >= 0 for v in values):
                    raise ValueError("nonfinite or negative residual second moment")
                return sum(values) / len(values)
            mean_sq = average("ensemble_mean_square_by_field")
            variance = average("stochastic_variance_by_field")
            generated = average("generated_second_moment_by_field")
            dns = average("dns_reference_second_moment_by_field")
            if not math.isclose(mean_sq + variance, generated, rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError("per-snapshot ensemble decomposition does not close")
            if dns <= 0:
                raise ValueError("DNS reference second moment must be positive")
            fields[field] = {
                "ensemble_mean_square": mean_sq,
                "stochastic_variance": variance,
                "generated_second_moment": generated,
                "dns_reference_second_moment": dns,
                "ensemble_mean_rms_over_dns": math.sqrt(mean_sq / dns),
                "ensemble_std_rms_over_dns": math.sqrt(variance / dns),
                "generated_rms_over_dns": math.sqrt(generated / dns),
                "mean_fraction_of_generated_second_moment": (
                    mean_sq / generated if generated > 0 else None
                ),
                "variance_fraction_of_generated_second_moment": (
                    variance / generated if generated > 0 else None
                ),
            }
        output[snapshot] = {
            "patches": len(patches),
            "sample_ids": sorted(p["sample_id"] for p in patches),
            "draws_per_patch": patches[0]["draws"],
            "fields": fields,
        }
    return output


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
            free = task.point_sample_free_mask(pair)
            matched = _forward_condition(model, problem, problem.spatial_condition)
            matched_mse = per_field_mse(matched, target, node_mask=free)
            zero_mse = per_field_mse(torch.zeros_like(target), target, node_mask=free)
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
                wrong_mse = per_field_mse(wrong, target, node_mask=free).detach().cpu().tolist()
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



def _verified_point_proxy_free_mask(pair: Any) -> torch.Tensor:
    """Shared unobserved-node support for comparing two SR-0 FM protocols.

    Point-sampled DNS proxy only: require tagged source and exact observed
    coarse/fine identity. This is not a physical LES/SGS restriction.
    """
    if any(
        metadata.get("sr0_pilot") != "point_sampled_dns_not_les"
        for metadata in pair.fine.source.sample_metadata
    ):
        raise ValueError("common-support evaluation requires explicitly tagged point-sampled DNS")
    anchors, coarse_ids = point_sample_anchors(pair.transfer)
    observed = pair.coarse_state[coarse_ids]
    if not torch.allclose(pair.fine_baseline[anchors], observed, atol=1e-5, rtol=1e-6):
        raise ValueError("common-support baseline fails exact point-sample identity")
    if pair.fine_target is None or not torch.allclose(
        pair.fine_target[anchors], observed, atol=1e-5, rtol=1e-6
    ):
        raise ValueError("common-support fine target fails point-sample identity")
    free = torch.ones(
        pair.fine.coords.shape[0], dtype=torch.bool, device=pair.fine.coords.device
    )
    free[anchors] = False
    for graph in range(pair.fine.num_graphs):
        start, end = (int(v) for v in pair.fine.ptr[graph : graph + 2])
        if not bool(free[start:end].any()):
            raise ValueError("common-support evaluation has a graph without unobserved nodes")
    return free


def _evaluate_repeated_paths(
    model: torch.nn.Module,
    task: SR0FlowMatching,
    pairs: list[Any],
    device: torch.device,
    *,
    repeats: int,
) -> dict[str, Any]:
    """Repeat paired FM counterfactuals over deterministic, independent paths.

    Within each repeated path, true and wrong coarse fields see identical
    noisy residual, time, fine graph, and velocity target. Each snapshot
    contributes equally to the final summary; variations across FM paths
    are distinct from variations across source snapshots.
    """
    if repeats < 2:
        raise ValueError("repeated-path evaluation needs at least two paths")
    donors = _wrong_condition_donors(pairs)
    staged = [pair.to(device=device, dtype=torch.float32) for pair in pairs]
    model.eval()
    per_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    common_by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with torch.inference_mode():
        conditions = [task.spatial_condition(pair) for pair in staged]
        common_free_masks = [_verified_point_proxy_free_mask(pair) for pair in staged]
        if task.point_sample_constraint:
            for pair, mask in zip(staged, common_free_masks, strict=True):
                if not torch.equal(task.point_sample_free_mask(pair), mask):
                    raise RuntimeError("constrained and common point-node supports differ")
        for i, donor in donors.items():
            if donor is not None and not torch.equal(
                staged[i].fine.coords, staged[donor].fine.coords
            ):
                raise ValueError("repeated counterfactual requires identical fine geometry")
        for path_index in range(repeats):
            # Path zero must reproduce the original fixed-path diagnostic.
            # Subsequent paths change only the deterministic noise/time seed.
            seed = task.flow.validation_seed + path_index * 1000003
            path_task = SR0FlowMatching(
                task.scales,
                validation_seed=seed,
                time_embedding_scale=task.flow.time_embedding_scale,
                spatial_condition_mode=task.spatial_condition_mode,
                point_sample_constraint=task.point_sample_constraint,
            )
            for index, pair in enumerate(staged):
                problem = path_task.make_validation_problem(pair)
                free = path_task.point_sample_free_mask(pair)
                reference = problem.flow.targets
                common_free = common_free_masks[index]
                matched_velocity = _forward_condition(model, problem, conditions[index])
                zero_velocity = torch.zeros_like(reference)
                correct = per_field_mse(
                    matched_velocity, reference, node_mask=free
                ).detach().cpu()
                zero = per_field_mse(
                    zero_velocity, reference, node_mask=free
                ).detach().cpu()
                common_correct = per_field_mse(
                    matched_velocity, reference, node_mask=common_free
                ).detach().cpu()
                common_zero = per_field_mse(
                    zero_velocity, reference, node_mask=common_free
                ).detach().cpu()
                donor_index = donors[index]
                wrong_velocity = (
                    _forward_condition(model, problem, conditions[donor_index])
                    if donor_index is not None else None
                )
                wrong = (
                    per_field_mse(wrong_velocity, reference, node_mask=free).detach().cpu()
                    if wrong_velocity is not None else None
                )
                common_wrong = (
                    per_field_mse(wrong_velocity, reference, node_mask=common_free).detach().cpu()
                    if wrong_velocity is not None else None
                )
                row = {
                    "path_index": path_index,
                    "sample_id": pairs[index].fine.source.sample_ids[0],
                    "matched": correct,
                    "zero": zero,
                    "wrong": wrong,
                }
                all_rows.append(row)
                per_snapshot[_source_snapshot(row["sample_id"])].append(row)
                common_row = {
                    "path_index": path_index,
                    "sample_id": row["sample_id"],
                    "matched": common_correct,
                    "zero": common_zero,
                    "wrong": common_wrong,
                }
                common_rows.append(common_row)
                common_by_snapshot[_source_snapshot(row["sample_id"])].append(common_row)

    def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        matched = torch.stack([row["matched"] for row in rows]).double()
        zero = torch.stack([row["zero"] for row in rows]).double()
        result: dict[str, Any] = {
            "matched_flow_mse": float(matched.mean()),
            "zero_velocity_flow_mse": float(zero.mean()),
            "improvement_over_zero_mse": float((zero - matched).mean()),
            "matched_by_field": _dict_fields(matched.mean(dim=0)),
        }
        paired = [row for row in rows if row["wrong"] is not None]
        if paired:
            wrong = torch.stack([row["wrong"] for row in paired]).double()
            paired_matched = torch.stack([row["matched"] for row in paired]).double()
            gap = wrong - paired_matched
            result.update({
                "counterfactual_paths": len(paired),
                "wrong_coarse_flow_mse": float(wrong.mean()),
                "wrong_minus_matched_mse": float(gap.mean()),
                "wrong_minus_matched_by_field": _dict_fields(gap.mean(dim=0)),
            })
        else:
            result.update({
                "counterfactual_paths": 0,
                "wrong_coarse_flow_mse": None,
                "wrong_minus_matched_mse": None,
                "wrong_minus_matched_by_field": None,
            })
        return result

    summary = _aggregate(all_rows)
    by_snapshot = {name: _aggregate(rows) for name, rows in per_snapshot.items()}
    common_snapshots = {
        name: _aggregate(rows) for name, rows in common_by_snapshot.items()
    }
    common_gaps = [
        row["wrong_minus_matched_mse"] for row in common_snapshots.values()
        if row["wrong_minus_matched_mse"] is not None
    ]
    common_support_summary = _aggregate(common_rows)
    common_support_summary.update({
        "fine_nodes_per_patch": int(staged[0].fine.coords.shape[0]),
        "coarse_nodes_per_patch": int(staged[0].transfer.num_coarse_nodes),
        "unobserved_nodes_per_patch": int(common_free_masks[0].sum()),
        "per_snapshot": common_snapshots,
        "snapshot_positive_gap_count": sum(gap > 0 for gap in common_gaps),
        "snapshot_with_counterfactual_count": len(common_gaps),
        "comparison_protocol": (
            "All models scored only on the same unobserved fine nodes. "
            "The random time and Gaussian at those nodes are identical for the "
            "same seed/sample ID. The unconstrained model sees noisy observed-node "
            "states, whereas the constrained model sees zero residual there; "
            "this is a comparison of native trained protocols, not identical full inputs."
        ),
    })
    gaps = [
        value["wrong_minus_matched_mse"]
        for value in by_snapshot.values()
        if value["wrong_minus_matched_mse"] is not None
    ]
    repeat_zero = [row for row in all_rows if row["path_index"] == 0]
    summary.update({
        "paths_per_patch": repeats,
        "num_patches": len(pairs),
        "num_snapshots": len(by_snapshot),
        "repeat_zero_matched_flow_mse": float(
            torch.stack([row["matched"] for row in repeat_zero]).double().mean()
        ),
        "snapshot_positive_gap_count": sum(value > 0 for value in gaps),
        "snapshot_with_counterfactual_count": len(gaps),
        "snapshot_gap_mean": sum(gaps) / len(gaps) if gaps else None,
        "snapshot_gap_std": (
            float(torch.tensor(gaps, dtype=torch.float64).std(unbiased=True))
            if len(gaps) > 1 else None
        ),
        "per_snapshot": by_snapshot,
        "common_unobserved_node_support": common_support_summary,
        "warning": (
            "Repeated FM time/noise probes reduce path randomness but do not "
            "increase the number of independent DNS snapshots. Snapshot gaps "
            "are descriptive, not inferential confidence intervals."
        ),
    })
    return summary


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
    all_origins: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    # max_patches bounds distinct source snapshots. When all_origins=True,
    # include every fixed patch origin for each selected source snapshot.
    chosen = []
    snapshots = set()
    for pair in pairs:
        sid = pair.fine.source.sample_ids[0]
        group = _source_snapshot(sid)
        if group in snapshots:
            if all_origins:
                chosen.append(pair)
            continue
        if len(snapshots) >= max_patches:
            break
        snapshots.add(group)
        chosen.append(pair)
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
            diag["residual_spatial_moments"] = ensemble_residual_moments_and_increments(
                samples, pair.fine_baseline, pair.fine_target,
                pair.fine.coords, pair.fine.edge_index,
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
    if "train_patch_pool_origins" in manifest:
        frozen = manifest.get("frozen_training_scales")
        if not isinstance(frozen, dict):
            raise ValueError("SR-0.4 diagnostics require pinned original train-only scaling")
        source = Path(frozen["source_run_dir"])
        if _sha256(source / "manifest.json") != frozen["source_manifest_sha256"]:
            raise ValueError("SR-0.4 scaling source manifest changed")
        if _sha256(source / "scales.pt") != frozen["source_scales_sha256"]:
            raise ValueError("SR-0.4 scaling source statistics changed")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.draws < 2 or args.sample_steps < 1 or args.max_generation_patches < 1:
        raise ValueError("draws >=2, sample_steps >=1, max_generation_patches >=1 required")
    path_repeats = int(getattr(args, "path_repeats", 1))
    if path_repeats < 1:
        raise ValueError("path_repeats must be >=1")
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
    # Compare the fixed original training origins rather than all 16
    # spatially augmented training origins. This preserves identical
    # source sample IDs and conditions with the SR-0.3 baseline.
    diverse_training = "train_patch_pool_origins" in manifest
    pairs = _build_pairs(
        dataset, manifest, mini_args, training_comparison_only=diverse_training
    )
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
        point_sample_constraint=bool(manifest.get("point_sample_constraint", False)),
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
    repeated_paths: dict[str, Any] = {}
    if path_repeats > 1:
        for split in ("train", "validation", "test"):
            repeated = _evaluate_repeated_paths(
                model, task, pairs[split], device, repeats=path_repeats,
            )
            if abs(
                repeated["repeat_zero_matched_flow_mse"] - evaluated[split]["matched_flow_mse"]
            ) > 1e-5:
                raise RuntimeError(f"repeated path-0 parity failed for {split}")
            repeated_paths[split] = repeated
            print(
                f"[SR-0.2] {split} repeated x{path_repeats}: "
                f"matched={repeated['matched_flow_mse']:.5f} "
                f"gap={repeated['wrong_minus_matched_mse']}",
                flush=True,
            )
    ensemble = {}
    examples = {}
    for split in ("validation", "test"):
        ensemble[split], example = _ensemble_split(
            model, task, pairs[split], device,
            draws=args.draws, sample_steps=args.sample_steps,
            max_patches=args.max_generation_patches,
            seed=int(manifest["seed"]) + 3000,
            all_origins=bool(getattr(args, "all_generation_origins", False)),
        )
        if bool(getattr(args, "all_generation_origins", False)):
            counts = _snapshot_generation_moments(ensemble[split])
            expected = len(manifest["patch_origins"])
            if any(row["patches"] != expected for row in counts.values()):
                raise ValueError("all-origin diagnostic did not cover every fixed origin")
        if example is not None:
            examples[split] = example
    continuation_path = run_dir / "continuation_provenance.json"
    continuation = None
    if continuation_path.is_file():
        continuation = json.loads(continuation_path.read_text())
        if continuation.get("protocol") != "M44_SR05_IMMUTABLE_CONTINUATION_V1":
            raise ValueError("unrecognized continuation provenance protocol")
        if continuation.get("source_manifest_sha256") != _sha256(manifest_path):
            raise ValueError("continued checkpoint manifest differs from parent")
        parent = Path(continuation["source_run_dir"])
        if _sha256(parent / "last.pt") != continuation["source_last_sha256"]:
            raise ValueError("parent training checkpoint changed after cloning")
        # best.pt is selected on validation and may predate the cloning
        # point. Audit *last.pt* for total optimizer-budget completion.
        last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        target_epochs = int(continuation["target_epochs"])
        initial_epochs = int(continuation["initial_epochs"])
        if last["epoch"] != target_epochs - 1 or len(last["history"]) != target_epochs:
            raise ValueError("continued last checkpoint has not finished the target epochs")
        if last["manifest_sha256"] != continuation["source_manifest_sha256"]:
            raise ValueError("continued last checkpoint manifest differs from parent")
        parent_last = torch.load(parent / "last.pt", map_location="cpu", weights_only=True)
        if last["history"][:initial_epochs] != parent_last["history"]:
            raise ValueError("continuation did not preserve original epoch history")
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError("continued run has not completed its training summary")
        training_summary = json.loads(summary_path.read_text())
        completed = int(training_summary["epochs_completed"])
        if completed != int(continuation["target_epochs"]):
            raise ValueError("continuation has not completed the target epoch budget")
        if int(training_summary["best_epoch"]) != int(checkpoint["best_epoch"]):
            raise ValueError("best checkpoint disagrees with completed training summary")
    else:
        completed = None
    ensemble_by_snapshot = {
        split: _snapshot_generation_moments(ensemble[split])
        for split in ("validation", "test")
    }
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
        "point_sample_constraint": task.point_sample_constraint,
        "flow_mse_support": (
            "unobserved_fine_nodes" if task.point_sample_constraint else "all_fine_nodes"
        ),
        "train_evaluation_patch_scope": (
            "original_fixed_train_origins_only" if diverse_training else "all_training_patches"
        ),
        "train_patch_diversity": manifest.get("train_patch_schedule"),
        "continuation_provenance": continuation,
        "completed_training_epochs": completed,
        "frozen_scale_provenance": manifest.get("frozen_training_scales"),
        "snapshot_splits": {k: len(manifest["snapshot_indices"][k]) for k in ("train", "validation", "test")},
        "flow_matching_counterfactual": evaluated,
        "repeated_flow_matching": repeated_paths,
        "generation_ensemble": ensemble,
        "generation_ensemble_by_snapshot": ensemble_by_snapshot,
        "all_generation_origins": bool(getattr(args, "all_generation_origins", False)),
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
            "SR-0.4 diagnostics use original fixed train origins for a paired baseline comparison.",
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
    parser.add_argument("--max-generation-patches", type=int, default=2,
                        help="maximum distinct source snapshots for ensemble generation")
    parser.add_argument("--all-generation-origins", action="store_true",
                        help="sample every fixed patch origin per selected snapshot")
    parser.add_argument("--path-repeats", type=int, default=1,
                        help="independent, deterministic FM paths per patch; 1 disables repetition")
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
