"""M48 teacher-only, case-resolved normalization diagnostics.

Uses the same original MS_SR loader and full-domain TorchCEDAR inverse as
M48.4a. All moments are float64 and all averaging weights independent
teacher snapshots equally, regardless of mesh size. This is an audit only:
it never fits a per-case training scaler or writes to the MS_SR dataset.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from .ms_sr_adapter import FIELDS, MANIFEST_SHA256, flat_channels, load_ms_sr_pair
from .ms_sr_training import _digest, record_identity


def _source_moments(coarse: torch.Tensor, residual: torch.Tensor) -> dict:
    c = flat_channels(coarse)
    r = flat_channels(residual)
    if (c.dtype != torch.float64 or r.dtype != torch.float64
            or not torch.isfinite(c).all() or not torch.isfinite(r).all()):
        raise ValueError("M48 normalization audit requires finite float64 fields")
    cm = c.mean(0)
    rm = r.mean(0)
    return {
        "coarse_mean": cm,
        "coarse_variance": (c - cm).square().mean(0),
        "residual_mean": rm,
        "residual_variance": (r - rm).square().mean(0),
        "residual_power": r.square().mean(0),
    }


def _population(moments: list[dict]) -> dict:
    """Law of total variance with one equal weight per source snapshot."""
    if not moments:
        raise ValueError("empty source population")
    coarse_means = torch.stack([m["coarse_mean"] for m in moments])
    coarse_mean = coarse_means.mean(0)
    coarse_variance = torch.stack([
        m["coarse_variance"] + (m["coarse_mean"] - coarse_mean).square()
        for m in moments
    ]).mean(0)
    residual_means = torch.stack([m["residual_mean"] for m in moments])
    residual_mean = residual_means.mean(0)
    residual_variance = torch.stack([
        m["residual_variance"] + (m["residual_mean"] - residual_mean).square()
        for m in moments
    ]).mean(0)
    residual_power = torch.stack([m["residual_power"] for m in moments]).mean(0)
    values = {
        "coarse_mean": coarse_mean,
        "coarse_std": coarse_variance.clamp_min(0).sqrt(),
        "residual_mean": residual_mean,
        "residual_std": residual_variance.clamp_min(0).sqrt(),
        "residual_rms": residual_power.clamp_min(0).sqrt(),
    }
    if any(value.shape != (5,) or not torch.isfinite(value).all()
           for value in values.values()):
        raise ValueError("nonfinite or invalid component moments")
    return values


def _to_list(tensors: dict) -> dict:
    return {name: value.tolist() for name, value in tensors.items()}


def _comparison(case: dict, pooled: dict) -> dict:
    if (torch.any(pooled["coarse_std"] <= 1e-12)
            or torch.any(pooled["residual_rms"] <= 1e-12)):
        raise ValueError("degenerate pooled coarse or residual scales")
    return _to_list({
        "coarse_mean_shift_in_global_std":
            (case["coarse_mean"] - pooled["coarse_mean"]) / pooled["coarse_std"],
        "coarse_std_over_global_std": case["coarse_std"] / pooled["coarse_std"],
        "residual_rms_over_global_rms": case["residual_rms"] / pooled["residual_rms"],
        "residual_mean_over_global_rms":
            case["residual_mean"] / pooled["residual_rms"],
        "residual_mean_over_case_rms":
            case["residual_mean"] / case["residual_rms"].clamp_min(1e-12),
    })


def audit_teacher_normalization(
    ms_sr_root: Path, records: list[dict], selection_sha256: str,
    *, manifest_sha256: str = MANIFEST_SHA256,
) -> dict:
    """Compare per-case and per-ReT groups with the full selected teacher pool.

    Every selected pair is loaded and CEDAR-qualified globally. No
    validation or locked_in_regime source is ever opened.
    """
    if not records or any(r["role"] != "teacher" or r["source_role"] != "teacher"
                          for r in records):
        raise ValueError("audit requires teacher-only source records")
    if selection_sha256 != _digest([record_identity(r) for r in records]):
        raise ValueError("teacher source selection SHA-256 mismatch")
    if len({(r["case"], str(r["snapshot_id"])) for r in records}) != len(records):
        raise ValueError("duplicate source snapshots")
    if len({(r["filter_id"], int(r["fine_factor"]), int(r["coarse_factor"]))
            for r in records}) != 1:
        raise ValueError("audit cannot mix filters or refinement pairs")

    source_moments = []
    case_moments: dict[str, list[dict]] = {}
    reynolds_moments: dict[str, list[dict]] = {}
    snapshots = []
    max_cedar_defect = 0.0
    for idx, record in enumerate(records, 1):
        pair, _, checks = load_ms_sr_pair(
            Path(ms_sr_root), role="teacher",
            filter_id=record["filter_id"],
            fine_factor=int(record["fine_factor"]),
            coarse_factor=int(record["coarse_factor"]),
            case_prefix=record["case"],
            snapshot_case=record["case"],
            snapshot_id=record["snapshot_id"],
            expected_manifest_sha256=manifest_sha256,
        )
        if (pair.metadata["role"] != "teacher"
                or pair.metadata["case"] != record["case"]
                or str(pair.metadata["snapshot_id"]) != str(record["snapshot_id"])):
            raise RuntimeError("loaded MS_SR record does not match audited source identity")
        defect = float(checks["C_residual"]["max_component_scaled_linf"])
        if not math.isfinite(defect) or defect > 1e-9:
            raise RuntimeError(f"CEDAR residual defect too large: {defect}")
        max_cedar_defect = max(max_cedar_defect, defect)

        moment = _source_moments(pair.coarse, pair.residual)
        source_moments.append(moment)
        case = record["case"]
        reynolds_group = case.split("/", 1)[0]
        case_moments.setdefault(case, []).append(moment)
        reynolds_moments.setdefault(reynolds_group, []).append(moment)
        snapshots.append({
            "source": record_identity(record),
            "physical_time_s": float(record["physical_time_s"]),
            "coarse_shape": list(pair.coarse.shape[-3:]),
            "fine_shape": list(pair.fine.shape[-3:]),
            "moments": _to_list(_population([moment])),
            "C_residual_max_component_scaled_linf": defect,
        })
        print(
            f"audited {idx}/{len(records)}: {case} "
            f"snapshot {record['snapshot_id']} C(R)={defect:.3e}",
            flush=True,
        )

    pooled = _population(source_moments)
    _comparison(pooled, pooled)  # Validate nondegenerate pooled scales.

    def groups(grouped: dict[str, list[dict]]) -> dict:
        result = {}
        for key in sorted(grouped):
            values = _population(grouped[key])
            result[key] = {
                "source_count": len(grouped[key]),
                "moments": _to_list(values),
                "vs_pooled": _comparison(values, pooled),
            }
        return result

    return {
        "schema": "graph_attention.m48_normalization_audit.v1",
        "scope": "teacher_only; no per-case scaler is fitted for training",
        "fields": list(FIELDS),
        "field_units": "as stored by MS_SR; no additional physical nondimensionalization",
        "weighting": "equal_independent_teacher_snapshot",
        "manifest_sha256": manifest_sha256,
        "selection_sha256": selection_sha256,
        "filter_id": records[0]["filter_id"],
        "fine_factor": int(records[0]["fine_factor"]),
        "coarse_factor": int(records[0]["coarse_factor"]),
        "source_count": len(records),
        "case_count": len(case_moments),
        "reynolds_group_count": len(reynolds_moments),
        "max_C_residual_component_scaled_linf": max_cedar_defect,
        "sources": [record_identity(r) for r in records],
        "pooled": _to_list(pooled),
        "per_case": groups(case_moments),
        "per_reynolds_group": groups(reynolds_moments),
        "per_snapshot": snapshots,
    }
