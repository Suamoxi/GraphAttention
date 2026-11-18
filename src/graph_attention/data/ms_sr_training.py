"""Teacher-only population statistics and source selection for M48.4.

The statistics refer to one fixed CEDAR filter and refinement task. They
weight each independent teacher snapshot equally, not each grid node equally,
so the larger ReT100 grids cannot dominate merely by having more cells.
No CEDAR operation is performed on patches.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .ms_sr_adapter import MANIFEST_SHA256, flat_channels, load_ms_sr_pair


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def record_identity(record: dict) -> dict:
    """Only immutable, physically meaningful identifiers enter the selection seal."""
    return {
        "role": record["role"],
        "case": record["case"],
        "snapshot_id": str(record["snapshot_id"]),
        "source_sha256": record["source_sha256"],
        "operator_sha256": record["operator_sha256"],
        "filter_id": record["filter_id"],
        "fine_factor": int(record["fine_factor"]),
        "coarse_factor": int(record["coarse_factor"]),
    }


def select_role_records(
    root: Path, *, role: str, filter_id: str = "w1p25_l1",
    case_prefix: str = "ReT060/", max_sources: int = 0,
    manifest_sha256: str = MANIFEST_SHA256,
) -> tuple[list[dict], str]:
    """Select unique sources from the *published* campaign, never locked tests.

    max_sources=0 selects all eligible sources. A positive limit cycles over
    sorted cases before selecting another snapshot of the same case; this
    makes a small qualification subset cover more than one flow condition.
    """
    if role not in ("teacher", "validation"):
        raise ValueError("M48.4 cannot select locked_in_regime records")
    if filter_id not in ("w1p25_l1", "w1p5_l1") or max_sources < 0:
        raise ValueError("invalid filter or max_sources")
    raw = (Path(root) / "dataset" / "manifest.json").read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != manifest_sha256:
        raise ValueError(f"MS_SR manifest hash mismatch: {actual}")
    manifest = json.loads(raw)
    if manifest.get("schema") != "ms-sr.campaign.v1":
        raise ValueError("unexpected MS_SR manifest schema")
    records = [
        r for r in manifest["pairs"]
        if r["role"] == role and r["source_role"] == role
        and r["filter_id"] == filter_id
        and int(r["fine_factor"]) == 4 and int(r["coarse_factor"]) == 8
        and r["case"].startswith(case_prefix)
    ]
    records.sort(key=lambda r: (
        r["case"], float(r["physical_time_s"]), int(r["snapshot_id"])
    ))
    identities = [(r["case"], str(r["snapshot_id"])) for r in records]
    if not records or len(set(identities)) != len(records):
        raise ValueError("empty or duplicate MS_SR source selection")
    if max_sources:
        cases = sorted({r["case"] for r in records})
        groups = {case: [r for r in records if r["case"] == case] for case in cases}
        round_robin = []
        for i in range(max(len(group) for group in groups.values())):
            for case in cases:
                if i < len(groups[case]):
                    round_robin.append(groups[case][i])
        records = round_robin[:max_sources]
    return records, _digest([record_identity(r) for r in records])


def fit_teacher_normalization(
    root: Path, records: list[dict], selection_sha256: str,
    *, manifest_sha256: str = MANIFEST_SHA256,
) -> dict:
    """Fit fixed coarse mean/std and residual RMS from teacher sources only.

    Each full-field pair is obtained from the audited MS_SR loader and its
    original globally solved TorchCEDAR. Never touch validation/test sources.
    """
    if not records or any(r["role"] != "teacher" or r["source_role"] != "teacher"
                          for r in records):
        raise ValueError("normalization must use teacher records exclusively")
    if selection_sha256 != _digest([record_identity(r) for r in records]):
        raise ValueError("selection seal does not match normalization sources")
    means, variances, residual_powers = [], [], []
    for index, record in enumerate(records, start=1):
        pair, _, _ = load_ms_sr_pair(
            Path(root), role="teacher",
            filter_id=record["filter_id"],
            fine_factor=int(record["fine_factor"]),
            coarse_factor=int(record["coarse_factor"]),
            case_prefix=record["case"],
            snapshot_case=record["case"],
            snapshot_id=record["snapshot_id"],
            expected_manifest_sha256=manifest_sha256,
        )
        coarse = flat_channels(pair.coarse)
        residual = flat_channels(pair.residual)
        local_mean = coarse.mean(dim=0)
        means.append(local_mean)
        variances.append((coarse - local_mean).square().mean(dim=0))
        residual_powers.append(residual.square().mean(dim=0))
        print(
            f"normalization {index}/{len(records)} "
            f"{record['case']} #{record['snapshot_id']}", flush=True,
        )
    mean = torch.stack(means).mean(dim=0)
    variance = torch.stack([
        var + (mu - mean).square()
        for mu, var in zip(means, variances)
    ]).mean(dim=0)
    coarse_std = variance.sqrt()
    residual_rms = torch.stack(residual_powers).mean(dim=0).sqrt()
    if (not torch.isfinite(mean).all()
            or not torch.isfinite(coarse_std).all()
            or not torch.isfinite(residual_rms).all()
            or torch.any(coarse_std <= 1e-12)
            or torch.any(residual_rms <= 1e-12)):
        raise ValueError("nonfinite/degenerate teacher population scales")
    return {
        "schema": "graph_attention.m48_teacher_normalization.v1",
        "role": "teacher",
        "manifest_sha256": manifest_sha256,
        "selection_sha256": selection_sha256,
        "source_count": len(records),
        "filter_id": records[0]["filter_id"],
        "fine_factor": 4,
        "coarse_factor": 8,
        "weighting": "equal_independent_teacher_snapshot",
        "coarse_mean": mean.tolist(),
        "coarse_std": coarse_std.tolist(),
        "residual_rms": residual_rms.tolist(),
        "sources": [record_identity(r) for r in records],
    }


def checked_normalization(
    payload: dict, records: list[dict], selection_sha256: str,
    *, manifest_sha256: str = MANIFEST_SHA256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reject stale/leaked normalization before loading any training snapshot."""
    if (payload.get("schema") != "graph_attention.m48_teacher_normalization.v1"
            or payload.get("role") != "teacher"
            or payload.get("weighting") != "equal_independent_teacher_snapshot"
            or payload.get("manifest_sha256") != manifest_sha256
            or payload.get("selection_sha256") != selection_sha256
            or payload.get("sources") != [record_identity(r) for r in records]
            or payload.get("source_count") != len(records)
            or payload.get("filter_id") != records[0]["filter_id"]
            or payload.get("fine_factor") != 4
            or payload.get("coarse_factor") != 8):
        raise ValueError("normalization provenance does not match teacher selection")
    tensors = tuple(
        torch.as_tensor(payload[key], dtype=torch.float64)
        for key in ("coarse_mean", "coarse_std", "residual_rms")
    )
    if any(value.shape != (5,) or not torch.isfinite(value).all() for value in tensors):
        raise ValueError("invalid normalization components")
    if any(torch.any(value <= 1e-12) for value in tensors[1:]):
        raise ValueError("degenerate normalization scales")
    return tensors
