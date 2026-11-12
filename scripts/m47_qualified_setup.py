"""Strict reviewed-window and derived-reference gates for M47 training."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from graph_attention.data.multicase_avbp import DEFAULT_M43_CASES, reference_filename
from scripts.m46_stationarity_selection import select_snapshots
from scripts.m47_reference_conditioning import CONDITION_NAMES


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible_sample_ids(
    selected_path: Path, inventory_path: Path, policy_path: Path,
) -> tuple[tuple[str, ...], dict]:
    """Rebuild eligibility from its reviewed policy; refuse stale or edited input."""
    full = json.loads(inventory_path.read_text())
    approved = json.loads(selected_path.read_text())
    policy = json.loads(policy_path.read_text())
    if approved.get("version") != "m46_qualified_manifest_v1" or approved.get("stationarity_filter") is not True:
        raise ValueError("training requires a reviewed M46 qualified manifest")
    q = approved.get("qualification", {})
    if (q.get("source_inventory_sha256") != file_sha256(inventory_path)
        or q.get("policy_sha256") != file_sha256(policy_path)):
        raise ValueError("qualified manifest has stale inventory or policy hashes")
    expected, _, _ = select_snapshots(full, policy)
    if (approved.get("cases") != expected["cases"]
        or approved.get("samples") != expected["samples"]
        or q.get("rules") != expected["qualification"]["rules"]):
        raise ValueError("qualified manifest selection differs from reviewed policy")
    case_ids = {case.case_id for case in DEFAULT_M43_CASES}
    if set(policy["cases"]) != case_ids:
        raise ValueError("M47 is restricted to exactly the six Re54/Re94 cases")
    for rule in policy["cases"].values():
        if (rule["first_counter"] != 41 or rule["last_counter"] != 100
            or rule["transient_end_counter"] != 10 or rule["require_full_window"] is not True):
            raise ValueError("M47 requires the reviewed 41..100 window in every legacy case")
    samples = approved["samples"]
    if len(samples) != 360 or Counter(s["case_id"] for s in samples) != Counter({c: 60 for c in case_ids}):
        raise ValueError("M47 requires 60 qualified snapshots in each of six cases")
    names = tuple(s["sample_id"] for s in samples)
    if len(set(names)) != len(names):
        raise ValueError("qualified manifest contains duplicate sample IDs")
    return names, {
        "qualified_manifest_sha256": file_sha256(selected_path),
        "source_inventory_sha256": file_sha256(inventory_path),
        "stationarity_policy_sha256": file_sha256(policy_path),
        "stationarity_selection": "exploratory stationarity candidates, counters 41..100",
    }


def validate_dataset_paths(dataset, selected_path: Path) -> None:
    """Do not trust only sample IDs; bind the approved mesh/snapshot paths."""
    manifest = json.loads(selected_path.read_text())
    by_id = {s["sample_id"]: s for s in manifest["samples"]}
    actual_ids = set(dataset.sample_ids)
    if actual_ids != set(by_id):
        raise ValueError("qualified sample IDs differ from training dataset")
    for spec in dataset.sample_specs:
        expected = by_id[spec.sample_id]
        if (Path(expected["snapshot"]).resolve(strict=True) != Path(spec.snapshot_file).resolve(strict=True)
            or Path(expected["mesh"]).resolve(strict=True) != Path(spec.mesh_file).resolve(strict=True)):
            raise ValueError(f"dataset/manifest path mismatch: {spec.sample_id}")


def verify_generated_references(reference_dir: Path, audit_path: Path) -> dict:
    audit = json.loads(audit_path.read_text())
    hashes = audit.get("generated_reference_hashes")
    case_ids = {case.case_id for case in DEFAULT_M43_CASES}
    if (audit.get("version") != "m47_physics_conditions_v1"
        or audit.get("status") != "ready"
        or not isinstance(hashes, dict) or set(hashes) != case_ids):
        raise ValueError("M47 requires complete prepared physical reference audit")
    for case_id, signature in hashes.items():
        path = reference_dir / reference_filename(case_id)
        if file_sha256(path) != signature:
            raise ValueError(f"prepared case reference changed since audit: {case_id}")
    return {"m47_reference_audit_sha256": file_sha256(audit_path),
            "reference_hashes": hashes,
            "condition_encoding": "Ma_condition=Ma_ref/0.1; log_Re_condition=ln(Re_ref)",
            "supported_condition_modes": sorted(CONDITION_NAMES)}