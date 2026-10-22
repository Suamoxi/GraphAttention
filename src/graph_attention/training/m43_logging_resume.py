"""Strict, opt-in M43-B logging-only compatibility for legacy ff85188 checkpoints.

The checkpoint and its run_signature.json remain immutable. A changed Git
revision is NEVER silently accepted. The only supported migration is from
the explicitly known M43-B pilot source revision, with all scientific
manifest fields byte-for-byte equivalent and every changed Git path inside
a narrow allowlist reviewed for logging/infrastructure-only changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


ORIGINAL_M43_B_REVISION = "ff851882091e4c4fd4ccfd5ecb69cb02b8bd0d90"

# This whitelist is intentionally narrow. Any later scientific code changes
# elsewhere force a new run signature; do not expand it just to force resume.
LOGGING_ONLY_APPROVED_PATHS = frozenset({
    "scripts/train_m43_b_multires.py",
    "src/graph_attention/training/m43_tensorboard.py",
    "src/graph_attention/training/m43_logging_resume.py",
    "scripts/slurm/m43_b_multires_train.slurm",
    "tests/unit/test_m43_b_multires.py",
    "tests/unit/test_m43_b_tensorboard.py",
    "docs/M43_B_MIXED_RESOLUTION_TRAINING.md",
})


def _signature_sha256(content: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(content), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _run_git(repo_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        capture_output=True, text=True, check=False,
    )


def changed_files_from_original(repo_root: Path, current_revision: str) -> list[str]:
    """Require an ancestry relationship and inspect *all* changed paths."""
    ancestor = _run_git(
        repo_root, "merge-base", "--is-ancestor",
        ORIGINAL_M43_B_REVISION, current_revision,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            "new code is not verifiably descended from the original "
            "M43-B training revision; refusing legacy checkpoint resume"
        )
    diff = _run_git(
        repo_root, "diff", "--name-only",
        f"{ORIGINAL_M43_B_REVISION}..{current_revision}",
    )
    if diff.returncode != 0:
        raise RuntimeError(
            "cannot verify logging-only revision delta: " + diff.stderr.strip()
        )
    changed = sorted({line.strip() for line in diff.stdout.splitlines() if line.strip()})
    if not changed or "scripts/train_m43_b_multires.py" not in changed:
        raise RuntimeError(
            "expected M43-B logging-only training runner delta not present"
        )
    unexpected = set(changed) - LOGGING_ONLY_APPROVED_PATHS
    if unexpected:
        raise RuntimeError(
            "non-logging source paths changed since M43-B pilot; refusing "
            "signature migration: " + ", ".join(sorted(unexpected))
        )
    return changed


def reconcile_resume_signature(
    stored: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    allow_logging_only_revision: bool,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return the authoritative *stored* signature if migration is authorized.

    Do not rewrite run_signature.json. A legacy checkpoint is always checked
    against its original SHA. This allows TensorBoard artifacts to be added
    without reinitializing the model, optimizer, RNG or train-only scaler.
    """
    if stored == candidate:
        return dict(candidate), None
    if not allow_logging_only_revision:
        raise RuntimeError(
            "resume signature changed; refusing mismatched run. For the "
            "strictly logging-only migration from ff85188, explicitly opt in "
            "with --allow-logging-only-resume"
        )
    original = stored.get("content")
    proposed = candidate.get("content")
    if not isinstance(original, dict) or not isinstance(proposed, dict):
        raise RuntimeError("malformed M43-B run signatures")
    if original.get("revision") != ORIGINAL_M43_B_REVISION:
        raise RuntimeError(
            "logging-only migration is permitted only for the original "
            "ff85188 M43-B pilot revision"
        )
    current_revision = proposed.get("revision")
    if not isinstance(current_revision, str) or not current_revision:
        raise RuntimeError("current Git revision cannot be verified")
    recomputed_original = dict(proposed)
    recomputed_original["revision"] = ORIGINAL_M43_B_REVISION
    if recomputed_original != original:
        raise RuntimeError(
            "scientific manifest differs beyond the allowed Git revision; "
            "cannot reuse training checkpoint"
        )
    if _signature_sha256(original) != stored.get("sha256"):
        raise RuntimeError("stored scientific run manifest has invalid SHA256")
    if _signature_sha256(proposed) != candidate.get("sha256"):
        raise RuntimeError("new scientific run manifest has invalid SHA256")
    changed = changed_files_from_original(repo_root, current_revision)
    audit = {
        "status": "explicit_logging_only_revision_migration",
        "original_training_revision": ORIGINAL_M43_B_REVISION,
        "executing_revision": current_revision,
        "original_signature_sha256": stored["sha256"],
        "scientific_manifest_except_revision_unchanged": True,
        "changed_paths_since_original": changed,
        "compatibility_gate": (
            "source revision descended from original, changed paths restricted "
            "to reviewed logging/infrastructure allowlist; this check is NOT "
            "proof of numerical equivalence without the targeted parity tests"
        ),
        "resume_state_policy": (
            "reuse original run_signature.json and last.pt signature; "
            "load original model, AdamW, training generator and scalers unchanged"
        ),
    }
    return dict(stored), audit
