import pytest
from scripts.resume_m49_full import LEGACY, RUN_NAME, validate_migration


def provenance(sha, dirty=False):
    return {"git": {"sha": sha, "dirty": dirty}}


def test_migration_is_explicit_and_single_target():
    manifest = {"runtime_provenance": provenance(LEGACY)}
    ckpt = {"model_class": "FullDiTGraphTransformer", "epoch": 40,
            "runtime_provenance": provenance(LEGACY)}
    target = provenance("new")
    record = validate_migration(manifest, ckpt, target, RUN_NAME)
    assert record["completed_epochs_at_migration"] == 41
    ckpt.update(runtime_provenance=target, revision_migration=record)
    assert validate_migration(manifest, ckpt, target, RUN_NAME) == record
    with pytest.raises(RuntimeError):
        validate_migration(manifest, ckpt, provenance("another"), RUN_NAME)
    with pytest.raises(RuntimeError):
        validate_migration(manifest, ckpt, target, "unrelated")


def test_missing_provenance_requires_explicit_review():
    from scripts.resume_m49_full import MISSING_GIT, REVIEWED_SHA256
    manifest = {"runtime_provenance": {"git": MISSING_GIT.copy()}}
    ckpt = {"model_class": "FullDiTGraphTransformer", "epoch": 40,
            "runtime_provenance": {"git": MISSING_GIT.copy()}}
    target = provenance("new")
    with pytest.raises(RuntimeError):
        validate_migration(manifest, ckpt, target, RUN_NAME)
    evidence = {"artifact_sha256": REVIEWED_SHA256.copy(), "legacy_worktree_sha": LEGACY,
                "historical_execution_revision_verified": False}
    record = validate_migration(manifest, ckpt, target, RUN_NAME, legacy_evidence=evidence)
    assert record["from_sha"] is None
    assert record["expected_legacy_sha"] == LEGACY
    assert record["source_checkpoint_git"] == MISSING_GIT
    ckpt.update(runtime_provenance=target, revision_migration=record)
    assert validate_migration(manifest, ckpt, target, RUN_NAME, legacy_evidence=evidence) == record
    with pytest.raises(RuntimeError):
        validate_migration(manifest, ckpt, provenance("another"), RUN_NAME, legacy_evidence=evidence)
