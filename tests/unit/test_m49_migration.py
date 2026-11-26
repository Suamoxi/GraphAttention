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
