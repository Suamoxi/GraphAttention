import pytest

from graph_attention.data import SplitManifest


def test_split_manifest_records_explicit_membership() -> None:
    manifest = SplitManifest(
        train_ids=("sample-a", "sample-b"),
        validation_ids=("sample-c",),
        test_ids=("sample-d",),
    )

    assert manifest.all_ids == ("sample-a", "sample-b", "sample-c", "sample-d")
    assert manifest.split_for("sample-a") == "train"
    assert manifest.split_for("sample-c") == "validation"
    assert manifest.split_for("sample-d") == "test"


def test_split_manifest_accepts_sequence_inputs_but_freezes_tuples() -> None:
    manifest = SplitManifest(train_ids=["sample-a"], validation_ids=["sample-b"])  # type: ignore[arg-type]

    assert manifest.train_ids == ("sample-a",)
    assert manifest.validation_ids == ("sample-b",)


def test_split_manifest_rejects_duplicate_membership() -> None:
    with pytest.raises(ValueError, match="unique across"):
        SplitManifest(train_ids=("sample-a",), test_ids=("sample-a",))


def test_split_manifest_rejects_empty_training_split_or_invalid_ids() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SplitManifest(train_ids=())
    with pytest.raises(ValueError, match="non-empty"):
        SplitManifest(train_ids=("sample-a",), validation_ids=("",))


def test_split_manifest_rejects_unknown_sample_lookup() -> None:
    manifest = SplitManifest(train_ids=("sample-a",))

    with pytest.raises(KeyError, match="sample-b"):
        manifest.split_for("sample-b")
