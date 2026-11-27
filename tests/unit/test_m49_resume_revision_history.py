"""Ordinary continuation records Git boundaries, not pinned-revision gates."""
import pytest

from scripts.train_generative import _resume_revision_history


def test_resume_history_records_each_changed_revision():
    old = {"git": {"sha": "old", "dirty": False}}
    new = {"git": {"sha": "new", "dirty": False}}
    checkpoint = {"epoch": 79, "runtime_provenance": old}
    assert _resume_revision_history(checkpoint, old) == []
    history = _resume_revision_history(checkpoint, new)
    assert history == [{"from_sha": "old", "to_sha": "new", "completed_epochs": 80}]
    checkpoint.update(runtime_provenance=new, resume_revision_history=history, epoch=99)
    assert _resume_revision_history(checkpoint, new) == history
    latest = {"git": {"sha": "latest", "dirty": False}}
    assert _resume_revision_history(checkpoint, latest) == [
        *history, {"from_sha": "new", "to_sha": "latest", "completed_epochs": 100}
    ]


@pytest.mark.parametrize("git", [{"sha": None, "dirty": None}, {"sha": "x", "dirty": True}])
def test_unidentified_or_dirty_source_still_rejected(git):
    with pytest.raises(RuntimeError, match="strict resume"):
        _resume_revision_history({"epoch": 1, "runtime_provenance": {"git": git}},
                                 {"git": {"sha": "current", "dirty": False}})
