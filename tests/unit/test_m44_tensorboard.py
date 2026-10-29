"""CPU tests for M44 native and retroactive TensorBoard scalar logging."""

import json
from pathlib import Path

import pytest

from graph_attention.training.m44_tensorboard import open_writer, validate_history, write_epoch
from scripts.backfill_m44_tensorboard import run


def _rows() -> list[dict]:
    return [
        {"epoch": 0, "train_loss": 1.2, "validation_loss": 1.4, "seconds": 1.1},
        {"epoch": 1, "train_loss": 1.0, "validation_loss": 1.1, "seconds": 1.3},
    ]


def _steps(log_dir: Path, tag: str) -> list[tuple[int, float]]:
    from tensorboard.backend.event_processing import event_accumulator

    acc = event_accumulator.EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return [(int(row.step), float(row.value)) for row in acc.Scalars(tag)]


def _assert_steps(log_dir: Path, tag: str, values: list[float]) -> None:
    observed = _steps(log_dir, tag)
    assert [step for step, _ in observed] == list(range(len(values)))
    assert [value for _, value in observed] == pytest.approx(values)


def test_history_validation_catches_nonfinite_and_discontinuous_epochs() -> None:
    validate_history(_rows())
    incorrect = _rows()
    incorrect[1] = {**incorrect[1], "epoch": 3}
    with pytest.raises(ValueError, match="contiguous"):
        validate_history(incorrect)
    incorrect = _rows()
    incorrect[1] = {**incorrect[1], "validation_loss": float("nan")}
    with pytest.raises(ValueError, match="nonfinite"):
        validate_history(incorrect)


def test_existing_json_history_import_writes_epoch_scalar_tags(tmp_path: Path) -> None:
    run_dir = tmp_path / "old_M44_run"
    run_dir.mkdir()
    manifest = {
        "protocol": "M44_SR0.1_POINT_SAMPLED_DNS_NOT_LES",
        "optimizer": {"lr": 0.001},
        "patch_origins": [[1, 1, 1], [9, 9, 9]],
        "snapshot_indices": {"train": list(range(15))},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "history.json").write_text(json.dumps({"history": _rows()}))
    result = run(run_dir)
    assert result["epochs_imported"] == 2
    assert result["optimizer_updates_per_epoch"] == 30
    assert list((run_dir / "tensorboard").glob("events.out.tfevents.*"))
    _assert_steps(run_dir / "tensorboard", "loss/validation_fm_velocity_mse", [1.4, 1.1])
    _assert_steps(run_dir / "tensorboard", "loss/best_validation_fm_velocity_mse", [1.4, 1.1])
    assert _steps(run_dir / "tensorboard", "optimizer/updates_per_epoch") == [
        (0, 30.0), (1, 30.0)
    ]
    with pytest.raises(FileExistsError, match="duplicate"):
        run(run_dir)
    result = run(run_dir, refresh=True)
    assert result["refreshed_existing_events"] is True
    _assert_steps(run_dir / "tensorboard", "loss/train_fm_velocity_mse", [1.2, 1.0])


def test_resume_existing_events_adds_only_restored_future_epoch(tmp_path: Path) -> None:
    path = tmp_path / "tensorboard"
    with open_writer(
        path, previous_history=[], resume_step=None,
        learning_rate=0.001, updates_per_epoch=30,
    ) as writer:
        write_epoch(writer, _rows()[0], best_validation=1.4,
                    learning_rate=0.001, updates_per_epoch=30)
    with open_writer(
        path, previous_history=_rows()[:1], resume_step=1,
        learning_rate=0.001, updates_per_epoch=30,
    ) as writer:
        write_epoch(writer, _rows()[1], best_validation=1.1,
                    learning_rate=0.001, updates_per_epoch=30)
    _assert_steps(path, "loss/validation_fm_velocity_mse", [1.4, 1.1])
    with pytest.raises(ValueError, match="resume_step"):
        open_writer(
            path, previous_history=_rows(), resume_step=1,
            learning_rate=0.001, updates_per_epoch=30,
        )


def test_resume_without_events_backfills_prior_epochs(tmp_path: Path) -> None:
    path = tmp_path / "tensorboard"
    with open_writer(
        path, previous_history=_rows()[:1], resume_step=1,
        learning_rate=0.001, updates_per_epoch=30,
    ) as writer:
        write_epoch(writer, _rows()[1], best_validation=1.1,
                    learning_rate=0.001, updates_per_epoch=30)
    _assert_steps(path, "loss/validation_fm_velocity_mse", [1.4, 1.1])


def test_backfill_respects_diversity_update_budget(tmp_path: Path) -> None:
    run_dir = tmp_path / "sr04_run"
    run_dir.mkdir()
    manifest = {
        "protocol": "M44_SR0.1_POINT_SAMPLED_DNS_NOT_LES",
        "optimizer": {"lr": 0.001},
        "patch_origins": [[1, 1, 1], [9, 9, 9]],
        "snapshot_indices": {"train": list(range(15))},
        "train_patch_schedule": {
            "patches_per_snapshot_per_epoch": 2,
            "pool_size": 16,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "history.json").write_text(json.dumps({"history": _rows()}))
    result = run(run_dir)
    assert result["optimizer_updates_per_epoch"] == 30
