"""M43-B logging-only TensorBoard + legacy checkpoint compatibility tests.

The tests verify that old scientific signatures remain authoritative and
that loggers cannot mutate model gradients/parameters or infer old step logs.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import torch

from graph_attention.training import m43_logging_resume
from graph_attention.training.m43_logging_resume import (
    ORIGINAL_M43_B_REVISION,
    changed_files_from_original,
    reconcile_resume_signature,
)
from graph_attention.training.m43_tensorboard import (
    M43TensorBoardMonitor,
    completed_optimizer_steps,
    gradient_global_l2,
    log_completed_epoch,
    log_completed_optimizer_step,
)


def _sig(revision: str, *, seed: int = 42) -> dict:
    content = {
        "mode": "train",
        "revision": revision,
        "seed": seed,
        "train_ids": ["Re94/Ma0.2:1", "Re54/Ma0.2:1"],
        "references": {"rho_ref": "synthetic_fixed"},
        "model": {"hidden_dim": 128, "num_layers": 10},
    }
    payload = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return {"sha256": hashlib.sha256(payload.encode()).hexdigest(), "content": content}


def _history_epoch(epoch: int, *, steps: int = 68) -> dict:
    return {
        "epoch": epoch,
        "train": {
            "mean": 1.1 + epoch / 10,
            "optimizer_steps": steps,
            "samples": 408,
            "physical_samples_per_second": 0.1,
            "mean_optimizer_step_seconds": 75.,
            "by_case": {"Re94/Ma0.2": {"mse": 1.2}},
            "by_resolution": {"65": {"mse": 1.05}, "129": {"mse": 1.15}},
            "forward_by_resolution": {
                "65": {"mean_seconds": 0.2},
                "129": {"mean_seconds": 1.2},
            },
            "backward_by_resolution": {
                "65": {"mean_seconds": 2.6},
                "129": {"mean_seconds": 20.0},
            },
        },
        "validation": {
            "mean": 1.0 + epoch / 10,
            "samples": 66,
            "by_case": {"Re94/Ma0.2": {"mse": 1.22}},
            "by_resolution": {"65": {"mse": 1.03}, "129": {"mse": 1.11}},
        },
        "epoch_wall_seconds": 5400.0,
    }


def test_history_step_offset_and_broken_history_rejected() -> None:
    assert completed_optimizer_steps([]) == 0
    assert completed_optimizer_steps([_history_epoch(0)]) == 68
    assert completed_optimizer_steps(
        [_history_epoch(0), _history_epoch(1, steps=69)]
    ) == 137
    with pytest.raises(ValueError, match="noncontiguous"):
        completed_optimizer_steps([_history_epoch(1)])
    with pytest.raises(ValueError, match="optimizer steps"):
        completed_optimizer_steps([_history_epoch(0, steps=0)])


def test_same_run_signature_never_needs_revision_migration(tmp_path: Path) -> None:
    signed = _sig(ORIGINAL_M43_B_REVISION)
    actual, audit = reconcile_resume_signature(
        signed, copy.deepcopy(signed),
        allow_logging_only_revision=False, repo_root=tmp_path,
    )
    assert actual == signed
    assert audit is None


def test_legacy_resume_is_explicit_and_keeps_original_checkpoint_sha(
    monkeypatch, tmp_path: Path,
) -> None:
    saved = _sig(ORIGINAL_M43_B_REVISION)
    proposed = _sig("1111111111111111111111111111111111111111")
    with pytest.raises(RuntimeError, match="opt in"):
        reconcile_resume_signature(
            saved, proposed, allow_logging_only_revision=False,
            repo_root=tmp_path,
        )
    monkeypatch.setattr(
        m43_logging_resume, "changed_files_from_original",
        lambda *args: [
            "scripts/train_m43_b_multires.py",
            "src/graph_attention/training/m43_tensorboard.py",
        ],
    )
    reused, audit = reconcile_resume_signature(
        saved, proposed, allow_logging_only_revision=True, repo_root=tmp_path,
    )
    assert reused == saved
    assert reused["sha256"] == saved["sha256"]
    assert reused["content"]["revision"] == ORIGINAL_M43_B_REVISION
    assert audit["status"] == "explicit_logging_only_revision_migration"
    assert audit["scientific_manifest_except_revision_unchanged"] is True


@pytest.mark.parametrize("tamper", ["seed", "references", "old_sha", "old_revision"])
def test_logging_migration_rejects_scientific_drift(
    monkeypatch, tmp_path: Path, tamper: str,
) -> None:
    saved = _sig(ORIGINAL_M43_B_REVISION)
    proposed = _sig("2222222222222222222222222222222222222222")
    if tamper == "seed":
        proposed = _sig(proposed["content"]["revision"], seed=13)
    if tamper == "references":
        proposed["content"]["references"]["rho_ref"] = "changed"
        proposed = _sig(proposed["content"]["revision"]) | {
            "content": proposed["content"],
        }
    if tamper == "old_sha":
        saved["sha256"] = "broken"
    if tamper == "old_revision":
        saved = _sig("3333333333333333333333333333333333333333")
    monkeypatch.setattr(
        m43_logging_resume, "changed_files_from_original",
        lambda *args: ["scripts/train_m43_b_multires.py"],
    )
    with pytest.raises(RuntimeError):
        reconcile_resume_signature(
            saved, proposed, allow_logging_only_revision=True, repo_root=tmp_path,
        )


def test_changed_paths_guard_rejects_numerical_source_modification(
    monkeypatch, tmp_path: Path,
) -> None:
    def git(_repo, *arguments):
        if arguments[0] == "merge-base":
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return subprocess.CompletedProcess(
            arguments, 0,
            "scripts/train_m43_b_multires.py\n"
            "src/graph_attention/models/dinat_dit.py\n", "",
        )

    monkeypatch.setattr(m43_logging_resume, "_run_git", git)
    with pytest.raises(RuntimeError, match="non-logging source paths"):
        changed_files_from_original(tmp_path, "1111111111111111111111111111111111111111")


def test_gradient_norm_logging_is_read_only() -> None:
    model = torch.nn.Linear(3, 2)
    loss = model(torch.ones(4, 3)).square().sum()
    loss.backward()
    params = {key: p.detach().clone() for key, p in model.named_parameters()}
    grads = {key: p.grad.detach().clone() for key, p in model.named_parameters()}
    norm = gradient_global_l2(model)
    assert norm > 0
    for key, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), params[key], rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, grads[key], rtol=0, atol=0)


class CaptureWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), step))


def test_step_and_epoch_tags_use_absolute_optimizer_step() -> None:
    captured = CaptureWriter()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = model(torch.ones(3, 2)).square().mean()
    loss.backward()
    before = [param.grad.clone() for param in model.parameters()]
    log_completed_optimizer_step(
        captured,
        epoch=1, global_step=69,
        step_mean_velocity_mse=1.25,
        running_epoch_velocity_mse=1.25,
        samples_per_step=6,
        elapsed_seconds=73.,
        per_case_loss={"Re94/Ma0.2": 1.3},
        per_resolution_loss={"65": 1.1, "129": 1.4},
        forward_per_resolution_seconds={"65": 0.4, "129": 3.6},
        backward_per_resolution_seconds={"65": 8.0, "129": 61.0},
        optimizer=optimizer, model=model, device=torch.device("cpu"),
    )
    for before_grad, param in zip(before, model.parameters()):
        torch.testing.assert_close(param.grad, before_grad, rtol=0, atol=0)
    log_completed_epoch(captured, _history_epoch(0), global_step=68)
    assert ("loss/train_step_velocity_mse", 1.25, 69) in captured.scalars
    assert ("loss/train_epoch_velocity_mse", 1.1, 68) in captured.scalars
    assert ("loss/validation_epoch_velocity_mse", 1.0, 68) in captured.scalars
    assert any("optimizer/gradient_global_l2" == tag
               and value > 0 and step == 69 for tag, value, step in captured.scalars)
    assert all(step in {68, 69} for _, _, step in captured.scalars)


def test_resume_event_files_reconstruct_only_historic_epoch_summaries(tmp_path: Path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    old_history = [_history_epoch(0)]
    with M43TensorBoardMonitor(tmp_path, resume=True, history=old_history) as monitor:
        assert monitor.global_step_offset == 68
        monitor.writer.add_scalar("loss/train_step_velocity_mse", 0.8, 69)
        monitor.log_epoch(_history_epoch(1), global_step=136)
    event_dir = tmp_path / "tensorboard"
    assert list(event_dir.glob("events.out.tfevents.*"))
    accumulated = EventAccumulator(str(event_dir)).Reload()
    assert accumulated.Scalars("loss/train_epoch_velocity_mse")[0].step == 68
    assert accumulated.Scalars("loss/train_epoch_velocity_mse")[-1].step == 136
    assert [x.step for x in accumulated.Scalars("loss/train_step_velocity_mse")] == [69]
