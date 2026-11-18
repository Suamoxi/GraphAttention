"""M47.4 CPU-only tests: immutable source gates and fixed-epoch contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import m47_4_stage
from scripts.m47_4_evaluate import validate_checkpoint_metadata


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n")


def _original_fixture(tmp_path: Path, monkeypatch) -> Path:
    folder = tmp_path / "source"
    folder.mkdir()
    train = [f"t{i}" for i in range(240)]
    val = [f"v{i}" for i in range(30)]
    test = [f"q{i}" for i in range(42)]
    excluded = [f"x{i}" for i in range(48)]
    content = {
        "mode": "train", "revision": m47_4_stage.ORIGINAL_REVISION,
        "seed": 42, "temporal_embargo": 2, "accumulation": 6,
        "m47_physical_conditions": {"condition_mode": "none"},
        "train_ids": train, "validation_ids": val, "test_ids": test,
    }
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(serialized.encode()).hexdigest()
    monkeypatch.setitem(m47_4_stage.ORIGINAL_SIGNATURES, "none", sha)
    _write(folder / "run_signature.json", {"sha256": sha, "content": content})
    _write(folder / "split.json", {
        "train_ids": train, "validation_ids": val,
        "test_ids": test, "excluded_ids": excluded,
    })
    _write(folder / "summary.json", {
        "status": "train_complete", "conditioning_mode": "none",
        "signature_sha256": sha, "history": [{"epoch": 0}, {"epoch": 1}],
        "best_epoch": 1,
        "m47_training_hyperparameters": {"learning_rate": 1e-4},
    })
    _write(folder / "history.json", {"epochs": [{"epoch": 0}, {"epoch": 1}]})
    for name in ("last.pt", "best.pt", "standardizers.pt"):
        (folder / name).write_bytes(name.encode())
    return folder


def _advance(workspace: Path, epochs: int) -> None:
    folder = workspace / "train_s42"
    summary = json.loads((folder / "summary.json").read_text())
    summary["history"] = [{"epoch": i} for i in range(epochs)]
    summary["best_epoch"] = epochs - 1
    _write(folder / "summary.json", summary)
    _write(folder / "history.json", {"epochs": summary["history"]})
    (folder / "last.pt").write_bytes(f"after{epochs}".encode())
    (folder / "best.pt").write_bytes(f"after{epochs}".encode())


def test_m47_4_stage_preserves_original_and_refuses_reuse(tmp_path, monkeypatch):
    source = _original_fixture(tmp_path, monkeypatch)
    workspace = tmp_path / "out" / "none"
    original = m47_4_stage.source_record(source, "none")
    m47_4_stage.prepare(source, workspace, "none")
    assert m47_4_stage.source_record(source, "none") == original
    assert (workspace / "train_s42" / "last.pt").read_bytes() == b"last.pt"
    with pytest.raises(FileExistsError):
        m47_4_stage.prepare(source, workspace, "none")
    with pytest.raises(ValueError):
        m47_4_stage.prepare(source, source / "nested", "none")


def test_m47_4_source_revision_and_history_are_strict(tmp_path, monkeypatch):
    source = _original_fixture(tmp_path, monkeypatch)
    signature = json.loads((source / "run_signature.json").read_text())
    signature["content"]["revision"] = "unverified"
    _write(source / "run_signature.json", signature)
    with pytest.raises(ValueError, match="source"):
        m47_4_stage.source_record(source, "none")


def test_m47_4_frozen_stage5_and_stage10_gates(tmp_path, monkeypatch):
    source = _original_fixture(tmp_path, monkeypatch)
    workspace = tmp_path / "independent" / "none"
    m47_4_stage.prepare(source, workspace, "none")
    with pytest.raises((ValueError, FileNotFoundError)):
        m47_4_stage.gate_stage10(source, workspace, "none")
    _advance(workspace, 5)
    m47_4_stage.freeze(source, workspace, "none", 5)
    m47_4_stage.gate_stage10(source, workspace, "none")
    _advance(workspace, 10)
    m47_4_stage.freeze(source, workspace, "none", 10)
    assert len(json.loads((workspace / "epoch05" / "summary.json").read_text())["history"]) == 5
    assert len(json.loads((workspace / "epoch10" / "summary.json").read_text())["history"]) == 10
    with pytest.raises(FileExistsError):
        m47_4_stage.freeze(source, workspace, "none", 10)


def test_m47_4_gate_rejects_changed_stage5_checkpoint(tmp_path, monkeypatch):
    source = _original_fixture(tmp_path, monkeypatch)
    workspace = tmp_path / "out" / "none"
    m47_4_stage.prepare(source, workspace, "none")
    _advance(workspace, 5)
    m47_4_stage.freeze(source, workspace, "none", 5)
    (workspace / "train_s42" / "last.pt").write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="diverged"):
        m47_4_stage.gate_stage10(source, workspace, "none")


def test_m47_4_checkpoint_metadata_requires_full_optimizer_rng_and_stage():
    valid = {
        "epoch": 4, "history": [{"epoch": i} for i in range(5)],
        "signature_sha256": "signature",
        "model_state_dict": {}, "optimizer_state_dict": {},
        "training_generator_state": object(),
    }
    validate_checkpoint_metadata(valid, signature_sha="signature", epochs=5)
    with pytest.raises(ValueError):
        validate_checkpoint_metadata(valid, signature_sha="signature", epochs=10)
    with pytest.raises(ValueError):
        validate_checkpoint_metadata(valid, signature_sha="wrong", epochs=5)
    with pytest.raises(ValueError, match="optimizer_state_dict"):
        validate_checkpoint_metadata(
            {k: v for k, v in valid.items() if k != "optimizer_state_dict"},
            signature_sha="signature", epochs=5,
        )


def _fake_evaluation_results():
    from scripts.m47_4_compare import CASES, CHANNELS, MODES, STAGES, TIMES
    rows = {}
    for mode in MODES:
        for epochs in STAGES:
            per_case = {
                case: {
                    str(t): {"samples": 5, "channels": {
                        ch: {"mse": 0.5} for ch in CHANNELS
                    }} for t in TIMES
                } for case in CASES
            }
            physical = {
                "Re54/Ma0.2": {
                    "generated_count": 4, "reference_count": 5,
                    "generated_sampling_keys": ["k0", "k1", "k2", "k3"],
                    "reference_sample_ids": ["v0", "v1", "v2", "v3", "v4"],
                    "unique_grid_shape": [64, 64, 64],
                    "source_grid_shape": [65, 65, 65],
                }
            }
            rows[(mode, epochs)] = {
                "status": "complete", "mode": mode,
                "epochs_completed": epochs,
                "training_signature_sha256": m47_4_stage.ORIGINAL_SIGNATURES[mode],
                "validation_per_case": 5, "test_snapshot_state_used": False,
                "flow_times": list(TIMES), "shared_sampling_seed": 20261116,
                "physical_generation_steps": 50,
                "physical_generation_samples_per_case": 4,
                "physical_generation_cases": ["Re54/Ma0.2"],
                "physical_generation": physical,
                "validation_velocity_mse_at_fixed_times": per_case,
            }
    return rows


def test_m47_4_six_run_comparator_requires_matched_latents_and_splits():
    from scripts.m47_4_compare import validate_six_summaries
    rows = _fake_evaluation_results()
    validate_six_summaries(rows)
    rows[("mach_re", 10)]["physical_generation"]["Re54/Ma0.2"][
        "generated_sampling_keys"][-1] = "unmatched"
    with pytest.raises(ValueError, match="latent keys"):
        validate_six_summaries(rows)


def test_m47_4_six_run_comparator_rejects_partial_or_wrong_signature():
    from scripts.m47_4_compare import validate_six_summaries
    rows = _fake_evaluation_results()
    rows.pop(("none", 5))
    with pytest.raises(ValueError, match="exactly three arms"):
        validate_six_summaries(rows)
    rows = _fake_evaluation_results()
    rows[("mach", 5)]["training_signature_sha256"] = "wrong"
    with pytest.raises(ValueError, match="inconsistent scientific contract"):
        validate_six_summaries(rows)
