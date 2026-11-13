"""Fast CPU logic gates for read-only M43 scientific continuation."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.validate_m43_adaln_continuation import (
    CASE_IDS, _continuation_comparison, _diagnostic_gate,
    _preflight, _state_comparison, _tensor_comparison, _validated_host_revision,
    _training_windows, _validation_pick,
)


def test_fixed_validation_subset_covers_each_case_without_replacement():
    ids = tuple(f"{case}:solut_hit_{i:08d}"
                for case in CASE_IDS for i in range(10))
    lookup = {sid: sid.split(":")[0] for sid in ids}
    picked = _validation_pick(ids, lookup, per_case=2)
    assert len(picked) == 12
    assert len(set(picked)) == 12
    assert all(sum(lookup[sid] == case for sid in picked) == 2
               for case in CASE_IDS)
    assert picked == _validation_pick(ids, lookup, per_case=2)


def test_invalid_validation_count_rejected():
    ids = tuple(f"{case}:0" for case in CASE_IDS)
    lookup = {sid: sid.split(":")[0] for sid in ids}
    for per_case in (0, 4, 2):
        with pytest.raises(ValueError):
            _validation_pick(ids, lookup, per_case=per_case)


def test_balanced_original_six_case_windows():
    samples = tuple(f"{case}:{step}" for step in range(3) for case in CASE_IDS)
    lookup = {sid: sid.split(":")[0] for sid in samples}
    dataset = SimpleNamespace(
        sample_ids=samples, case_by_sample_id=lookup,
    )
    windows = _training_windows(tuple(range(18)), dataset, steps=3)
    assert len(windows) == 3
    assert all(len(window) == 6 for window in windows)
    assert windows[0] == samples[:6]
    bad = list(range(18))
    bad[6], bad[7] = bad[7], bad[6]
    with pytest.raises(RuntimeError, match="six-case"):
        _training_windows(tuple(bad), dataset, steps=3)
    with pytest.raises(ValueError, match="steps"):
        _training_windows(tuple(range(18)), dataset, steps=0)


def _mini_snapshot(*, weight_delta=0., grad_delta=0., moment_delta=0.):
    return {
        "gradients": {
            "block.weight": torch.tensor([0.1 + grad_delta, 0.2]),
            "unused": None,
            "tiny.bias": torch.tensor([1.e-10]),
        },
        "post_update": {
            "weights": {
                "block.weight": torch.tensor([1. + weight_delta, 2.]),
                "tiny.bias": torch.tensor([0.]),
            },
            "optimizer": {
                "groups": [{"lr": 0.0001, "weight_decay": 0.}],
                "state": {
                    0: {
                        "step": torch.tensor(100.),
                        "exp_avg": torch.tensor([0.01 + moment_delta, 0.02]),
                        "exp_avg_sq": torch.tensor([.001, .002]),
                    }
                },
            },
        },
    }


def test_combined_tensor_metrics_and_near_zero_gradients():
    a = {"main": torch.tensor([1., 2.]), "tiny": torch.tensor([1.e-10])}
    b = {"main": torch.tensor([1.0001, 2.]), "tiny": torch.tensor([2.e-10])}
    metrics = _tensor_comparison(a, b)
    assert 0 < metrics["combined_relative_l2"] < 1.e-3
    assert metrics["worst_relative"][0]["name"] == "tiny"
    assert metrics["worst_meaningful_relative"][0]["name"] == "main"
    with pytest.raises(RuntimeError, match="presence"):
        _tensor_comparison({"x": None}, {"x": torch.ones(1)})
    with pytest.raises(FloatingPointError):
        _tensor_comparison({"x": torch.tensor([float("nan")])},
                           {"x": torch.tensor([0.])})


def test_adamw_moments_and_gradient_comparison():
    a = _mini_snapshot()
    b = _mini_snapshot(weight_delta=1.e-5,
                       grad_delta=2.e-5, moment_delta=1.e-5)
    outcome = _state_comparison(a, b)
    assert outcome["weights"]["combined_relative_l2"] > 0
    assert outcome["gradients"]["combined_relative_l2"] > 0
    assert outcome["exp_avg"]["combined_relative_l2"] > 0
    assert outcome["exp_avg_sq"]["combined_relative_l2"] == 0.
    b["post_update"]["optimizer"]["state"][0]["step"] += 1
    with pytest.raises(RuntimeError, match="counter"):
        _state_comparison(a, b)


def _step_record(loss: float, *, case_tag="same", rng="fixed"):
    cases = {
        case: {"mse": loss} for case in CASE_IDS
    }
    return {
        "step_sample_ids": [f"{case}:{case_tag}" for case in CASE_IDS],
        "generator_sha256_after_step": rng,
        "validation_ids": [f"{case}:val" for case in CASE_IDS],
        "samples_per_step": 6,
        "step_mean_velocity_mse": loss,
        "validation": {
            "samples": 6, "mean": loss, "by_case": cases,
        },
    }


def test_continuation_rejects_rng_or_sample_mismatch():
    ref = [_step_record(.2)]
    bad = [_step_record(.2, rng="different")]
    with pytest.raises(RuntimeError, match="RNG"):
        _continuation_comparison(ref, bad, [_mini_snapshot()],
                                 [_mini_snapshot()])
    bad = [_step_record(.2, case_tag="other")]
    with pytest.raises(RuntimeError, match="sample ordering"):
        _continuation_comparison(ref, bad, [_mini_snapshot()],
                                 [_mini_snapshot()])


def test_multi_step_drift_reports_and_thresholds_are_explicit():
    ref = [_step_record(.2), _step_record(.15)]
    candidate = [_step_record(.200001), _step_record(.150001)]
    states = [_mini_snapshot(), _mini_snapshot()]
    comparisons = _continuation_comparison(ref, candidate, states, states)
    assert len(comparisons) == 2
    assert comparisons[0]["validation_velocity_mse_abs"] < 2.e-6
    passed = _diagnostic_gate(comparisons, comparisons, comparisons)
    assert passed["passed"]
    assert passed["type"] == "diagnostic_provisional_not_scientific_resume_authorization"
    far = [_step_record(.4), _step_record(.4)]
    differences = _continuation_comparison(ref, far, states, states)
    rejected = _diagnostic_gate(differences, comparisons, comparisons)
    assert not rejected["passed"]


def test_preflight_rejects_scientific_output_directory(tmp_path: Path):
    run = tmp_path / "original_run"
    run.mkdir()
    args = Namespace(
        train_run_dir=run, output_dir=run / "unwanted",
        root=tmp_path / "archive", reference_dir=tmp_path / "refs",
    )
    with pytest.raises(ValueError, match="diagnostic output"):
        _preflight(args)


def test_host_verified_git_provenance_without_container_git():
    full_sha = "7fd84718b051545baa8a55d38c711a45c5cc7aaf"
    assert _validated_host_revision(full_sha) == full_sha
    assert _validated_host_revision(full_sha.upper()) == full_sha
    for bad in ("", "HEAD", "7fd8471", "0"*39, "z"*40, "../HEAD"):
        with pytest.raises(ValueError, match="host-verified Git SHA"):
            _validated_host_revision(bad)
