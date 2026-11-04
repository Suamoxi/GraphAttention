"""Synthetic regression tests for M43 review-gated stationarity qualification."""
from __future__ import annotations

import numpy as np
import pytest

from graph_attention.evaluation.m43_qualification import (
    QualificationConfig,
    block_bootstrap_mean,
    effective_block_length,
    qualify_case,
    snapshot_labels,
    spectral_energy_fractions,
    validate_reviewed_cutoffs,
    wasserstein_quantile_1d,
)


def _signals(spinup: int = 20, *, drifting: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    n = 100
    if drifting:
        envelope = np.linspace(0.2, 2.6, n)
    else:
        envelope = np.ones(n)
        envelope[:spinup] = np.linspace(0.3, 1.0, spinup)
    noise = rng.normal(0, 0.01, n)
    return {
        "turbulent_k": 8 * (envelope + noise),
        "enstrophy": 20 * (envelope + noise),
        "std_rho": 0.03 * (envelope + noise),
        "f4_spectral_unresolved_energy_fraction": 0.05 * (envelope + noise),
        "spectrum_low_fraction": 0.3 * (envelope + noise),
        "spectrum_mid_fraction": 0.6 * (envelope + noise),
        "spectrum_high_fraction": 0.1 * (envelope + noise),
    }


def test_spinup_not_misclassified_as_early_stationarity() -> None:
    result = qualify_case(_signals())
    assert result["candidate_stationary_start_index"] is not None
    assert result["candidate_stationary_start_index"] >= 12
    assert result["candidate_stationary_start_index"] <= 50
    assert result["reviewed"] is False
    labels = snapshot_labels(100, result)
    assert labels[0] == "transient_candidate"
    assert labels[-1] == "stationary_candidate"
    assert "stationary" not in labels


def test_strongly_drifting_trajectory_is_not_declared_stationary() -> None:
    result = qualify_case(_signals(drifting=True))
    assert result["status"].startswith("provisional") or result["status"].startswith("inconclusive")
    assert result["reviewed"] is False
    # A plausible 36-point suffix can be a false plateau. No hard claim is made.
    assert all("stationary" != name for name in snapshot_labels(100, result))


def test_parameter_sensitivity_is_reported() -> None:
    result = qualify_case(_signals())
    assert len(result["sensitivity"]) >= 6
    assert result["sensitivity_range_indices"] is not None
    assert "bootstrap_block_length_snapshots" in result["diagnostics"]["turbulent_k"]


def test_rejects_invalid_history() -> None:
    data = _signals()
    data["enstrophy"][25] = np.nan
    with pytest.raises(ValueError, match="invalid"):
        qualify_case(data)
    short = {name: a[:20] for name, a in _signals().items()}
    with pytest.raises(ValueError, match="insufficient"):
        qualify_case(short)


def test_reviewed_manifest_is_tri_state_and_inclusive() -> None:
    result = qualify_case(_signals())
    approved = {"transient_end_index": 14, "stationary_start_index": 30}
    labels = snapshot_labels(100, result, approved)
    assert labels.count("transient") == 15
    assert labels.count("transition") == 15
    assert labels.count("stationary") == 70
    assert labels[14] == "transient" and labels[15] == "transition"
    assert labels[29] == "transition" and labels[30] == "stationary"
    with pytest.raises(ValueError, match="cutoff"):
        validate_reviewed_cutoffs({"transient_end_index": 10,
                                   "stationary_start_index": 11}, 100)
    with pytest.raises(ValueError):
        validate_reviewed_cutoffs({"transient_end_index": 10,
                                   "stationary_start_index": True}, 100)


def test_stable_signal_bootstrap_has_finite_mean_interval() -> None:
    a = 2 + np.sin(np.arange(60) / 4)
    length = effective_block_length(a)
    mean, low, high = block_bootstrap_mean(a, length=length, repeats=100, seed=5)
    assert 2 <= length <= 12
    assert low < mean < high


def test_spectral_energy_fractions_are_nonnegative_and_sum_one() -> None:
    k = np.arange(1, 41, dtype=float) * 2
    e = np.exp(-k / 15)
    out = spectral_energy_fractions(k, e)
    assert all(0 < v < 1 for v in out.values())
    assert sum(out.values()) == pytest.approx(1)
    with pytest.raises(ValueError, match="invalid"):
        spectral_energy_fractions(k, np.full_like(k, -1))


def test_wasserstein_is_symmetric_and_respects_translation() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=1000)
    b = a + 3.0
    assert wasserstein_quantile_1d(a, b) == pytest.approx(3.0, abs=1e-12)
    assert wasserstein_quantile_1d(a, b) == pytest.approx(wasserstein_quantile_1d(b, a))
    with pytest.raises(ValueError):
        wasserstein_quantile_1d(a, np.array([float("nan")]))
