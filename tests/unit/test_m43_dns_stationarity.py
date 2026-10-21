"""Synthetic scientific regression gates for M43 DNS-only audit.

These tests do not read production DNS or require a GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from graph_attention.evaluation.dns_matching import matched_regime_comparison
from graph_attention.evaluation.dns_residual import (
    coarse_descriptors,
    conservative_residual,
    lift_blocks,
    radial_velocity_spectrum,
    restrict_blocks,
    spatial_residual_statistics,
    spectral_split,
    stationarity_candidates,
)


def _physical_state(side: int = 16) -> np.ndarray:
    rng = np.random.default_rng(18)
    xyz = rng.normal(size=(side, side, side, 3)) * 0.02
    rho = 1.0 + rng.uniform(0, 0.1, size=(side, side, side))
    energy = 4.0 * np.ones_like(rho)
    return np.concatenate((rho[..., None], (xyz * rho[..., None]),
                           energy[..., None]), axis=-1)


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_discrete_conservative_right_inverse_and_null_space(factor: int) -> None:
    state = _physical_state()
    coarse, residual = conservative_residual(state, factor)
    np.testing.assert_allclose(restrict_blocks(residual, factor), 0.0, atol=2.e-14)
    np.testing.assert_allclose(
        restrict_blocks(lift_blocks(coarse, factor), factor), coarse, atol=2.e-14
    )


def test_filter_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="factor"):
        restrict_blocks(_physical_state(), 3)


def test_spectral_orthogonality_and_energy_conservation() -> None:
    rng = np.random.default_rng(23)
    vel = rng.normal(size=(16, 16, 16, 3))
    low, high, evidence = spectral_split(vel, (0.1,) * 3, 4)
    np.testing.assert_allclose(low + high, vel, atol=2.e-13)
    assert evidence["parseval_relative_error"] < 2.e-13
    dot = np.mean(np.sum(low * high, axis=-1))
    assert abs(dot) < 2.e-13


def test_velocity_spectrum_integrates_to_kinetic_energy() -> None:
    rng = np.random.default_rng(24)
    vel = rng.normal(size=(16, 16, 16, 3))
    k, E = radial_velocity_spectrum(vel, (0.5,) * 3)
    dk = float(k[1] - k[0])
    desired = 0.5 * float(np.sum(np.var(vel, axis=(0, 1, 2))))
    np.testing.assert_allclose(np.sum(E) * dk, desired, rtol=3.e-13)


def test_uniform_fine_flow_has_zero_residual_increments() -> None:
    q = np.zeros((16, 16, 16, 5))
    q[..., 0] = 1.4
    q[..., 1:4] = [1.0, -0.1, 0.02]
    q[..., 4] = 3.2
    features, summary, residual = coarse_descriptors(q, 4, (0.1,) * 3)
    assert all(val == 0 for val in (
        summary["unresolved_velocity_k"],
        summary["within_s2"],
        summary["cross_s2"],
    ))
    assert np.all(residual == 0)
    assert features["log_strain"].shape == (4, 4, 4)


def test_periodic_cross_cell_increments_are_nonnegative() -> None:
    q = _physical_state()
    features, summary, residual = coarse_descriptors(q, 4, (0.1,) * 3)
    assert residual.shape == (16, 16, 16, 3)
    assert np.isfinite(summary["coarse_null_max_abs"])
    assert summary["coarse_null_max_abs"] < 1.e-13
    assert np.all(features["within_s2"] >= 0)
    assert np.all(features["cross_s2"] >= 0)
    assert np.any(features["cross_s2"] > 0)
    lag = spatial_residual_statistics(residual, (0.1,) * 3, 3)
    assert np.isfinite(lag["s2_ll"]).all()
    assert np.all(lag["s2_ll"] >= 0)


def test_stationarity_heuristic_is_not_a_declared_cutoff() -> None:
    early = np.array([[4., 5.]] * 12)
    late = np.array([[1.01, 1.02], [0.99, 0.98]] * 30)
    candidate = stationarity_candidates(np.vstack([early, late]), window=8)
    assert candidate.status == "candidate_only"
    assert candidate.earliest_index is not None
    assert candidate.latest_index is not None


def test_stationarity_heuristic_rejects_short_timeseries() -> None:
    assert stationarity_candidates(np.ones((12, 3)), window=8).status == "inconclusive"


def _regime(
    seed: int, lo: float, hi: float, offset: float = 0.0, snapshots: int = 12
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = 6000
    x = rng.uniform(lo, hi, count)
    z = rng.uniform(0, 1, count)
    # Piecewise-constant conditional target in broad bins.
    target = 2 + (np.floor(x * 4) * 0.5) + (np.floor(z * 4) * 0.25) + offset
    return {
        "snapshot_index": np.repeat(np.arange(snapshots), count // snapshots),
        "log_coarse_energy": x,
        "log_strain": z,
        "unresolved_k": target,
        "within_s2": target * 0.5,
        "cross_s2": target * 0.3,
    }


def test_matching_reduces_composition_bias_without_erasing_conditional_effect() -> None:
    transient = _regime(1, 0.1, 0.8)
    stationary = _regime(2, 0.3, 1.0)
    result = matched_regime_comparison(
        transient, stationary, min_rows_per_bin=30, bootstrap_repeats=40
    )
    assert result["status"] == "ok"
    before = (result["stationary"]["unconditional"]["unresolved_k"]
              - result["transient"]["unconditional"]["unresolved_k"])
    after = result["matched_difference_stationary_minus_transient"]["unresolved_k"]
    assert abs(after) < abs(before) + 1.e-6
    assert abs(after) < 0.3
    assert result["confidence_intervals_95_snapshot_bootstrap"]["unresolved_k"] is not None


def test_matching_identifies_true_conditional_difference() -> None:
    a = _regime(3, 0.1, 1)
    b = _regime(4, 0.1, 1, offset=1.0)
    result = matched_regime_comparison(a, b, bootstrap_repeats=40)
    assert result["status"] == "ok"
    difference = result["matched_difference_stationary_minus_transient"]["unresolved_k"]
    assert 0.8 < difference < 1.2


def test_no_common_support_returns_inconclusive_not_extrapolated() -> None:
    a = _regime(5, 0.1, 0.3)
    b = _regime(6, 4., 5.)
    result = matched_regime_comparison(a, b)
    assert result["status"] == "inconclusive_no_common_support"


def test_insufficient_independent_snapshots_returns_inconclusive() -> None:
    a = _regime(7, 0.1, 1, snapshots=12)
    b = _regime(8, 0.1, 1, snapshots=12)
    b["snapshot_index"][:] = 0
    result = matched_regime_comparison(a, b, min_snapshots_per_bin=2)
    assert result["status"] == "inconclusive_no_common_support"
