"""CPU-only exact full-field Gaussian FM positive-control tests."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.m47_8_gaussian_oracle import (
    direct_transport, exact_fm_velocity, integrate, longitudinal_fraction,
    operators, run, target_spectrum_fraction,
)


def test_target_covariance_has_correct_isotropic_eigenvalue_trace():
    sol, dil = target_spectrum_fraction(0.018731345684936694)
    assert (2 * sol + dil) / 3 == pytest.approx(1.0)
    assert dil / (2 * sol + dil) == pytest.approx(0.018731345684936694)


def test_analytic_gaussian_transport_changes_longitudinal_fraction():
    khat, active = operators(17)
    rng = np.random.default_rng(47)
    source = rng.standard_normal((17, 17, 17, 3))
    covariance = target_spectrum_fraction(0.02)
    exact = direct_transport(source, khat, covariance)
    assert longitudinal_fraction(source, khat, active) == pytest.approx(1 / 3, abs=0.035)
    assert longitudinal_fraction(exact, khat, active) == pytest.approx(0.02, abs=0.003)
    a = integrate(source, 25, khat, covariance)
    b = integrate(source, 100, khat, covariance)
    error_a = np.linalg.norm(a - exact)
    error_b = np.linalg.norm(b - exact)
    assert error_b < error_a * 0.1


def test_source_and_target_velocity_analytical_endpoints():
    khat, _ = operators(9)
    src = np.random.default_rng(21).standard_normal((9, 9, 9, 3))
    cov = target_spectrum_fraction(0.02)
    np.testing.assert_allclose(exact_fm_velocity(src, 0, khat, cov), -src, atol=1e-12)


def test_report_is_deterministic():
    r1 = run(side=9, samples=1, steps=(10,), seed=7)
    r2 = run(side=9, samples=1, steps=(10,), seed=7)
    assert r1 == r2
    assert r1["records"][0]["paired_relative_rmse"] < 0.05
