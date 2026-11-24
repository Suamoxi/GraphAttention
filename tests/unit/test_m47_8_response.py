"""CPU-only M47.8A Fourier response tests; no checkpoint access."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.m47_8_response_math import mode_leakage, response_geometry, transfer_column
from scripts.m47_8_response_audit import _mode_arg, _verify_mode


def _carriers(mode=(1, 1, 0), side=17):
    mesh = np.meshgrid(*(np.arange(side) / side for _ in range(3)), indexing="ij")
    angle = 2 * np.pi * sum(n * x for n, x in zip(mode, mesh))
    return np.cos(angle)[..., None], np.sin(angle)[..., None]


def test_complex_mode_transfer_recovers_identity_and_projector():
    mode = (1, 1, 0)
    amplitude = 0.03
    cos, sin = _carriers(mode)
    k = np.array(mode, dtype=float) / np.sqrt(2)
    longitudinal = np.outer(k, k)
    matrix = 1.4 * (np.eye(3) - longitudinal) + 0.1 * longitudinal
    actual = np.empty((3, 3), dtype=np.complex128)
    for column in range(3):
        a = amplitude * cos * matrix[:, column]
        b = amplitude * sin * matrix[:, column]
        actual[:, column] = transfer_column(a, b, mode, amplitude)
        assert mode_leakage(a, mode) < 1e-12
    np.testing.assert_allclose(actual, matrix, atol=1e-12)
    stats = response_geometry(actual, mode, (1.0, 1.0, 1.0))
    assert stats["gain_long_real"] == pytest.approx(0.1)
    assert stats["gain_trans_real"] == pytest.approx(1.4)
    assert stats["offdiagonal_relative_frobenius"] > 0.2
    assert stats["projector_fit_relative_error"] < 1e-12
    assert stats["long_trans_mixing_relative"] < 1e-12


def test_uncorrelated_component_response_has_no_spectral_gap():
    diag = response_geometry(np.eye(3), (1, 1, 1), (1, 1, 1))
    assert diag["long_trans_gap_relative"] < 1e-12
    assert diag["offdiagonal_relative_frobenius"] == 0


def test_fourier_modes_and_amplitudes_are_validated():
    zeros = np.zeros((9, 9, 9, 3))
    with pytest.raises(ValueError, match="amplitude"):
        transfer_column(zeros, zeros, (1, 0, 0), 0)
    with pytest.raises(ValueError, match="nonzero"):
        transfer_column(zeros, zeros, (0, 0, 0), 0.1)
    with pytest.raises(ValueError, match="nonzero"):
        response_geometry(np.eye(3), (0, 0, 0), (1, 1, 1))
    with pytest.raises(ValueError, match="lengths"):
        response_geometry(np.eye(3), (1, 0, 0), (0, 1, 1))
    assert _mode_arg("2,3,1") == (2, 3, 1)
    with pytest.raises(ValueError, match="Nyquist"):
        _verify_mode((32, 1, 0), (64, 64, 64))
    _verify_mode((8, 6, 5), (64, 64, 64))
