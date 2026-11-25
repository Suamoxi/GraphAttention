"""CPU-only M47.8A Fourier response tests; no checkpoint access."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from scripts.m47_8_response_math import mode_leakage, response_geometry, transfer_column
from scripts.m47_8_response_audit import _mode_arg, _verify_mode, _periodic_carriers


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


def test_actual_task_nondimensional_coords_produce_pure_fourier_carriers():
    # Original periodic mesh has 9^3 nodes, while FFT uses unique 8^3.
    # Task coords are (physical mesh coords / L_ref), with arbitrary L_ref
    # and box extents; node ordering can be non-C-order.
    n = 8
    mesh = np.indices((n + 1, n + 1, n + 1)).reshape(3, -1).T
    lengths_in_reference_units = np.array([0.23, 1.17, 3.79])
    scaled = mesh * (lengths_in_reference_units / n)
    rng = np.random.default_rng(47)
    permutation = rng.permutation(scaled.shape[0])
    model_coords = torch.tensor(scaled[permutation], dtype=torch.float32)
    grid = SimpleNamespace(
        shape=(n, n, n),
        source_shape=(n + 1, n + 1, n + 1),
        spacing=(1.0e-6, 2.0e-6, 3.0e-6),
        periodic_endpoint_mode="drop_max",
        linear_indices=permutation,
    )
    mode = (2, 1, 3)
    carriers, maximum_leak = _periodic_carriers(model_coords, mode, grid)
    assert maximum_leak < 1e-10
    assert len(carriers) == 2
    for carrier in carriers:
        assert carrier.shape == ((n + 1) ** 3,)

    # Guard against the v1 bug: physical rad/m frequency multiplied by
    # already nondimensional coordinates aliases to other FFT modes.
    physical_freq = np.array(mode) * 2 * np.pi / (n * np.array(grid.spacing))
    bad_phase = (
        model_coords - model_coords.amin(dim=0)
    ) @ model_coords.new_tensor(physical_freq)
    from scripts.m47_6_spatial_metrics import scatter_unique
    bad = np.zeros((*grid.shape, 3))
    bad[..., 0] = scatter_unique(torch.cos(bad_phase).numpy(), grid)
    assert mode_leakage(bad, mode) > 0.1


def test_carrier_rejects_wrong_periodic_grid():
    grid = SimpleNamespace(
        shape=(8, 8, 8), source_shape=(8, 8, 8),
        periodic_endpoint_mode="drop_max",
    )
    with pytest.raises(ValueError, match="endpoint-inclusive"):
        _periodic_carriers(torch.zeros((512, 3)), (1, 1, 0), grid)
