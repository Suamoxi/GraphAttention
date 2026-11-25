"""CPU-only M47.8B mathematical and data-validity regression tests."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.m47_8b_trajectory_math import (
    compare_response, exact_steps, field_summary, paired_state_metrics,
)


def test_exact_heun_times_and_invalid_times():
    assert exact_steps((0.25, 0.5, 0.75), 100) == (25, 50, 75)
    with pytest.raises(ValueError, match="not on"):
        exact_steps((0.25,), 50)
    with pytest.raises(ValueError, match="unique"):
        exact_steps((0.5, 0.5), 100)
    with pytest.raises(ValueError, match="inside"):
        exact_steps((1.0,), 100)


def test_standardized_matched_state_distance():
    teacher = torch.ones((7, 5), dtype=torch.float32)
    gen = torch.ones((7, 5), dtype=torch.float32) * 2
    result = paired_state_metrics(teacher, gen)
    assert result["teacher_rms_standardized"] == pytest.approx(1)
    assert result["generated_rms_standardized"] == pytest.approx(2)
    assert result["paired_rmse_standardized"] == pytest.approx(1)
    assert result["paired_rmse_over_teacher_rms"] == pytest.approx(1)
    assert result["paired_mse_by_channel"]["rhou"] == pytest.approx(1)


def _response(j, *, mode=(2, 2, 0), t=0.5):
    j = np.asarray(j)
    return {
        "mode": list(mode), "time": t, "relative_amplitude": 0.04,
        "matrix_real": j.real.tolist(), "matrix_imag": j.imag.tolist(),
        "gain_long_real": 0.3, "gain_long_imag": 0.0,
        "gain_trans_real": 0.8, "gain_trans_imag": 0.0,
        "response_leakage_fraction_mean": 0.03,
    }


def test_response_pair_comparison_and_provenance_guard():
    teacher = _response(np.eye(3))
    same = compare_response(teacher, _response(np.eye(3)))
    assert same["matrix_delta_relative"] == pytest.approx(0)
    assert same["matrix_real_cosine_similarity"] == pytest.approx(1)
    other = _response(np.array([[1., 0.4, 0.], [0.2, 1., 0.], [0., 0., 1.]]))
    paired = compare_response(teacher, other)
    assert paired["matrix_delta_relative"] > 0.1
    assert paired["matrix_real_cosine_similarity"] < 1
    with pytest.raises(ValueError, match="parameters"):
        compare_response(teacher, _response(np.eye(3), mode=(4, 3, 1)))


def test_momentum_proxy_is_distinct_from_actual_velocity_and_rejects_bad_rho():
    n = 8
    coords = np.indices((n + 1, n + 1, n + 1))
    q = np.zeros((n + 1, n + 1, n + 1, 5), dtype=np.float32)
    q[..., 0] = 1.0
    q[..., 2] = np.sin(2 * np.pi * coords[0] / n)
    q[..., 4] = 5.0
    grid = SimpleNamespace(
        shape=(n, n, n), source_shape=(n + 1, n + 1, n + 1),
        linear_indices=np.arange((n + 1) ** 3),
        periodic_endpoint_mode="drop_max",
        spacing=(1.0 / n,) * 3, k_nyquist_min=np.pi * n,
    )
    scalers = SimpleNamespace(inputs=SimpleNamespace(inverse=lambda x, _: x))
    batch = SimpleNamespace(input_channels=("rho", "rhou", "rhov", "rhow", "rhoE"))
    valid = field_summary(
        torch.from_numpy(q.reshape(-1, 5)), scalers, batch,
        grid, u_ref=2.0, fixed_density=np.ones((n, n, n)),
    )
    assert valid["nonpositive_density_fraction"] == 0
    assert valid["nonpositive_internal_energy_fraction"] == 0
    assert valid["actual_velocity_if_density_positive"]["longitudinal_fraction"] < 1e-10
    q[..., 0] = -1.0
    invalid = field_summary(
        torch.from_numpy(q.reshape(-1, 5)), scalers, batch,
        grid, u_ref=2.0, fixed_density=np.ones((n, n, n)),
    )
    assert invalid["nonpositive_density_fraction"] == 1
    assert invalid["actual_velocity_if_density_positive"] is None
