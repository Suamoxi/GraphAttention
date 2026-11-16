"""M47.2 periodic-grid and independent-population scientific invariant tests."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from scripts.m47_generation_metrics import (
    CHANNELS, dilatational_energy_fraction, field_diagnostics, mean_curves,
    periodic_endpoint_mismatch, population_wasserstein,
    relative_spectrum_l1, sampled_nodes, unique_node_mask, wasserstein_1d,
)
from scripts.m47_generation_compare import CASES, MODES, check_contract


def _mesh(side=5):
    axis = np.linspace(0.0, 1.0, side)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def _periodic_state(coords):
    x, y, z = coords.T
    u = np.stack((np.sin(2 * np.pi * y),
                  np.sin(2 * np.pi * z),
                  np.sin(2 * np.pi * x)), axis=-1)
    rho = np.ones(len(coords))
    return np.column_stack((rho, u, 4.0 + np.sum(u**2, axis=1) / 2))


def test_drop_max_periodic_seams_not_double_counted():
    coords = _mesh(5)
    grid = infer_cartesian_grid_3d(coords, periodic_endpoint_mode="drop_max")
    assert grid.source_shape == (5, 5, 5)
    assert grid.shape == (4, 4, 4)
    assert unique_node_mask(grid).sum() == 64
    state = _periodic_state(coords)
    first = sampled_nodes(state, grid, key="matched_sample", count=23)
    second = sampled_nodes(state, grid, key="matched_sample", count=23)
    assert first.shape == (23, 5)
    assert np.array_equal(first, second)
    seam = periodic_endpoint_mismatch(state, grid)
    assert max(abs(x) for x in seam.values() if x is not None) < 1e-12
    # A perturbed duplicate endpoint is excluded from one-point statistics,
    # but the seam diagnostic must reveal the discontinuity.
    state[-1, 1] = 100.0
    assert np.max(sampled_nodes(state, grid, key="all", count=100)[:, 1]) < 100
    assert periodic_endpoint_mismatch(state, grid)["rhou"] > 1.0


def test_wasserstein_empirical_unequal_population_sizes_and_zero_spread():
    assert wasserstein_1d([0, 1, 2], [1, 2, 3]) == pytest.approx(1.0)
    assert wasserstein_1d([0, 0], [2]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        wasserstein_1d([], [1])
    a = [np.zeros((3, 5))]
    b = [np.ones((4, 5))]
    vals = population_wasserstein(a, b)
    assert all(x["wasserstein_nondimensional"] == pytest.approx(1) for x in vals.values())
    assert all(x["wasserstein_over_ref_std"] is None for x in vals.values())


def test_helmholtz_solenoidal_and_dilatational_synthetic_modes():
    coords = _mesh(9)
    grid = infer_cartesian_grid_3d(coords, periodic_endpoint_mode="drop_max")
    x, y, z = coords.T
    sol = np.column_stack((np.ones(len(coords)), np.sin(2 * np.pi * y),
                          np.zeros(len(coords)), np.zeros(len(coords)),
                          np.full(len(coords), 10.0)))
    dil = np.column_stack((np.ones(len(coords)), np.sin(2 * np.pi * x),
                          np.zeros(len(coords)), np.zeros(len(coords)),
                          np.full(len(coords), 10.0)))
    assert dilatational_energy_fraction(sol, grid) == pytest.approx(0.0, abs=1e-12)
    assert dilatational_energy_fraction(dil, grid) == pytest.approx(1.0, abs=1e-12)


def test_field_report_keeps_negative_density_visible_and_spectra_null():
    coords = _mesh(5)
    grid = infer_cartesian_grid_3d(coords, periodic_endpoint_mode="drop_max")
    state = _periodic_state(coords)
    report = field_diagnostics(state, grid, tuple(CHANNELS), k_bins=4,
                               max_lag=1, velocity_scale_m_per_s=10.0)
    assert report["density_positive_everywhere"]
    assert report["thermodynamically_admissible_everywhere"]
    assert report["velocity_energy_spectrum"] is not None
    assert report["dilatational_energy_fraction"] == pytest.approx(0.0, abs=1e-12)
    state[0, 0] = -1.0
    report = field_diagnostics(state, grid, tuple(CHANNELS), k_bins=4,
                               max_lag=1, velocity_scale_m_per_s=10.0)
    assert report["nonpositive_density_fraction"] == pytest.approx(1 / 64)
    assert not report["density_positive_everywhere"]
    assert report["velocity_energy_spectrum"] is None
    assert report["dilatational_energy_fraction"] is None
    assert report["field_mean"]["rho"] == pytest.approx(1.0 - 2.0 / 64.0)


def test_relative_spectrum_error_and_curve_population():
    reference = {"k_rad_per_m": [2.0, 4.0], "E_m3_per_s2": [2.0, 4.0]}
    generated = {"k_rad_per_m": [2.0, 4.0], "E_m3_per_s2": [4.0, 2.0]}
    assert relative_spectrum_l1(reference, reference) == pytest.approx(0.0)
    assert relative_spectrum_l1(reference, generated) == pytest.approx(4 / 6)
    assert relative_spectrum_l1(reference, None) is None
    with pytest.raises(ValueError):
        relative_spectrum_l1(reference, {"k_rad_per_m": [2.0, 5.0], "E_m3_per_s2": [2.0, 4.0]})
    rows = [{"spectrum": reference}, {"spectrum": generated}, {"spectrum": None}]
    out = mean_curves(rows, "spectrum", "k_rad_per_m", "E_m3_per_s2")
    assert out["sample_count"] == 2
    assert out["E_m3_per_s2"] == pytest.approx([3.0, 3.0])


def _fake_results():
    base = {
        "version": "m47_independent_ensemble_v1", "status": "complete",
        "generator_code_revision": "rev", "held_out_reference_pairing": False,
        "gaussian_sampling_seed": 123, "sampling_steps": 50,
        "heun_model_evaluations_per_generated_field": 100,
        "samples_per_case": 8, "node_subsample_per_field": 8192,
        "energy_spectral_bins": 32, "spatial_correlation_max_lag": 8,
        "periodic_fft_endpoint_mode": "drop_max", "cases": {},
    }
    case = {
        "case": "", "reference_sample_ids": [str(i) for i in range(7)],
        "generated_sampling_keys": [str(i) for i in range(8)],
        "source_grid_shape": [65, 65, 65], "unique_grid_shape": [64, 64, 64],
        "physical_grid_spacing_m": [0.01] * 3, "reference_velocity_scale_m_per_s": 1,
        "reference_count": 7, "generated_count": 8,
        "reference_density_valid_count": 7,
        "reference_thermodynamically_valid_count": 7,
        "reference_mean_velocity_spectrum": None,
        "reference_mean_dilatational_energy_fraction": None,
        "reference_mean_velocity_correlation": None,
        "reference_seam_jump_over_std": dict.fromkeys(CHANNELS, 0.0),
        "generated_density_valid_count": 8,
        "generated_thermodynamically_valid_count": 8,
        "one_point_wasserstein": {c: {"wasserstein_over_ref_std": 0.1} for c in CHANNELS},
    }
    base["cases"] = {c: {**deepcopy(case), "case": c} for c in CASES}
    return {name: {**deepcopy(base), "conditioning_mode": name} for name in MODES}


def test_matched_comparison_rejects_mismatched_keys_and_references():
    results = _fake_results()
    check_contract(results)
    results["mach"]["cases"][CASES[0]]["generated_sampling_keys"][-1] = "other"
    with pytest.raises(ValueError, match="mismatched reference population"):
        check_contract(results)
    results = _fake_results()
    results["mach_re"]["sampling_steps"] = 100
    with pytest.raises(ValueError, match="mismatched ensemble setup"):
        check_contract(results)
    results = _fake_results()
    results["mach_re"]["cases"][CASES[0]]["reference_count"] = 6
    with pytest.raises(ValueError):
        check_contract(results)
