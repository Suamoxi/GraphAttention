"""CPU-only regression tests for M47.5 regime and periodicity diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from scripts.m47_5_regime_diagnostics import (
    ANCHOR_CASE, _paired_change, cross_domain_edge_audit,
    intervention_plan, unique_grid_wrap_ratio,
)
from scripts.m47_generation_metrics import CHANNELS


def _coordinates(side: int = 5) -> np.ndarray:
    axis = np.linspace(0, 1, side)
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1,
    ).reshape(-1, 3)


def _state_with_linear_rhou(coords: np.ndarray) -> np.ndarray:
    state = np.zeros((len(coords), 5), dtype=np.float64)
    state[:, 0] = 1.0
    state[:, 1] = coords[:, 0]
    state[:, 4] = 10.0
    return state


def test_unique_fft_wrap_excludes_max_endpoint_and_uses_neighbour_jump():
    coords = _coordinates(5)
    grid = infer_cartesian_grid_3d(
        coords, periodic_endpoint_mode="drop_max",
    )
    state = _state_with_linear_rhou(coords)
    ratios = unique_grid_wrap_ratio(state, grid)
    # Unique x positions 0,1/4,1/2,3/4: jump 3/4 vs 1/4.
    assert ratios["rhou"] == pytest.approx(3.0)
    assert ratios["rho"] is None
    assert ratios["rhoE"] is None
    assert ratios["rhov"] is None

    # Changes on a discarded maximum-coordinate plane cannot affect FFT jumps.
    changed = state.copy()
    changed[coords[:, 0] == 1, 1] = -100.0
    assert unique_grid_wrap_ratio(changed, grid) == ratios


def test_cross_domain_edge_audit_does_not_assume_cartesian_neighbours():
    coords = _coordinates(5)
    crossing = np.array([[0, 1], [100, 2]], dtype=np.int64)
    # flat source index (4,0,0)=100; 0->100 crosses x extent.
    report = cross_domain_edge_audit(
        coords, {"one_hop": crossing},
    )
    assert report["one_hop"]["directed_edges"] == 2
    assert report["one_hop"]["full_span_crossing_counts_xyz"] == [1, 0, 0]
    with pytest.raises(ValueError, match="missing nodes"):
        cross_domain_edge_audit(coords, {
            "one_hop": np.array([[0], [len(coords)]], dtype=np.int64)
        })


def test_interventions_change_only_selected_physical_descriptor():
    references = {
        "Re54/Ma0.05": {"Ma_condition": 0.05, "log_Re_condition": 4.01},
        ANCHOR_CASE: {"Ma_condition": 0.2, "log_Re_condition": 4.02},
        "Re54/Ma0.3": {"Ma_condition": 0.3, "log_Re_condition": 4.03},
        "Re94/Ma0.2": {"Ma_condition": 0.2, "log_Re_condition": 4.54},
    }
    assert intervention_plan("none", {key: {} for key in references}) == []
    mach = intervention_plan(
        "mach", {key: {"Ma_condition": row["Ma_condition"]}
                 for key, row in references.items()},
    )
    assert [item["id"] for item in mach] == ["mach_low", "mach_high"]
    assert mach[0]["conditioning"] == {"Ma_condition": 0.05}
    combined = intervention_plan("mach_re", references)
    assert [item["id"] for item in combined] == [
        "mach_low", "mach_high", "re_high_fixed_ma",
    ]
    assert combined[0]["conditioning"] == {
        "Ma_condition": 0.05, "log_Re_condition": 4.02,
    }
    assert combined[-1]["conditioning"] == {
        "Ma_condition": 0.2, "log_Re_condition": 4.54,
    }


def test_paired_state_difference_is_zero_for_identical_realizations():
    coords = _coordinates(5)
    grid = infer_cartesian_grid_3d(
        coords, periodic_endpoint_mode="drop_max",
    )
    state = _state_with_linear_rhou(coords)
    diffs = _paired_change(
        state, state.copy(), grid, {ch: 1.0 for ch in CHANNELS},
    )
    assert all(value == 0.0 for value in diffs.values())
    changed = state.copy()
    changed[:, 1] += 0.5
    diffs = _paired_change(
        changed, state, grid, {ch: 0.5 for ch in CHANNELS},
    )
    assert diffs["rhou"] == pytest.approx(1.0)
    assert diffs["rho"] == pytest.approx(0.0)
