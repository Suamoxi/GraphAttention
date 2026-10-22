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
    # Floating-point block averages and rho division introduce roundoff:
    # the physically uniform flow must produce negligible, not bitwise-zero,
    # velocity residuals and squared increments.
    np.testing.assert_allclose(
        [summary["unresolved_velocity_k"],
         summary["within_s2"], summary["cross_s2"]],
        0.0, atol=1.e-25, rtol=0.0,
    )
    np.testing.assert_allclose(residual, 0.0, atol=1.e-14, rtol=0.0)
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


def test_synthetic_hdf5_preflight_and_resume(tmp_path) -> None:
    """Read-only 9^3 stored / 8^3 unique periodic AVBP-like integration gate."""
    import json
    from types import SimpleNamespace

    import h5py

    from scripts.audit_m43_dns_stationarity import (
        _run_case,
        preflight,
    )

    case = "Re94/Ma0.2"
    directory = tmp_path / "data" / "Re94" / "Ma0.2"
    mesh_dir = directory / "MESH"
    solut_dir = directory / "SOLUT"
    mesh_dir.mkdir(parents=True)
    solut_dir.mkdir()
    unique_side = 8
    n = unique_side + 1
    x, y, z = np.meshgrid(*[np.linspace(0., 1., n)] * 3, indexing="ij")
    mesh = mesh_dir / "mesh.mesh.h5"
    with h5py.File(mesh, "w") as h:
        group = h.create_group("Coordinates")
        for name, value in zip("xyz", (x, y, z)):
            group.create_dataset(name, data=value.reshape(-1))
        h.create_dataset("Connectivity/hex->node",
                         data=np.zeros((unique_side**3, 8), dtype=np.int32))
    for index in (1, 2):
        rho = 1. + 0.01 * np.sin(2 * np.pi * x)
        ux = 0.2 * np.sin(2 * np.pi * (x + (index - 1) * 0.05))
        uy = 0.1 * np.cos(2 * np.pi * y)
        uz = 0.05 * np.sin(2 * np.pi * z)
        with h5py.File(solut_dir / f"solut_hit_{index:08d}.h5", "w") as h:
            h.attrs["physical_time"] = float(index) * 0.01
            group = h.create_group("GaseousPhase")
            for name, value in {
                "rho": rho, "rhou": rho * ux,
                "rhov": rho * uy, "rhow": rho * uz,
                "rhoE": 3.0 + np.zeros_like(rho),
            }.items():
                group.create_dataset(name, data=value.reshape(-1))
    output = tmp_path / "results"
    summary = preflight(tmp_path / "data", output, [case])
    assert summary[0]["snapshot_count"] == 2
    assert summary[0]["physical_time_metadata_monotonic"]
    assert not summary[0]["time_units_authoritative"]
    options = SimpleNamespace(
        root=tmp_path / "data",
        output_dir=output,
        max_snapshots=2,
        factors=(2,),
        max_lag=2,
        sample_cells=12,
        seed=8,
        bootstrap=0,
    )
    _run_case(options, case, {})
    _run_case(options, case, {})  # second pass reuses finished snapshot records
    case_dir = output / "cases" / "Re94_Ma0.2"
    report = json.loads((case_dir / "summary.json").read_text())
    assert report["snapshots_completed"] == 2
    assert report["comparison_status"] == "awaiting_reviewed_cutoff"
    assert len(list((case_dir / "snapshots").glob("*.json"))) == 2
    assert (case_dir / "figures" / "time_history.pdf").is_file()
    assert (case_dir / "candidate_cutoff_for_review.json").is_file()



def test_run_params_infers_only_a_provisional_time_schedule(tmp_path) -> None:
    from pathlib import Path

    from scripts.audit_m43_dns_stationarity import (
        _run_params_save_schedule,
        _schedule_snapshot_time,
    )

    run = tmp_path / "RUN"
    run.mkdir()
    params = run / "run.params"
    params.write_text(
        "! a disabled/old setting must not replace the active one\n"
        "!save_solution.time = 9.0d-3\n"
        "simulation_end_time = 2.0D-4\n"
        "save_solution.time = 2.0d-6\n"
        "random_forcing.char_time = 1.995d-6\n"
    )
    files = tuple(Path(f"solut_hit_{i:08d}.h5") for i in range(1, 101))
    schedule = _run_params_save_schedule(tmp_path, files)
    assert schedule["status"] == "configuration_consistent_origin_unverified"
    assert not schedule["time_origin_verified"]
    assert not schedule["hdf5_time_verified"]
    assert schedule["end_time_over_save_interval"] == pytest.approx(100)
    assert _schedule_snapshot_time(schedule, 0)["nominal_time_from_zero"] == pytest.approx(2e-6)
    assert _schedule_snapshot_time(schedule, 99)["nominal_time_from_zero"] == pytest.approx(2e-4)
    assert _schedule_snapshot_time(schedule, 99)["elapsed_in_forcing_times"] == pytest.approx(
        99 * 2e-6 / 1.995e-6
    )
    params.write_text(params.read_text().replace("2.0D-4", "2.5D-4"))
    inconsistent = _run_params_save_schedule(tmp_path, files)
    assert inconsistent["status"] == "configuration_inconsistent_do_not_infer"
    assert _schedule_snapshot_time(inconsistent, 0)["nominal_time_from_zero"] is None


def test_run_params_cadence_rejects_gapped_or_restarted_file_sequences(tmp_path) -> None:
    from pathlib import Path

    from scripts.audit_m43_dns_stationarity import _run_params_save_schedule

    run = tmp_path / "RUN"
    run.mkdir()
    (run / "run.params").write_text(
        "simulation_end_time = 2d-4\n"
        "save_solution.time = 2d-6\n"
        "random_forcing.char_time = 1.995d-6\n"
    )
    files = tuple(Path(f"solut_hit_{i:08d}.h5") for i in range(2, 102))
    result = _run_params_save_schedule(tmp_path, files)
    assert result["status"] == "configuration_inconsistent_do_not_infer"
    assert not result["filename_sequence_starts_at_one_and_is_contiguous"
    ]
