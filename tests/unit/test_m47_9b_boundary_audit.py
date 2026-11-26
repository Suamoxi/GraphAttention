"""Integration preflight for the read-only archived boundary-audit output."""
from types import SimpleNamespace

import numpy as np

from scripts.m47_9b_periodic_boundary_audit import _csv_rows, _plot_summaries, _population


def _fixture():
    n = 16
    xyz = np.indices((n + 1,) * 3)
    x, y, z = 2 * np.pi * xyz / n
    q = np.empty((n + 1, n + 1, n + 1, 5), dtype=np.float64)
    q[..., 0] = 1
    q[..., 1] = np.sin(y) + 0.3 * np.sin(x)
    q[..., 2] = np.sin(z) + 0.2 * np.cos(x)
    q[..., 3] = np.sin(x) + 0.1 * np.cos(z)
    q[..., 4] = 10
    grid = SimpleNamespace(
        source_shape=(n + 1,) * 3, shape=(n,) * 3,
        spacing=(1., 1., 1.), periodic_endpoint_mode="drop_max",
        linear_indices=np.arange((n + 1)**3),
    )
    return q.reshape(-1, 5), grid


def test_population_archive_pipeline_and_outputs(tmp_path):
    field, grid = _fixture()
    population = _population(
        field[None], ("field0",),
        ("rho.value", "rhou.x", "rhov.y", "rhow.z", "rhoE.value"),
        grid, 17.36, 4,
    )
    assert population["valid_count"] == 1
    assert population["invalid_count"] == 0
    summary = population["summary"]
    assert summary["velocity_faces_by_axis"]["x"]["stored_duplicate_endpoint_rms"]["mean"] < 1e-12
    assert len(summary["divergence_axis_profiles"]["x"]["divergence_enrichment_over_uniform"]["mean"]) == 16
    assert sum(v["voxel_fraction"]["mean"] for v in summary["boundary_depth_shells"].values()) == 1
    populations = {"reference": population, "M37": population, "M38": population}
    _csv_rows(populations, tmp_path, 4)
    _plot_summaries(populations, tmp_path / "plots", 4)
    for name in ("per_field.csv", "per_axis.csv", "per_shell.csv"):
        assert (tmp_path / name).stat().st_size > 0
    for name in ("face_mismatch.png", "divergence_by_depth.png", "divergence_slice_profiles.png"):
        assert (tmp_path / "plots" / name).stat().st_size > 0
