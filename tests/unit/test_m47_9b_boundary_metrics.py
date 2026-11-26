"""CPU regression tests for boundary mismatch and divergence localization."""
import numpy as np
import pytest

from scripts.m47_9b_boundary_metrics import divergence_localization, face_jump_statistics


def _periodic_velocity(n=16):
    xyz = np.indices((n + 1,) * 3)
    x, y, z = 2 * np.pi * xyz / n
    return np.stack((np.sin(y) + 0.2 * np.cos(z),
                     np.sin(z) + 0.2 * np.cos(x),
                     np.sin(x) + 0.2 * np.cos(y)), axis=-1)


def test_repeated_endpoint_is_not_periodic_one_cell_jump():
    field = _periodic_velocity()
    face = face_jump_statistics(field, velocity_vector=True)
    for axis in "xyz":
        row = face[axis]
        assert row["stored_duplicate_endpoint_rms"] < 1e-12
        assert row["periodic_one_cell_wrap_rms"] > 0
        assert 0.5 < row["wrap_over_interior"] < 2.0


def test_mutating_only_stored_duplicate_changes_endpoint_not_fft_wrap():
    field = _periodic_velocity()
    baseline = face_jump_statistics(field, velocity_vector=True)
    field[-1, :-1, :-1, 0] += 10
    changed = face_jump_statistics(field, velocity_vector=True)
    assert changed["x"]["stored_duplicate_endpoint_rms"] > 1
    assert changed["x"]["periodic_one_cell_wrap_rms"] == pytest.approx(
        baseline["x"]["periodic_one_cell_wrap_rms"]
    )
    assert changed["x"]["ordinary_one_cell_interior_rms"] == pytest.approx(
        baseline["x"]["ordinary_one_cell_interior_rms"]
    )


def test_unique_seam_discontinuity_without_duplicate_endpoint_error():
    n = 16
    field = np.zeros((n + 1, n + 1, n + 1, 3))
    field[0, :-1, :-1, 0] += 7
    field[-1, :-1, :-1, 0] += 7  # stored duplicate remains consistent
    face = face_jump_statistics(field, velocity_vector=True)
    assert face["x"]["stored_duplicate_endpoint_rms"] == 0
    assert face["x"]["wrap_over_interior"] > 2


def test_divergence_core_matches_existing_central_stencil_formula():
    from scripts.m47_9b_boundary_metrics import divergence_localization
    n = 16
    rng = np.random.default_rng(7)
    u = rng.normal(size=(n, n, n, 3))
    h = (0.5, 0.75, 1.25)
    r = divergence_localization(u, h, core_trim=4)
    g = [[(np.roll(u[..., c], -1, axis=a) - np.roll(u[..., c], 1, axis=a)) / (2 * h[a])
          for a in range(3)] for c in range(3)]
    div = g[0][0] + g[1][1] + g[2][2]
    curl = (g[1][2]-g[2][1], g[2][0]-g[0][2], g[0][1]-g[1][0])
    vort2 = sum(v**2 for v in curl)
    assert r["full_divergence_to_vorticity_rms"] == pytest.approx(
        np.sqrt(np.mean(div**2) / np.mean(vort2)))
    s = (slice(4, -4),) * 3
    assert r["core_divergence_to_vorticity_rms"] == pytest.approx(
        np.sqrt(np.mean(div[s]**2) / np.mean(vort2[s])))
    assert sum(x["point_count"] for x in r["shells"].values()) == n**3
    assert sum(x["divergence_energy_fraction"] for x in r["shells"].values()) == pytest.approx(1)
    for axis in "xyz":
        assert len(r["axis_profiles"][axis]["divergence_enrichment_over_uniform"]) == n


def test_boundary_artifact_is_localized_by_disjoint_depth_shells():
    n = 16
    u = np.zeros((n, n, n, 3))
    xyz = np.indices((n, n, n))
    u[..., 0] = np.sin(2 * np.pi * xyz[1] / n)
    u[..., 1] = np.sin(2 * np.pi * xyz[0] / n)
    u[0, ..., 0] += 8
    r = divergence_localization(u, (1, 1, 1), core_trim=4)
    assert r["shells"]["0"]["divergence_enrichment_over_uniform"] > 1
    assert r["outer_shell_divergence_energy_fraction"] > .5


def test_zero_field_avoids_division_by_zero():
    field = np.zeros((13, 13, 13, 3))
    face = face_jump_statistics(field, velocity_vector=True)
    assert face["x"]["wrap_over_interior"] is None
    r = divergence_localization(field[:-1, :-1, :-1], (1, 1, 1), core_trim=4)
    assert r["full_divergence_to_vorticity_rms"] is None
    assert r["shells"]["0"]["divergence_energy_fraction"] is None


def test_rejects_nonfinite_or_impossible_shapes():
    with pytest.raises(ValueError):
        face_jump_statistics(np.ones((4, 4, 4, 3)))
    with pytest.raises(ValueError):
        divergence_localization(np.ones((8, 8, 8, 3)), (1, 1, 1), core_trim=4)
