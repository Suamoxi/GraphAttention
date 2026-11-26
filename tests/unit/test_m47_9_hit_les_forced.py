"""M47.9: regression tests for historical periodic HIT Helmholtz analysis."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.m47_9_hit_les_forced_helmholtz import (
    _fingerprint, _generation, _population,
)


def _sample(mode):
    # The audit checks trim=4, so the unique test grid must have n > 8.
    n = 12
    xyz = np.indices((n + 1,) * 3)
    phase = 2 * np.pi * (xyz[0] + xyz[1]) / n
    p = np.cos(phase)
    q = np.zeros((n + 1, n + 1, n + 1, 5), dtype=np.float64)
    q[..., 0] = 1
    q[..., 1] = p
    q[..., 2] = p if mode == "longitudinal" else -p
    q[..., 4] = 5
    grid = SimpleNamespace(
        source_shape=(n + 1, n + 1, n + 1), shape=(n, n, n),
        spacing=(1.0, 1.0, 1.0), k_nyquist_min=np.pi,
        linear_indices=np.arange((n + 1) ** 3),
        periodic_endpoint_mode="drop_max",
    )
    return q.reshape(-1, 5), grid


def test_solenoidal_and_dilatational_sinusoid_are_distinguished():
    trans, grid = _sample("transverse")
    long, _ = _sample("longitudinal")
    channels = ("rho.value", "rhou.x", "rhov.y", "rhow.z", "rhoE.value")
    a = _population(trans[None], ("T",), channels, grid, 17.360947554785138)
    b = _population(long[None], ("L",), channels, grid, 17.360947554785138)
    assert a["valid_count"] == b["valid_count"] == 1
    t = a["scalar_statistics"]
    l = b["scalar_statistics"]
    assert t["longitudinal_energy_fraction"]["mean"] < 1e-12
    assert l["longitudinal_energy_fraction"]["mean"] == pytest.approx(1, abs=1e-12)
    assert t["cross_cancellation"]["mean"] == pytest.approx(1, abs=1e-12)
    assert l["cross_cancellation"]["mean"] == pytest.approx(-1, abs=1e-12)
    assert t["tke_fluctuation_m2_s2"]["mean"] > 0


def test_bad_density_is_not_masked_or_silently_evaluated():
    field, grid = _sample("transverse")
    field[0, 0] = -1
    with pytest.raises(ValueError, match="no valid positive-density fields"):
        _population(
            field[None], ("bad",),
            ("rho", "rhou", "rhov", "rhow", "rhoE"),
            grid, 17.36,
        )


def test_artifact_reference_fingerprint_and_arg_validation():
    a = np.arange(75, dtype=np.float64).reshape(3, 5, 5)
    assert _fingerprint(a) == _fingerprint(a.copy())
    assert _fingerprint(a) != _fingerprint(a + .001)
    assert _generation("M37=/scratch/foo")[0] == "M37"
    with pytest.raises(Exception):
        _generation("bad-no-path")
    with pytest.raises(Exception):
        _generation("M38=relative/path")


def test_archived_generation_loader_accepts_valid_population_ids(tmp_path, monkeypatch):
    """Exercise _load itself so mismatched local variable names fail preflight."""
    import scripts.m47_9_hit_les_forced_helmholtz as audit

    for filename in ("generated_test.pt", "resolved_config.yaml", "summary.json"):
        (tmp_path / filename).write_text("fixture")

    grid = SimpleNamespace(source_shape=(3, 3, 1))
    monkeypatch.setattr(
        audit, "_load_grid_3d",
        lambda source_cfg, benchmark_cfg: (grid, tmp_path / "mesh.h5"),
    )
    monkeypatch.setattr(audit.OmegaConf, "load", lambda _: {})
    monkeypatch.setattr(audit.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        audit, "_fixed_mesh_samples",
        lambda artifact: (
            np.zeros((2, 9, 5)), np.zeros((2, 9, 5)),
            ("generated_0", "generated_1"),
            ("reference_0", "reference_1"),
            ("rho", "rhou", "rhov", "rhow", "rhoE"),
        ),
    )

    result = audit._load(tmp_path, {})
    assert result[2] == ("generated_0", "generated_1")
    assert result[3] == ("reference_0", "reference_1")
    assert result[0].shape == (2, 9, 5)
