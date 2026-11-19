"""CPU-only physical checks for M47.7 spectral tensor and qualified split."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.m47_7_covariance_metrics import (
    SECTORS, aggregate_spectral_tensor, spectral_tensor_audit,
)
from scripts.m47_7_single_regime_train import source_split


def _grid(n: int = 32):
    return SimpleNamespace(
        shape=(n, n, n),
        spacing=(1 / n, 1 / n, 1 / n),
        k_nyquist_min=np.pi * n,
    )


def _wave(plus: bool = False, n: int = 32):
    x = np.arange(n) / n
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    scalar = np.cos(2 * np.pi * (xx + yy))
    velocity = np.stack((
        scalar,
        scalar if plus else -scalar,
        np.zeros_like(scalar),
    ), axis=-1)
    return velocity


def test_wave_cross_covariance_cancels_longitudinal_diagonal():
    result = spectral_tensor_audit(_wave(False), _grid())
    assert result["longitudinal_energy_fraction_integrated"] < 1e-12
    assert result["cross_cancellation_integrated"] == pytest.approx(1.0, abs=1e-11)
    assert result["maximum_longitudinal_covariance_closure_error"] < 1e-11
    assert np.sum(result["E_longitudinal_cross_m3_per_s2"]) < 0
    assert np.sum(result["E_longitudinal_diagonal_m3_per_s2"]) > 0
    assert len(result["k_rad_per_m"]) == len(result["mode_count_per_shell"])


def test_potential_wave_is_longitudinal_and_cross_term_is_positive():
    result = spectral_tensor_audit(_wave(True), _grid())
    assert result["longitudinal_energy_fraction_integrated"] == pytest.approx(1.0)
    assert result["cross_cancellation_integrated"] == pytest.approx(-1.0)
    assert np.sum(result["E_longitudinal_cross_m3_per_s2"]) > 0


def test_independent_isotropic_gaussian_has_one_third_longitudinal_energy():
    u = np.random.default_rng(42).standard_normal((32, 32, 32, 3))
    result = spectral_tensor_audit(u, _grid())
    assert result["longitudinal_energy_fraction_integrated"] == pytest.approx(
        1 / 3, abs=0.035,
    )
    assert abs(result["cross_cancellation_integrated"]) < 0.04
    grouped = aggregate_spectral_tensor([result, result])
    assert grouped["mean_cross_cancellation_integrated"] == pytest.approx(
        result["cross_cancellation_integrated"]
    )
    counts = sum(
        np.asarray(result["angular_sectors"][sector]["mode_count_per_shell"])
        for sector in SECTORS
    )
    assert np.array_equal(counts, result["mode_count_per_shell"])


def test_reject_invalid_fields_and_no_mixing_of_wavenumber_grids():
    u = _wave()
    with pytest.raises(ValueError, match="finite"):
        spectral_tensor_audit(np.full_like(u, np.nan), _grid())
    with pytest.raises(ValueError, match="bins"):
        spectral_tensor_audit(u, _grid(), bins=0)
    good = spectral_tensor_audit(u, _grid())
    modified = dict(good)
    modified["k_rad_per_m"] = list(good["k_rad_per_m"])
    modified["k_rad_per_m"][0] *= 1.1
    with pytest.raises(ValueError, match="wavenumber"):
        aggregate_spectral_tensor([good, modified])


def test_single_regime_source_split_preserves_m47_1_membership(
    monkeypatch, tmp_path: Path,
):
    from scripts import m47_7_single_regime_train as train
    monkeypatch.setattr(train, "validate_provenance", lambda *a: {"verified": True})
    path = tmp_path / "epoch10"
    path.mkdir()
    names = lambda case, count: [
        f"{case}:solut_hit_{i:08d}" for i in range(count)
    ]
    content = {
        "train_ids": names("Re54/Ma0.2", 15) + names("Re94/Ma0.2", 15),
        "validation_ids": [
            f"Re54/Ma0.2:solut_hit_{i:08d}" for i in (80, 81, 82, 83, 84)
        ],
        "test_ids": [f"Re54/Ma0.2:solut_hit_{i:08d}" for i in (91, 92, 93)],
    }
    (path / "split.json").write_text(json.dumps(content))
    filtered, full = source_split(path)
    assert filtered.train_ids == tuple(names("Re54/Ma0.2", 15))
    assert len(filtered.validation_ids) == 5
    assert len(filtered.test_ids) == 3
    assert full == content
