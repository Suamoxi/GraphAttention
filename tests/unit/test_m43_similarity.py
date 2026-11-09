"""Scientific M43 similarity regression gates; synthetic only, no DNS required."""
from __future__ import annotations

import numpy as np
import pytest

from graph_attention.evaluation.m43_similarity import (
    SimilarityReferences, circular_block_indices, common_shell_spectra,
    conservative_shell_limit, similarity_inverse, similarity_transform,
    spectral_relative_l1,
)


def _refs(mach: float = 0.1) -> SimilarityReferences:
    return SimilarityReferences(rho_ref=1.2, u_ref=2.0, l_ref=1.0,
                                rho0=1.1, a_ref=2/mach, e_ref=80.0)


def _samples(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rho_star = 1.1/1.2 + rng.uniform(-0.02, 0.02, (6, 13))
    velocity = rng.normal(size=(6, 13, 3))
    rhoe_star = rho_star * 20 + rng.normal(size=(6, 13))
    return np.concatenate((rho_star[..., None], rho_star[..., None]*velocity,
                           rhoe_star[..., None], velocity), axis=-1)


def test_reference_state_is_exactly_invertible() -> None:
    x = _samples()
    refs = _refs()
    y = similarity_transform(x, refs)
    np.testing.assert_allclose(y[..., 0], (x[..., 0]*1.2-1.1)/(1.1*0.1**2))
    np.testing.assert_allclose(y[..., 4], x[..., 4]-x[..., 0]*20)
    np.testing.assert_allclose(similarity_inverse(y, refs), x[..., :5], atol=1e-12)


def test_density_scaling_uses_reference_mach_not_case_label() -> None:
    x = _samples()
    a = similarity_transform(x, _refs(0.1))
    b = similarity_transform(x, _refs(0.2))
    np.testing.assert_allclose(a[..., 0], 4*b[..., 0], rtol=1e-12)
    np.testing.assert_allclose(a[..., 4], b[..., 4])
    np.testing.assert_allclose(a[..., 1:4], b[..., 1:4])


def test_reference_centering_is_not_fitted_to_sample() -> None:
    x = _samples()
    actual = similarity_transform(x, _refs())
    wrong = SimilarityReferences(1.2, 2.0, 1.0, 1.2, 20.0, 80.0)
    shifted = similarity_transform(x, wrong)
    assert abs(actual[..., 0].mean() - shifted[..., 0].mean()) > 7


def test_fail_closed_on_bad_state_and_refs() -> None:
    x = _samples()
    bad = x.copy()
    bad[..., 5] += 0.1
    with pytest.raises(ValueError, match="inconsistent"):
        similarity_transform(bad, _refs())
    bad = x.copy()
    bad[..., 0] = 0
    with pytest.raises(ValueError, match="rho"):
        similarity_transform(bad, _refs())
    with pytest.raises(ValueError, match="rho0"):
        SimilarityReferences(1.2, 2, 1, 0, 20, 80)
    with pytest.raises(ValueError, match="nonpositive"):
        similarity_inverse(np.full((2, 3, 5), -1e7), _refs())


def test_common_wavenumber_support_respects_nyquist_of_64_cubed_grid() -> None:
    # k fundamental: 2*pi/L with L_ref=L=1. Re54 64^3; Re94 128^3.
    k = 2*np.pi*np.arange(1, 116)
    specs = {
        "Re54": (k, np.ones((5, len(k))) * np.exp(-k/20)),
        "Re94": (k, np.ones((5, len(k))) * np.exp(-k/25)),
    }
    refs = {case: _refs() for case in specs}
    shared, data, support = common_shell_spectra(
        specs, refs, {"Re54": 64, "Re94": 128})
    assert support["Re54"] == pytest.approx(0.70*np.pi*64)
    assert shared[-1] <= 0.70*np.pi*64 + 1e-12
    assert shared[-1] < np.pi*64
    assert shared[-1] < shared[0]*30
    assert all(x.shape == (5, 128) for x in data.values())
    with pytest.raises(ValueError, match="uniform"):
        conservative_shell_limit(np.array([1,2,3,5,6,7,8,9,10,11]), unique_side=64,
                                 l_ref=1)


def test_spectral_distance_and_temporal_blocks() -> None:
    x = np.linspace(1, 5, 200)
    a = np.exp(-x)
    np.testing.assert_allclose(spectral_relative_l1(x, a, a), 0, atol=1e-14)
    assert spectral_relative_l1(x, a, 2*a) == pytest.approx(2/3, abs=1e-12)
    rng = np.random.default_rng(4)
    for n, length in ((61, 9), (20, 3)):
        chosen = circular_block_indices(n, length, rng)
        assert chosen.shape == (n,)
        assert np.all((chosen >= 0) & (chosen < n))
        assert sum(((np.diff(chosen) + n) % n) == 1) >= n - (n+length-1)//length - 1
