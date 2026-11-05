"""Reference-state and well-resolved spectral similarity diagnostics for M43.

Numerical operations only. They never estimate physical reference values from DNS.
The input is the existing M3.3 eight-channel sampled state; output is a
five-channel *diagnostic* state, not a change to the training representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


DIAGNOSTIC_CHANNELS = ("density_fluctuation", "u", "v", "w", "energy_shifted")


@dataclass(frozen=True)
class SimilarityReferences:
    rho_ref: float
    u_ref: float
    l_ref: float
    rho0: float
    a_ref: float
    e_ref: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")

    @property
    def mach(self) -> float:
        return self.u_ref / self.a_ref

    @property
    def e0_over_u0_squared(self) -> float:
        return self.e_ref / self.u_ref**2


def similarity_transform(values: np.ndarray, refs: SimilarityReferences) -> np.ndarray:
    """Transform sampled M3.3 [rho*,rhou*,rhov*,rhow*,rhoE*,u*,v*,w*].

    density_fluctuation = (rho - rho0)/(rho0 * (U_ref/a_ref)**2).
    energy_shifted = (rhoE - rho*e_ref)/(rho_ref * U_ref**2).
    All physical reference values are fixed case inputs, never fitted to snapshots.
    """
    arr = np.asarray(values, dtype=np.float64)
    if (arr.ndim < 2 or arr.shape[-1] != 8 or not np.isfinite(arr).all()
        or np.any(arr[..., 0] <= 0)):
        raise ValueError("expected finite M3.3 sampled fields with last dimension 8 and rho*>0")
    expected_velocity = arr[..., 1:4] / arr[..., 0, None]
    if not np.allclose(arr[..., 5:8], expected_velocity, rtol=1e-9, atol=1e-9):
        raise ValueError("cached velocity is inconsistent with conservative M3.3 momenta")
    density = (arr[..., 0] * refs.rho_ref - refs.rho0) / (refs.rho0 * refs.mach**2)
    shifted = arr[..., 4] - arr[..., 0] * refs.e0_over_u0_squared
    return np.concatenate((density[..., None], arr[..., 5:8], shifted[..., None]), axis=-1)


def similarity_inverse(values: np.ndarray, refs: SimilarityReferences) -> np.ndarray:
    """Recover all five M3.3 conservative channels from the diagnostic state."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[-1] != 5 or not np.isfinite(arr).all():
        raise ValueError("expected finite diagnostic state with last dimension 5")
    rho_star = refs.rho0 * (1.0 + refs.mach**2 * arr[..., 0]) / refs.rho_ref
    if np.any(rho_star <= 0):
        raise ValueError("inverse similarity state has nonpositive density")
    momenta_star = rho_star[..., None] * arr[..., 1:4]
    rhoe_star = arr[..., 4] + rho_star * refs.e0_over_u0_squared
    return np.concatenate((rho_star[..., None], momenta_star, rhoe_star[..., None]), axis=-1)


def conservative_shell_limit(k: np.ndarray, *, unique_side: int, l_ref: float,
                             nyquist_fraction: float = 0.70) -> float:
    """Conservative spherical-shell limit below *directional* grid Nyquist.

    k is the full physical radial spectrum with uniform fundamental shells.
    Modes in spherical shells beyond directional Nyquist are incomplete on
    a Cartesian FFT cube. A safety margin avoids edge/shell-bin effects.
    """
    k = np.asarray(k, dtype=np.float64)
    if (k.ndim != 1 or len(k) < 10 or not np.isfinite(k).all()
        or np.any(np.diff(k) <= 0) or k[0] <= 0
        or not isinstance(unique_side, int) or unique_side < 16
        or not isfinite(l_ref) or l_ref <= 0
        or not 0.25 <= nyquist_fraction <= 0.95):
        raise ValueError("invalid spectral shells, reference length, or safety factor")
    spacing = np.diff(k)
    dk = float(np.median(spacing))
    if not np.allclose(spacing, dk, rtol=1e-7, atol=1e-10*dk):
        raise ValueError("expected uniformly spaced physical radial shells")
    if not np.isclose(k[0], dk, rtol=1e-6):
        raise ValueError("first shell must be the fundamental nonzero wavenumber")
    nyquist_star = (unique_side / 2) * dk * l_ref
    return min(float(nyquist_fraction * nyquist_star), float(k[-1] * l_ref))


def common_shell_spectra(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    references: dict[str, SimilarityReferences],
    grid_sides: dict[str, int], *, nyquist_fraction: float = 0.70,
    points: int = 128,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Interpolate time-resolved physical E(k) to shared dimensionless support.

    Each input spectrum has shape [snapshots, radial_shells]. The output
    retains the snapshot dimension for time-block uncertainty estimation.
    """
    if set(spectra) != set(references) or set(spectra) != set(grid_sides) or points < 16:
        raise ValueError("spectra/references/grid case IDs differ or grid too small")
    supports = {}
    lows = []
    highs = []
    for case, (k, e) in spectra.items():
        wave = np.asarray(k, dtype=np.float64)
        data = np.asarray(e, dtype=np.float64)
        if (data.ndim != 2 or data.shape[1] != len(wave)
            or not np.isfinite(data).all() or np.any(data < 0)):
            raise ValueError(f"{case}: invalid time-resolved spectra")
        r = references[case]
        supports[case] = conservative_shell_limit(
            wave, unique_side=grid_sides[case], l_ref=r.l_ref,
            nyquist_fraction=nyquist_fraction,
        )
        lows.append(float(wave[0] * r.l_ref))
        highs.append(supports[case])
    start, stop = max(lows), min(highs)
    if stop <= start * 1.5:
        raise ValueError("insufficient shared conservative spectral support")
    common_k = np.geomspace(start, stop, points)
    result = {}
    for case, (wave, data) in spectra.items():
        refs = references[case]
        kstar = np.asarray(wave, dtype=float) * refs.l_ref
        e_star = np.asarray(data, dtype=float) / (refs.u_ref**2 * refs.l_ref)
        result[case] = np.stack([np.interp(common_k, kstar, row) for row in e_star])
    return common_k, result, supports


def spectral_relative_l1(k: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    """Energy-weighted relative L1 on a shared physical nondimensional k grid."""
    x = np.asarray(k, dtype=float)
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if (x.ndim != 1 or len(x) < 2 or a.shape != x.shape or b.shape != x.shape
        or np.any(np.diff(x) <= 0) or not np.isfinite(a).all()
        or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0)):
        raise ValueError("invalid shared spectra")
    dx = np.diff(x)
    integrate = lambda y: float(np.sum(0.5 * (y[:-1] + y[1:]) * dx))
    return integrate(np.abs(a - b)) / max(0.5 * (integrate(a) + integrate(b)), 1e-30)


def circular_block_indices(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """Resample whole snapshots as temporal blocks; do not resample spatial nodes."""
    if n < 4 or not 1 <= length <= n:
        raise ValueError("invalid block resampling shape")
    starts = rng.integers(n, size=(n + length - 1) // length)
    return ((starts[:, None] + np.arange(length)[None, :]) % n).ravel()[:n]
