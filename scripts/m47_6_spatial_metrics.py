"""Independent spatial-physics diagnostics for endpoint-inclusive periodic HIT.

Inputs are nondimensional conservative fields in the reviewed M47 channel order.
Physical velocity is obtained with each case's audited U_ref. No clipping,
periodification or other generated-field modification is performed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

CHANNELS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def scatter_unique(values: np.ndarray, grid: Any) -> np.ndarray:
    """Map flat original node order onto the original FFT unique grid."""
    values = np.asarray(values, dtype=np.float64)
    expected = int(np.prod(grid.source_shape))
    if values.shape[0] != expected:
        raise ValueError(f"expected {expected} original nodes")
    tail = values.shape[1:]
    full = np.empty((expected, *tail), dtype=np.float64)
    full[grid.linear_indices] = values
    full = full.reshape((*grid.source_shape, *tail))
    if grid.periodic_endpoint_mode != "drop_max":
        raise ValueError("M47.6 requires endpoint-inclusive source with drop_max FFT")
    return full[tuple(slice(0, n) for n in grid.shape)]


def physical_velocity(state: np.ndarray, grid: Any, u_ref: float) -> np.ndarray:
    """Return [Nx,Ny,Nz,3] velocity in m/s without touching source fields."""
    q = np.asarray(state, dtype=np.float64)
    if q.shape != (int(np.prod(grid.source_shape)), 5):
        raise ValueError("expected five conservative channels per source node")
    if not (np.isfinite(q).all() and np.isfinite(u_ref) and u_ref > 0):
        raise ValueError("nonfinite field or invalid velocity scale")
    if np.any(q[:, 0] <= 0):
        raise ValueError("strictly positive density required for velocity")
    return scatter_unique(u_ref * q[:, 1:4] / q[:, 0:1], grid)


def interior_divcurl(
    velocity: np.ndarray, spacing_m: tuple[float, float, float],
    *, trims: tuple[int, ...] = (0, 2, 4, 8),
) -> dict[str, dict[str, float | None]]:
    """Central differences, periodic at outermost planes; trimmed core is seam-free.

    RMS divergence and vorticity have units s^-1. For trim >= 2, sampled
    stencils cannot cross the periodic boundary. The diagnostic ratio is
    a local-gradient statistic, not a Helmholtz energy decomposition.
    """
    u = np.asarray(velocity, dtype=np.float64)
    if u.ndim != 4 or u.shape[-1] != 3 or min(u.shape[:3]) < 5:
        raise ValueError("velocity must have shape [Nx,Ny,Nz,3], Nx,Ny,Nz>=5")
    if not np.isfinite(u).all():
        raise ValueError("nonfinite physical velocity")
    h = tuple(float(x) for x in spacing_m)
    if len(h) != 3 or not all(np.isfinite(x) and x > 0 for x in h):
        raise ValueError("invalid spatial grid spacing")
    deriv = [[(np.roll(u[..., c], -1, axis=a) - np.roll(u[..., c], 1, axis=a))
              / (2 * h[a]) for a in range(3)] for c in range(3)]
    divergence = deriv[0][0] + deriv[1][1] + deriv[2][2]
    curl_x = deriv[2][1] - deriv[1][2]
    curl_y = deriv[0][2] - deriv[2][0]
    curl_z = deriv[1][0] - deriv[0][1]
    vort_squared = curl_x**2 + curl_y**2 + curl_z**2
    output = {}
    for trim in trims:
        if int(trim) != trim or trim < 0 or 2 * trim >= min(u.shape[:3]):
            raise ValueError("invalid boundary exclusion width")
        crop = tuple(slice(trim, -trim if trim else None) for _ in range(3))
        d2 = float(np.mean(divergence[crop] ** 2))
        w2 = float(np.mean(vort_squared[crop]))
        output[str(trim)] = {
            "excluded_cells_per_face": int(trim),
            "remaining_nodes": int(np.prod(divergence[crop].shape)),
            "rms_divergence_s_minus_1": float(np.sqrt(d2)),
            "rms_vorticity_s_minus_1": float(np.sqrt(w2)),
            "divergence_to_vorticity_rms": float(np.sqrt(d2 / w2)) if w2 > 0 else None,
            "divergence_gradient_fraction": d2 / (d2 + w2) if d2 + w2 > 0 else None,
        }
    return output


def _wave_numbers(shape: tuple[int, int, int], spacing: tuple[float, float, float]):
    return np.meshgrid(*(
        2 * np.pi * np.fft.fftfreq(n, d=dx) for n, dx in zip(shape, spacing)
    ), indexing="ij")


def helmholtz_spectrum(
    velocity: np.ndarray, grid: Any, *, k_bins: int = 32,
    window: str = "none",
) -> dict[str, Any]:
    """Density-free FFT projection into longitudinal and transverse energies.

    The raw FFT assumes periodicity. The optional Hann-window result probes
    boundary sensitivity but is NOT a physical solenoidal decomposition:
    multiplication by a window itself introduces divergence.
    """
    u = np.asarray(velocity, dtype=np.float64)
    if u.shape != (*grid.shape, 3) or not np.isfinite(u).all():
        raise ValueError("velocity/grid mismatch or nonfinite velocity")
    if k_bins < 1:
        raise ValueError("k_bins must be positive")
    if window not in ("none", "hann"):
        raise ValueError("window must be none or hann")
    field = u - u.mean(axis=(0, 1, 2), keepdims=True)
    if window == "hann":
        w = np.ones(grid.shape, dtype=np.float64)
        for axis, n in enumerate(grid.shape):
            shape = [1, 1, 1]
            shape[axis] = n
            w *= np.hanning(n).reshape(shape)
        normalization = float(np.sqrt(np.mean(w**2)))
        if normalization <= 0:
            raise ValueError("degenerate Hann window")
        field = field * (w / normalization)[..., None]
    uhat = np.fft.fftn(field, axes=(0, 1, 2), norm="ortho")
    wave = _wave_numbers(grid.shape, grid.spacing)
    wave2 = sum(ki**2 for ki in wave)
    kdot = sum(ki * uhat[..., i] for i, ki in enumerate(wave))
    nonzero = wave2 > 0
    npoints = int(np.prod(grid.shape))
    total_energy = 0.5 * np.sum(np.abs(uhat)**2, axis=3) / npoints
    dil_energy = np.zeros(grid.shape, dtype=np.float64)
    dil_energy[nonzero] = 0.5 * np.abs(kdot[nonzero])**2 / wave2[nonzero] / npoints
    sol_energy = np.maximum(0.0, total_energy - dil_energy)
    kmax = float(grid.k_nyquist_min)
    edges = np.linspace(0, kmax, k_bins + 1)
    kmag = np.sqrt(wave2)
    mask = (kmag > 0.0) & (kmag <= kmax)
    indices = np.clip(np.searchsorted(edges, kmag[mask], side="right") - 1, 0, k_bins - 1)
    count = np.bincount(indices, minlength=k_bins)
    active = count > 0
    if not active.any():
        raise ValueError("no valid nonzero Fourier modes")
    dk = kmax / k_bins

    def shell(energy: np.ndarray) -> list[float]:
        return (np.bincount(indices, weights=energy[mask], minlength=k_bins)[active]
                / dk).tolist()

    sum_tot = float(np.sum(total_energy[mask]))
    sum_dil = float(np.sum(dil_energy[mask]))
    sum_full = float(np.sum(total_energy[nonzero]))
    all_fraction = float(np.sum(dil_energy[nonzero]) / sum_full) if sum_full > 0 else None
    return {
        "window": window,
        "assumed_periodic": window == "none",
        "interpretation": (
            "Hann-tapered Fourier projection is boundary-sensitivity only; "
            "windowing generates additional divergence"
            if window == "hann" else "Periodic FFT of original unique-grid velocity"
        ),
        "k_rad_per_m": (0.5 * (edges[:-1] + edges[1:]))[active].tolist(),
        "E_total_m3_per_s2": shell(total_energy),
        "E_solenoidal_m3_per_s2": shell(sol_energy),
        "E_dilatational_m3_per_s2": shell(dil_energy),
        "fraction_dilatational_in_k_range": sum_dil / sum_tot if sum_tot > 0 else None,
        "fraction_dilatational_all_modes": all_fraction,
        "max_relative_shell_closure_error": float(
            np.max(np.abs(np.asarray(shell(total_energy)) - np.asarray(shell(sol_energy))
                          - np.asarray(shell(dil_energy))))
            / max(float(np.max(np.asarray(shell(total_energy)))), 1e-30)
        ),
    }


def standardized_momentum_diagnostics(
    standardized: np.ndarray, grid: Any, *, k_bins: int = 32,
) -> dict[str, Any]:
    """Spectral development of standardized momentum, not physical turbulence.

    Intermediate Flow Matching states are noisy standardized variables; they
    are not required to have positive density or internal energy.
    """
    q = np.asarray(standardized, dtype=np.float64)
    if q.shape != (int(np.prod(grid.source_shape)), 5) or not np.isfinite(q).all():
        raise ValueError("unexpected intermediate standardized state")
    momentum = scatter_unique(q[:, 1:4], grid)
    centered = momentum - momentum.mean(axis=(0, 1, 2), keepdims=True)
    fft = np.fft.fftn(centered, axes=(0, 1, 2), norm="ortho")
    modes = 0.5 * np.sum(np.abs(fft)**2, axis=3) / np.prod(grid.shape)
    wave = _wave_numbers(grid.shape, grid.spacing)
    kmag = np.sqrt(sum(ki**2 for ki in wave))
    kmax = float(grid.k_nyquist_min)
    edges = np.linspace(0, kmax, k_bins + 1)
    valid = (kmag > 0) & (kmag <= kmax)
    idx = np.clip(np.searchsorted(edges, kmag[valid], side="right") - 1, 0, k_bins - 1)
    active = np.bincount(idx, minlength=k_bins) > 0
    spec = np.bincount(idx, weights=modes[valid], minlength=k_bins)[active] / (kmax / k_bins)
    correlations = []
    for ax in range(3):
        left = centered[tuple(slice(None, -1) if j == ax else slice(None) for j in range(3))]
        right = centered[tuple(slice(1, None) if j == ax else slice(None) for j in range(3))]
        norm = float(np.sqrt(np.mean(left**2) * np.mean(right**2)))
        correlations.append(float(np.mean(left * right) / norm) if norm > 0 else None)
    return {
        "units": "standardized momentum squared per rad/m (not physical velocity)",
        "k_rad_per_m": (0.5 * (edges[:-1] + edges[1:]))[active].tolist(),
        "E_standardized_momentum_per_rad_per_m": spec.tolist(),
        "mean_absolute_standardized_momentum": float(np.mean(np.abs(momentum))),
        "interior_nearest_neighbour_correlation_xyz": correlations,
    }
