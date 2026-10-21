"""DNS-only, periodic structured-HIT residual statistics.

All velocities are dimensional unless the caller explicitly transforms them.
A block average of *unique periodic samples* is conservative for the discrete
equal-voxel quadrature. It is NOT claimed to be AVBP finite-volume restriction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CONSERVATIVE_FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def _cube(field: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] < 1:
        raise ValueError("expected [nx,ny,nz,channels]")
    if min(values.shape[:3]) < 1 or not np.isfinite(values).all():
        raise ValueError("invalid spatial dimensions or nonfinite fields")
    return values


def restrict_blocks(field: np.ndarray, factor: int) -> np.ndarray:
    """Equal-volume discrete restriction on unique periodic Cartesian samples."""
    a = _cube(field)
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 2:
        raise ValueError("factor must be an integer >= 2")
    nx, ny, nz, channels = a.shape
    if any(n % factor for n in (nx, ny, nz)):
        raise ValueError("filter factor does not divide unique periodic dimensions")
    return a.reshape(
        nx // factor, factor, ny // factor, factor, nz // factor, factor, channels
    ).mean(axis=(1, 3, 5))


def lift_blocks(field: np.ndarray, factor: int) -> np.ndarray:
    a = _cube(field)
    for axis in range(3):
        a = np.repeat(a, factor, axis=axis)
    return a


def conservative_residual(field: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    coarse = restrict_blocks(field, factor)
    residual = _cube(field) - lift_blocks(coarse, factor)
    return coarse, residual


def velocity_from_conservative(q: np.ndarray) -> np.ndarray:
    a = _cube(q)
    if a.shape[-1] != 5 or np.any(a[..., 0] <= 0):
        raise ValueError("expected five conservative fields and positive density")
    return a[..., 1:4] / a[..., 0, None]


def spectral_split(
    velocity: np.ndarray, spacing: tuple[float, float, float], factor: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Orthogonal spherical sharp-cutoff split (velocity, not conservative q)."""
    u = _cube(velocity)
    if u.shape[-1] != 3:
        raise ValueError("spectral_split expects a three-component velocity")
    if not isinstance(factor, int) or factor < 2:
        raise ValueError("factor >= 2 required")
    d = np.asarray(spacing, dtype=np.float64)
    if d.shape != (3,) or np.any(d <= 0):
        raise ValueError("positive 3-D Cartesian spacing required")
    wave = [2 * np.pi * np.fft.fftfreq(u.shape[a], d=d[a]) for a in range(3)]
    kmag = np.sqrt(sum(k**2 for k in np.meshgrid(*wave, indexing="ij")))
    cutoff = float(np.min(np.pi / (d * factor)))
    mask = kmag <= cutoff + 1.0e-10 * cutoff
    spectrum = np.fft.fftn(u, axes=(0, 1, 2), norm="ortho")
    low = np.fft.ifftn(spectrum * mask[..., None], axes=(0, 1, 2), norm="ortho").real
    high = u - low
    total = float(np.mean(np.sum(u**2, axis=-1)))
    split = float(np.mean(np.sum(low**2 + high**2, axis=-1)))
    return low, high, {
        "cutoff_rad_per_length": cutoff,
        "parseval_relative_error": abs(total - split) / max(total, 1.0e-30),
    }


def radial_velocity_spectrum(
    velocity: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """3-D shell-integrated velocity-specific energy density E(k) in physical k."""
    u = _cube(velocity)
    if u.shape[-1] != 3:
        raise ValueError("velocity must have three components")
    d = np.asarray(spacing, dtype=np.float64)
    if not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError("spacing must be finite and positive")
    u = u - u.mean(axis=(0, 1, 2), keepdims=True)
    n = np.asarray(u.shape[:3])
    wave = [2 * np.pi * np.fft.fftfreq(n[a], d=d[a]) for a in range(3)]
    k = np.sqrt(sum(x**2 for x in np.meshgrid(*wave, indexing="ij")))
    dk = float(np.min(2 * np.pi / (n * d)))
    fhat = np.fft.fftn(u, axes=(0, 1, 2), norm="ortho")
    mode_energy = 0.5 * np.sum(np.abs(fhat)**2, axis=-1) / np.prod(n)
    shells = np.floor(k / dk + 0.5).astype(np.int32)
    power = np.bincount(shells.reshape(-1), weights=mode_energy.reshape(-1))
    indices = np.arange(1, power.size)
    return indices.astype(np.float64) * dk, power[1:] / dk


def periodic_gradient(a: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Gradient with last dimension indexing derivative direction."""
    return np.stack(
        [(np.roll(a, -1, axis=j) - np.roll(a, 1, axis=j)) / (2 * spacing[j])
         for j in range(3)],
        axis=-1,
    )


def coarse_descriptors(
    conservative: np.ndarray, factor: int, spacing: tuple[float, float, float]
) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray]:
    """Coarse features and local fine residual targets per coarse block.

    Boundary increment targets are measured across fine nodes straddling
    coarse-block interfaces, including periodic wrap, at one fine-grid step.
    """
    q = _cube(conservative)
    if q.shape[-1] != 5:
        raise ValueError("expected five conservative fields")
    c, delta_q = conservative_residual(q, factor)
    uc = velocity_from_conservative(c)
    u = velocity_from_conservative(q)
    favre_lift = lift_blocks(uc, factor)
    delta_u = u - favre_lift
    block_energy = restrict_blocks(0.5 * np.sum(delta_u**2, axis=-1)[..., None], factor)[..., 0]
    mean_uc = uc.mean(axis=(0, 1, 2), keepdims=True)
    local_coarse_energy = 0.5 * np.sum((uc - mean_uc)**2, axis=-1)
    coarse_spacing = tuple(float(x) * factor for x in spacing)
    jac = periodic_gradient(uc, coarse_spacing)  # [..., velocity component, deriv axis]
    strain = 0.5 * (jac + np.swapaxes(jac, -1, -2))
    strain_mag = np.sqrt(np.maximum(0, 2 * np.sum(strain**2, axis=(-1, -2))))
    curl = np.stack([
        jac[..., 2, 1] - jac[..., 1, 2],
        jac[..., 0, 2] - jac[..., 2, 0],
        jac[..., 1, 0] - jac[..., 0, 1],
    ], axis=-1)
    vort_mag = np.linalg.norm(curl, axis=-1)
    divergence = np.trace(jac, axis1=-2, axis2=-1)
    inner = np.zeros_like(block_energy)
    crossing = np.zeros_like(block_energy)
    # A per-cell face is assigned to its upstream (lower-index) block.
    for axis in range(3):
        shifted = np.roll(delta_u, -1, axis=axis)
        increment_sq = np.sum((shifted - delta_u)**2, axis=-1)
        for offset, dest in ((factor - 2, inner), (factor - 1, crossing)):
            sl = [slice(None)] * 3
            sl[axis] = slice(offset, None, factor)
            face = increment_sq[tuple(sl)]
            if axis == 0:
                block_values = face.reshape(
                    c.shape[0], c.shape[1], factor, c.shape[2], factor
                ).mean(axis=(2, 4))
            elif axis == 1:
                block_values = face.reshape(
                    c.shape[0], factor, c.shape[1], c.shape[2], factor
                ).mean(axis=(1, 4))
            else:
                block_values = face.reshape(
                    c.shape[0], factor, c.shape[1], factor, c.shape[2]
                ).mean(axis=(1, 3))
            dest += block_values / 3.0
    rho_mean = float(np.mean(c[..., 0]))
    features = {
        "log_coarse_energy": np.log1p(local_coarse_energy),
        "log_strain": np.log1p(strain_mag),
        "log_vorticity": np.log1p(vort_mag),
        "divergence": divergence,
        "rho_relative": c[..., 0] / rho_mean - 1,
        "unresolved_k": block_energy,
        "within_s2": inner,
        "cross_s2": crossing,
    }
    summary = {
        "coarse_null_max_abs": float(np.max(np.abs(restrict_blocks(delta_q, factor)))),
        "coarse_restriction_max_abs": float(np.max(np.abs(
            restrict_blocks(lift_blocks(c, factor), factor) - c
        ))),
        "unresolved_velocity_k": float(np.mean(block_energy)),
        "within_s2": float(np.mean(inner)),
        "cross_s2": float(np.mean(crossing)),
        "resolved_k": float(np.mean(local_coarse_energy)),
        "mean_strain": float(np.mean(strain_mag)),
        "mean_vorticity": float(np.mean(vort_mag)),
    }
    return features, summary, delta_u


def spatial_residual_statistics(
    residual_velocity: np.ndarray, spacing: tuple[float, float, float], max_lag: int
) -> dict[str, np.ndarray]:
    u = _cube(residual_velocity)
    if u.shape[-1] != 3:
        raise ValueError("three-component residual velocity required")
    if not 1 <= max_lag <= min(u.shape[:3]) // 2:
        raise ValueError("invalid max_lag")
    center = u - u.mean(axis=(0, 1, 2), keepdims=True)
    var = np.mean(center**2, axis=(0, 1, 2))
    corr, s2, s3, s4 = [], [], [], []
    for lag in range(1, max_lag + 1):
        vals = []
        for axis in range(3):
            shifted = np.roll(center, -lag, axis=axis)
            v = shifted[..., axis] - center[..., axis]
            cv = (float(np.mean(center[..., axis] * shifted[..., axis]) / var[axis])
                  if var[axis] > 0 else float("nan"))
            vals.append((cv, float(np.mean(v**2)), float(np.mean(v**3)),
                         float(np.mean(v**4))))
        arr = np.asarray(vals)
        corr.append(float(np.nanmean(arr[:, 0])) if np.any(np.isfinite(arr[:, 0]))
                    else float("nan"))
        s2.append(float(np.mean(arr[:, 1])))
        s3.append(float(np.mean(arr[:, 2])))
        s4.append(float(np.mean(arr[:, 3])))
    return {
        "lag": np.arange(1, max_lag + 1, dtype=np.int32),
        "r": np.arange(1, max_lag + 1, dtype=np.float64) * float(np.mean(spacing)),
        "r_ll": np.asarray(corr),
        "s2_ll": np.asarray(s2),
        "s3_ll": np.asarray(s3),
        "s4_ll": np.asarray(s4),
    }


@dataclass(frozen=True)
class StationarityCandidate:
    status: str
    earliest_index: int | None
    latest_index: int | None
    explanation: str


def stationarity_candidates(
    data: np.ndarray, *, window: int = 12, tolerance: float = 0.15
) -> StationarityCandidate:
    """Conservative rolling-mean heuristic; NEVER an automatic cutoff."""
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 * window or x.shape[1] < 2:
        return StationarityCandidate("inconclusive", None, None, "insufficient samples")
    if not np.isfinite(x).all() or np.any(x < 0):
        return StationarityCandidate("inconclusive", None, None, "invalid diagnostics")
    anchor = np.mean(x[-window:], axis=0)
    if np.any(anchor <= 1.0e-14):
        return StationarityCandidate("inconclusive", None, None, "near-zero reference diagnostic")
    window_means = np.stack(
        [x[i:i + window].mean(axis=0) for i in range(len(x) - window + 1)]
    )
    error = np.max(np.abs(window_means - anchor) / anchor, axis=1)
    possible = np.flatnonzero(np.maximum.accumulate(error[::-1])[::-1] <= tolerance)
    if not possible.size:
        return StationarityCandidate(
            "inconclusive", None, None, "no sustained agreement with late-time means"
        )
    start = int(possible[0])
    return StationarityCandidate(
        "candidate_only", max(0, start - window // 2),
        min(len(x) - 1, start + window // 2),
        "rolling means agree with final-window means; manual spectral review required",
    )
