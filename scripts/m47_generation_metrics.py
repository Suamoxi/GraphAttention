"""Population diagnostics for M47 periodic-HIT Flow Matching ensembles.

All inputs are *convectively nondimensionalized conservative fields*, never
standardized training tensors. Spectral wavenumbers come from dimensional
mesh coordinates (rad/m). Invalid density/internal energy is reported, never
silently clipped, shifted, or converted into plausible velocity spectra.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from graph_attention.evaluation.spectra_3d import (
    CartesianGrid3D,
    sample_velocity_energy_spectra_3d,
    scatter_to_grid_3d,
)
from graph_attention.evaluation.spatial_statistics_3d import (
    sample_velocity_spatial_statistics_3d,
)

CHANNELS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def unique_node_mask(grid: CartesianGrid3D) -> np.ndarray:
    """Exclude repeated maximum-coordinate planes in endpoint-inclusive HIT."""
    if grid.periodic_endpoint_mode != "drop_max":
        raise ValueError("M47 FFT diagnostics require periodic_endpoint_mode='drop_max'")
    mapped = np.unravel_index(grid.linear_indices, grid.source_shape, order="C")
    return np.logical_and.reduce([axis < side for axis, side in zip(mapped, grid.shape)])


def sampled_nodes(
    state: np.ndarray, grid: CartesianGrid3D, *, key: str, count: int,
) -> np.ndarray:
    """Deterministic bounded spatial sample, identical across ablation arms."""
    if count < 1 or not key:
        raise ValueError("count must be positive and key nonempty")
    unique = np.asarray(state)[unique_node_mask(grid)]
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little")
    rng = np.random.default_rng(seed)
    choice = rng.choice(len(unique), size=min(count, len(unique)), replace=False)
    return unique[np.sort(choice)]


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Exact empirical W1 for two finite, possibly unequal-sized 1-D samples."""
    lhs = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    rhs = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    if not len(lhs) or not len(rhs):
        raise ValueError("Wasserstein samples must be nonempty")
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise ValueError("Wasserstein samples must be finite")
    union = np.sort(np.concatenate((lhs, rhs)))
    intervals = np.diff(union)
    cdf_gap = np.abs(
        np.searchsorted(lhs, union[:-1], side="right") / len(lhs)
        - np.searchsorted(rhs, union[:-1], side="right") / len(rhs)
    )
    return float(np.sum(intervals * cdf_gap))


def population_wasserstein(
    reference_nodes: list[np.ndarray], generated_nodes: list[np.ndarray],
) -> dict[str, Any]:
    """Channelwise nondimensional W1 and W1 normalized by reference std."""
    if not reference_nodes or not generated_nodes:
        raise ValueError("both populations must be nonempty")
    ref = np.concatenate(reference_nodes, axis=0)
    gen = np.concatenate(generated_nodes, axis=0)
    if ref.ndim != 2 or gen.ndim != 2 or ref.shape[1] != len(CHANNELS) or gen.shape[1] != len(CHANNELS):
        raise ValueError("expected five ordered conservative fields")
    out = {}
    for column, name in enumerate(CHANNELS):
        std = float(np.std(ref[:, column]))
        value = wasserstein_1d(ref[:, column], gen[:, column])
        out[name] = {
            "wasserstein_nondimensional": value,
            "reference_std_nondimensional": std,
            "wasserstein_over_ref_std": (value / std if std > 0 else None),
        }
    return out


def dilatational_energy_fraction(
    state: np.ndarray, grid: CartesianGrid3D,
) -> float:
    """FFT Helmholtz decomposition of velocity fluctuations on unique grid."""
    values = np.asarray(state, dtype=np.float64)
    rho = values[:, 0]
    if np.any(rho <= 0) or not np.isfinite(rho).all():
        raise ValueError("positive density required for velocity decomposition")
    velocity = np.stack([values[:, i] / rho for i in (1, 2, 3)], axis=-1)
    fields = np.stack(
        [scatter_to_grid_3d(velocity[:, i], grid) for i in range(3)], axis=-1
    )
    fields -= np.mean(fields, axis=(0, 1, 2), keepdims=True)
    u_hat = np.fft.fftn(fields, axes=(0, 1, 2), norm="ortho")
    wave = np.meshgrid(
        *(2 * np.pi * np.fft.fftfreq(n, d=dx)
          for n, dx in zip(grid.shape, grid.spacing)), indexing="ij"
    )
    squared = sum(k * k for k in wave)
    projection = np.zeros_like(u_hat)
    nonzero = squared > 0
    dot = sum(wave[i] * u_hat[..., i] for i in range(3))
    for i in range(3):
        projection[..., i][nonzero] = wave[i][nonzero] * dot[nonzero] / squared[nonzero]
    dil = float(np.sum(np.abs(projection) ** 2))
    sol = float(np.sum(np.abs(u_hat - projection) ** 2))
    if dil + sol <= 0:
        raise ValueError("zero total velocity-fluctuation energy")
    return dil / (dil + sol)


def periodic_endpoint_mismatch(
    state: np.ndarray, grid: CartesianGrid3D,
) -> dict[str, float | None]:
    """Endpoint jump RMS / unique-grid std; FFT drop_max must not hide seams."""
    values = np.asarray(state, dtype=np.float64)
    unique = values[unique_node_mask(grid)]
    result: dict[str, float | None] = {}
    for i, name in enumerate(CHANNELS):
        full = np.empty(int(np.prod(grid.source_shape)), dtype=np.float64)
        full[grid.linear_indices] = values[:, i]
        full = full.reshape(grid.source_shape)
        differences = []
        for axis in range(3):
            first = np.take(full, 0, axis=axis)
            last = np.take(full, -1, axis=axis)
            differences.append((last - first).reshape(-1))
        rms = float(np.sqrt(np.mean(np.concatenate(differences) ** 2)))
        std = float(np.std(unique[:, i]))
        result[name] = (rms / std if std > 0.0 else (0.0 if rms == 0.0 else None))
    return result


def field_diagnostics(
    state: np.ndarray, grid: CartesianGrid3D, channel_names: tuple[str, ...],
    *, k_bins: int, max_lag: int, velocity_scale_m_per_s: float,
) -> dict[str, Any]:
    """Analyze one independent field; inadmissible states retain raw moments."""
    q = np.asarray(state, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 5 or q.shape[0] != int(np.prod(grid.source_shape)):
        raise ValueError("M47 states must contain five conserved fields on source mesh")
    if tuple(c.split(".")[0] for c in channel_names) != CHANNELS:
        raise ValueError("unexpected M47 conservative-channel order")
    if not np.isfinite(q).all():
        raise ValueError("generated state contains NaN or Inf")
    if velocity_scale_m_per_s <= 0 or not math.isfinite(velocity_scale_m_per_s):
        raise ValueError("velocity_scale_m_per_s must be positive and finite")
    unique = q[unique_node_mask(grid)]
    rho = unique[:, 0]
    positive = rho > 0
    internal = np.full(rho.shape, np.nan, dtype=np.float64)
    internal[positive] = (
        unique[positive, 4]
        - np.sum(unique[positive, 1:4] ** 2, axis=1) / (2 * rho[positive])
    )
    valid_internal = int(np.count_nonzero(positive & (internal > 0)))
    result: dict[str, Any] = {
        "field_mean": {name: float(np.mean(unique[:, i])) for i, name in enumerate(CHANNELS)},
        "field_std": {name: float(np.std(unique[:, i])) for i, name in enumerate(CHANNELS)},
        "min_density": float(np.min(rho)),
        "periodic_endpoint_jump_over_field_std": periodic_endpoint_mismatch(q, grid),
        "nonpositive_density_fraction": float(np.mean(~positive)),
        "density_or_internal_energy_invalid_fraction": float(1.0 - valid_internal / len(rho)),
        "nonpositive_internal_energy_given_positive_density_fraction": (
            float(np.mean(internal[positive] <= 0)) if np.any(positive) else None
        ),
        "density_positive_everywhere": bool(np.all(positive)),
        "thermodynamically_admissible_everywhere": bool(valid_internal == len(rho)),
        "velocity_energy_spectrum": None,
        "dilatational_energy_fraction": None,
        "velocity_spatial_correlation": None,
    }
    if not result["density_positive_everywhere"]:
        return result
    try:
        centers, spectra = sample_velocity_energy_spectra_3d(
            q[None], grid, channel_names, num_k_bins=k_bins, subtract_mean=True,
        )
        dil_fraction = dilatational_energy_fraction(q, grid)
        correlation = sample_velocity_spatial_statistics_3d(
            q[None], grid, channel_names, max_lag=max_lag,
        )
    except ValueError as error:
        result["derived_diagnostics_unavailable_reason"] = str(error)
        return result
    result["velocity_energy_spectrum"] = {
        "k_rad_per_m": centers.tolist(),
        "E_m3_per_s2": (spectra[0] * velocity_scale_m_per_s ** 2).tolist(),
        "units_note": "Dimensional velocity spectrum using case U_ref and physical wavenumber",
    }
    result["dilatational_energy_fraction"] = float(dil_fraction)
    result["velocity_spatial_correlation"] = {
        "lag_m": (correlation.lag * grid.spacing[0]).tolist(),
        "vector_correlation": correlation.vector_correlation[0].tolist(),
    }
    return result


def mean_curves(records: list[dict[str, Any]], name: str, x_key: str, y_key: str) -> dict | None:
    """Mean physically permissible curves; report exclusion counts externally."""
    curves = [r[name] for r in records if r[name] is not None]
    if not curves:
        return None
    abscissa = np.asarray(curves[0][x_key], dtype=np.float64)
    ordinates = []
    for curve in curves:
        x = np.asarray(curve[x_key], dtype=np.float64)
        if x.shape != abscissa.shape or not np.array_equal(x, abscissa):
            raise ValueError("inconsistent population curve abscissae")
        ordinates.append(np.asarray(curve[y_key], dtype=np.float64))
    return {x_key: abscissa.tolist(), y_key: np.mean(ordinates, axis=0).tolist(), "sample_count": len(curves)}


def relative_spectrum_l1(reference: dict | None, generated: dict | None) -> float | None:
    """Discrete integral absolute E(k) difference over the common shell centers."""
    if reference is None or generated is None:
        return None
    rk = np.asarray(reference["k_rad_per_m"], dtype=np.float64)
    gk = np.asarray(generated["k_rad_per_m"], dtype=np.float64)
    if not np.array_equal(rk, gk):
        raise ValueError("reference and generated spectra have different physical wavenumbers")
    re = np.asarray(reference["E_m3_per_s2"], dtype=np.float64)
    ge = np.asarray(generated["E_m3_per_s2"], dtype=np.float64)
    # Equal-width original bins; empty bins were dropped consistently. The
    # common width cancels in the ratio, even when populated centers have gaps.
    denominator = float(np.sum(re))
    return float(np.sum(np.abs(ge - re)) / denominator) if denominator > 0 else None


def finite_mean(values: list[float | None]) -> float | None:
    keep = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(keep)) if keep else None
