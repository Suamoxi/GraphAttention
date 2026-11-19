"""M47.7 Fourier covariance audit of physical velocity vectors.

This diagnostic intentionally goes beyond isotropic scalar energy spectra.
For a solenoidal Fourier vector, k_i uhat_i = 0. Consequently the
wavenumber-weighted cross-component terms must cancel the diagonal
longitudinal energy, even when ordinary shell-averaged off-diagonal
cospectra vanish by isotropy.

No projection, denoising or periodification is applied. The unique FFT
grid excludes stored duplicated maximum-coordinate endpoint planes.
"""
from __future__ import annotations

from typing import Any

import numpy as np

PAIRS = ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz"))
SECTORS = ("axis_like", "face_diagonal_like", "body_diagonal_like", "other")


def spectral_tensor_audit(
    velocity: np.ndarray, grid: Any, *, bins: int = 32,
) -> dict[str, Any]:
    """Angularly resolved shellwise auto/cross covariance, physical velocity.

    k is in rad/m. Spectral density integrates to m^2/s^2 over dk and is
    therefore in m^3/s^2. The FFT assumes periodicity, so results must be
    interpreted alongside interior finite-difference diagnostics.
    """
    u = np.asarray(velocity, dtype=np.float64)
    if u.shape != (*grid.shape, 3) or not np.isfinite(u).all():
        raise ValueError("velocity must have finite shape [Nx,Ny,Nz,3]")
    if not isinstance(bins, int) or bins < 4:
        raise ValueError("bins must be >= 4")
    dims = tuple(int(n) for n in grid.shape)
    spacing = tuple(float(dx) for dx in grid.spacing)
    if any(n < 8 for n in dims) or any(not np.isfinite(h) or h <= 0 for h in spacing):
        raise ValueError("invalid spatial discretization")
    kmax = float(grid.k_nyquist_min)
    if not np.isfinite(kmax) or kmax <= 0:
        raise ValueError("invalid Nyquist wavenumber")

    centered = u - u.mean(axis=(0, 1, 2), keepdims=True)
    U = np.fft.fftn(centered, axes=(0, 1, 2), norm="ortho")
    k = np.meshgrid(
        *(2 * np.pi * np.fft.fftfreq(n, d=h)
          for n, h in zip(dims, spacing)),
        indexing="ij",
    )
    k2 = sum(component**2 for component in k)
    kmag = np.sqrt(k2)
    select = (kmag > 0) & (kmag <= kmax)
    if not np.any(select):
        raise ValueError("no Fourier modes inside Nyquist sphere")

    N = int(np.prod(dims))
    energies = [0.5 * np.abs(U[..., i])**2 / N for i in range(3)]
    total = sum(energies)
    safe_k2 = np.where(k2 > 0, k2, 1)
    diag = sum((k[i]**2 / safe_k2) * energies[i] for i in range(3))
    cross_parts = {
        name: (k[i] * k[j] / safe_k2) * np.real(U[..., i] * U[..., j].conj()) / N
        for i, j, name in PAIRS
    }
    cross = sum(cross_parts.values())
    longitudinal = np.abs(sum(k[i] * U[..., i] for i in range(3)))**2 / (2 * N * safe_k2)
    transverse = total - longitudinal
    if np.min(transverse[select]) < -1e-10 * float(np.max(total[select])):
        raise ValueError("Fourier projection yielded negative transverse energy")
    transverse = np.maximum(transverse, 0)

    max_closure = float(
        np.max(np.abs(longitudinal[select] - (diag + cross)[select]))
        / max(float(np.max(total[select])), 1e-30)
    )
    if max_closure > 1e-10:
        raise ValueError(f"longitudinal covariance closure failure {max_closure}")

    edges = np.linspace(0, kmax, bins + 1)
    dk = kmax / bins
    shell_index = np.clip(np.searchsorted(edges, kmag[select], side="right") - 1,
                          0, bins - 1)
    counts_full = np.bincount(shell_index, minlength=bins)
    active = counts_full > 0

    def shell(a: np.ndarray, subset: np.ndarray | None = None) -> np.ndarray:
        inclusion = select if subset is None else select & subset
        where = np.clip(np.searchsorted(edges, kmag[inclusion], side="right") - 1,
                        0, bins - 1)
        return np.bincount(
            where, weights=np.asarray(a, dtype=np.float64)[inclusion],
            minlength=bins,
        )[active] / dk

    k_centers = (0.5 * (edges[:-1] + edges[1:]))[active].tolist()
    shell_counts = counts_full[active]
    E = shell(total)
    D = shell(diag)
    C = shell(cross)
    L = shell(longitudinal)
    T = shell(transverse)

    # Non-overlapping angular sectors in |k_x|, |k_y|, |k_z|. These
    # characterize direction-dependent transversality without imposing
    # an arbitrary sign on conjugate Fourier modes.
    unit = [np.abs(ki) / np.where(kmag > 0, kmag, 1) for ki in k]
    sorted_abs = np.sort(np.stack(unit, axis=-1), axis=-1)
    axis = sorted_abs[..., 2] >= 0.9
    face = (~axis) & (sorted_abs[..., 1] >= 0.6) & (sorted_abs[..., 0] <= 0.35)
    body = (~axis) & (~face) & (sorted_abs[..., 0] >= 0.4)
    sectors = {
        "axis_like": axis, "face_diagonal_like": face,
        "body_diagonal_like": body,
        "other": ~(axis | face | body),
    }
    if not np.all(sum(mask.astype(np.int8) for mask in sectors.values()) == 1):
        raise ValueError("non-disjoint angular sector classification")

    def nonzero_fraction(num: np.ndarray, den: np.ndarray) -> list[float | None]:
        return [float(a / b) if b > 0 else None for a, b in zip(num, den)]

    angular = {}
    for name, mask in sectors.items():
        sector_E, sector_L = shell(total, mask), shell(longitudinal, mask)
        angular[name] = {
            "mode_count_per_shell": np.bincount(
                np.clip(
                    np.searchsorted(edges, kmag[select & mask], side="right") - 1,
                    0, bins - 1,
                ), minlength=bins,
            )[active].astype(int).tolist(),
            "E_total_m3_per_s2": sector_E.tolist(),
            "E_longitudinal_m3_per_s2": sector_L.tolist(),
            "longitudinal_fraction_per_shell": nonzero_fraction(sector_L, sector_E),
        }

    diag_energy = float(np.sum(diag[select]))
    cross_energy = float(np.sum(cross[select]))
    total_energy = float(np.sum(total[select]))
    longitudinal_energy = float(np.sum(longitudinal[select]))
    return {
        "physical_wavenumber_units": "rad/m",
        "spectral_density_units": "m^3/s^2",
        "interpretation_warning": (
            "Raw periodic FFT of unique grid; seams affect the decomposition. "
            "Ordinary shellwise off-diagonal cospectra may vanish for isotropic "
            "solenoidal turbulence; weighted cross terms reveal transversality."
        ),
        "k_rad_per_m": k_centers,
        "mode_count_per_shell": shell_counts.astype(int).tolist(),
        "E_total_m3_per_s2": E.tolist(),
        "E_x_m3_per_s2": shell(energies[0]).tolist(),
        "E_y_m3_per_s2": shell(energies[1]).tolist(),
        "E_z_m3_per_s2": shell(energies[2]).tolist(),
        "E_longitudinal_m3_per_s2": L.tolist(),
        "E_transverse_m3_per_s2": T.tolist(),
        "E_longitudinal_diagonal_m3_per_s2": D.tolist(),
        "E_longitudinal_cross_m3_per_s2": C.tolist(),
        "E_longitudinal_cross_xy_m3_per_s2": shell(cross_parts["xy"]).tolist(),
        "E_longitudinal_cross_xz_m3_per_s2": shell(cross_parts["xz"]).tolist(),
        "E_longitudinal_cross_yz_m3_per_s2": shell(cross_parts["yz"]).tolist(),
        "real_cross_cospectra_m3_per_s2": {
            name: shell(np.real(U[..., i] * U[..., j].conj()) / (2 * N)).tolist()
            for i, j, name in PAIRS
        },
        "cross_cancellation_per_shell": [
            float(-c / d) if d > 0 else None for c, d in zip(C, D)
        ],
        "longitudinal_fraction_per_shell": nonzero_fraction(L, E),
        "longitudinal_energy_fraction_integrated": (
            longitudinal_energy / total_energy if total_energy > 0 else None
        ),
        "cross_cancellation_integrated": (
            -cross_energy / diag_energy if diag_energy > 0 else None
        ),
        "maximum_longitudinal_covariance_closure_error": max_closure,
        "angular_sectors": angular,
    }


def aggregate_spectral_tensor(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Population mean curves and per-field scalar ratios (never pool fields)."""
    if not records:
        raise ValueError("cannot aggregate empty population")
    k = records[0]["k_rad_per_m"]
    if any(row["k_rad_per_m"] != k for row in records):
        raise ValueError("inconsistent wavenumber bins")
    keys = (
        "E_total_m3_per_s2", "E_x_m3_per_s2", "E_y_m3_per_s2",
        "E_z_m3_per_s2", "E_longitudinal_m3_per_s2",
        "E_transverse_m3_per_s2", "E_longitudinal_diagonal_m3_per_s2",
        "E_longitudinal_cross_m3_per_s2",
        "E_longitudinal_cross_xy_m3_per_s2",
        "E_longitudinal_cross_xz_m3_per_s2",
        "E_longitudinal_cross_yz_m3_per_s2",
    )
    result: dict[str, Any] = {"count": len(records), "k_rad_per_m": k}
    for name in keys:
        result[name] = np.mean([row[name] for row in records], axis=0).tolist()
    for name in ("longitudinal_energy_fraction_integrated",
                 "cross_cancellation_integrated"):
        vals = [row[name] for row in records]
        result["mean_" + name] = float(np.mean(vals)) if all(
            value is not None for value in vals
        ) else None
    result["mean_angular_sector_longitudinal_fraction"] = {}
    for sector in SECTORS:
        counts = np.asarray([
            row["angular_sectors"][sector]["mode_count_per_shell"]
            for row in records
        ], dtype=np.int64)
        e = np.mean([
            row["angular_sectors"][sector]["E_total_m3_per_s2"]
            for row in records
        ], axis=0)
        l = np.mean([
            row["angular_sectors"][sector]["E_longitudinal_m3_per_s2"]
            for row in records
        ], axis=0)
        result["mean_angular_sector_longitudinal_fraction"][sector] = {
            "mode_count_per_shell": counts[0].astype(int).tolist(),
            "longitudinal_fraction_per_shell": [
                float(a / b) if b > 0 else None for a, b in zip(l, e)
            ],
        }
        if not np.all(counts == counts[0]):
            raise ValueError("directional mode count changed across samples")
    return result
