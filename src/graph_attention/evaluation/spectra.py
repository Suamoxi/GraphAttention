"""Cartesian 2-D spectral utilities for post-processing generated CFD slices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartesianGrid2D:
    """Structured-grid metadata inferred from flat 2-D node coordinates."""

    shape: tuple[int, int]
    spacing: tuple[float, float]
    linear_indices: np.ndarray
    k_nyquist_min: float


def infer_cartesian_grid_2d(
    coords: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
    shape_hint: tuple[int, int] | None = None,
) -> CartesianGrid2D:
    """Infer a complete uniform Cartesian 2-D grid from flat coordinates."""

    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected coords with shape [N, 2], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Cartesian coordinates contain NaN or Inf")

    tol = max(float(tolerance), 0.0)
    axes = [_unique_axis_values(arr[:, axis], tol) for axis in range(2)]
    shape = (int(axes[0].size), int(axes[1].size))
    if shape_hint is not None and shape != tuple(shape_hint):
        raise ValueError(f"coordinate grid shape {shape} does not match expected {shape_hint}")
    if shape[0] < 2 or shape[1] < 2:
        raise ValueError(f"Cartesian 2-D grid requires at least two points per axis, got {shape}")
    if shape[0] * shape[1] != arr.shape[0]:
        raise ValueError(
            f"coordinates do not form a complete Cartesian product: shape={shape}, nodes={arr.shape[0]}"
        )

    spacing: list[float] = []
    axis_indices: list[np.ndarray] = []
    for axis, axis_values in enumerate(axes):
        diffs = np.diff(axis_values)
        mean_spacing = float(np.mean(diffs))
        if mean_spacing <= 0.0:
            raise ValueError("Cartesian grid spacing must be positive")
        allowed_spacing = max(
            tol * max(1.0, abs(mean_spacing)),
            float(np.finfo(np.float32).eps) * max(1.0, abs(mean_spacing)),
        )
        if float(np.max(np.abs(diffs - mean_spacing))) > allowed_spacing:
            raise ValueError("coordinates are not uniformly spaced")
        spacing.append(mean_spacing)

        positions = np.searchsorted(axis_values, arr[:, axis])
        candidates = np.stack(
            [
                np.clip(positions - 1, 0, axis_values.size - 1),
                np.clip(positions, 0, axis_values.size - 1),
            ],
            axis=1,
        )
        deltas = np.abs(axis_values[candidates] - arr[:, axis, None])
        indices = candidates[np.arange(arr.shape[0]), np.argmin(deltas, axis=1)]
        scale = np.maximum.reduce(
            [np.ones(arr.shape[0]), np.abs(axis_values[indices]), np.abs(arr[:, axis])]
        )
        allowed = np.maximum(
            tol * scale,
            float(np.finfo(np.float32).eps) * scale,
        )
        if np.any(np.abs(axis_values[indices] - arr[:, axis]) > allowed):
            raise ValueError("coordinates cannot be mapped to the inferred Cartesian grid")
        axis_indices.append(indices.astype(np.int64))

    linear_indices = np.ravel_multi_index(tuple(axis_indices), dims=shape, order="C").astype(
        np.int64
    )
    if np.unique(linear_indices).size != arr.shape[0]:
        raise ValueError("multiple nodes map to the same Cartesian grid point")

    dx, dy = spacing
    return CartesianGrid2D(
        shape=shape,
        spacing=(dx, dy),
        linear_indices=linear_indices,
        k_nyquist_min=float(min(np.pi / dx, np.pi / dy)),
    )


def scatter_to_grid(values: np.ndarray, grid: CartesianGrid2D) -> np.ndarray:
    """Scatter one flat node field into the Cartesian grid order."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    expected = grid.shape[0] * grid.shape[1]
    if flat.size != expected:
        raise ValueError(f"expected {expected} node values, got {flat.size}")
    result = np.empty(expected, dtype=np.float64)
    result[grid.linear_indices] = flat
    return result.reshape(grid.shape, order="C")


def population_radial_spectra(
    samples: np.ndarray,
    grid: CartesianGrid2D,
    *,
    num_k_bins: int,
    subtract_mean: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean radial shell power for samples shaped [S, N, C]."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected samples with shape [S, N, C], got {values.shape}")
    expected_nodes = grid.shape[0] * grid.shape[1]
    if values.shape[1] != expected_nodes:
        raise ValueError(f"expected {expected_nodes} nodes per sample, got {values.shape[1]}")
    bins = int(num_k_bins)
    if bins < 1:
        raise ValueError("num_k_bins must be positive")

    nx, ny = grid.shape
    dx, dy = grid.spacing
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kmag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    edges = np.linspace(0.0, grid.k_nyquist_min, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    valid = (kmag > 0.0) & (kmag <= grid.k_nyquist_min)
    raw_bin = np.searchsorted(edges, kmag[valid], side="right") - 1
    bin_index = np.clip(raw_bin, 0, bins - 1)

    accumulated = np.zeros((values.shape[2], bins), dtype=np.float64)
    for sample in values:
        for channel in range(values.shape[2]):
            field = scatter_to_grid(sample[:, channel], grid)
            if subtract_mean:
                field = field - float(np.mean(field))
            transformed = np.fft.fft2(field, norm="ortho")
            mode_power = np.abs(transformed) ** 2
            accumulated[channel] += np.bincount(
                bin_index,
                weights=mode_power[valid],
                minlength=bins,
            )[:bins]

    return centers, accumulated / values.shape[0]


def spectral_band_rows(
    generated_power: np.ndarray,
    target_power: np.ndarray,
    k_centers: np.ndarray,
    *,
    k_nyquist: float,
    channel_names: tuple[str, ...],
    bands: dict[str, tuple[float, float]],
    eps: float,
) -> list[dict[str, float | str]]:
    """Integrate generated/reference spectral power over normalized k bands."""

    generated = np.asarray(generated_power, dtype=np.float64)
    target = np.asarray(target_power, dtype=np.float64)
    centers = np.asarray(k_centers, dtype=np.float64)
    if generated.shape != target.shape:
        raise ValueError("generated and target spectra must have identical shapes")
    if generated.ndim != 2 or generated.shape[0] != len(channel_names):
        raise ValueError("spectrum arrays do not match channel semantics")
    if generated.shape[1] != centers.size:
        raise ValueError("spectrum arrays do not match k centers")
    if k_nyquist <= 0.0:
        raise ValueError("k_nyquist must be positive")

    fraction = centers / float(k_nyquist)
    rows: list[dict[str, float | str]] = []
    for channel, name in enumerate(channel_names):
        for band_name, (lower, upper) in bands.items():
            if lower < 0.0 or upper <= lower or upper > 1.0:
                raise ValueError(
                    f"invalid normalized spectral band '{band_name}': ({lower}, {upper})"
                )
            mask = (fraction >= lower) & (fraction < upper)
            if upper == 1.0:
                mask = (fraction >= lower) & (fraction <= upper)
            generated_band = float(np.sum(generated[channel, mask]))
            target_band = float(np.sum(target[channel, mask]))
            ratio = generated_band / target_band if target_band > eps else float("nan")
            rows.append(
                {
                    "channel": name,
                    "band": band_name,
                    "k_fraction_min": float(lower),
                    "k_fraction_max": float(upper),
                    "generated_power": generated_band,
                    "target_power": target_band,
                    "generated_over_target": ratio,
                }
            )
    return rows


def _unique_axis_values(values: np.ndarray, tolerance: float) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    if sorted_values.size == 0:
        return sorted_values
    unique = [float(sorted_values[0])]
    for value in sorted_values[1:]:
        scale = max(1.0, abs(unique[-1]), abs(float(value)))
        allowed = max(
            tolerance * scale,
            float(np.finfo(np.float32).eps) * scale,
        )
        if abs(float(value) - unique[-1]) > allowed:
            unique.append(float(value))
    return np.asarray(unique, dtype=np.float64)
