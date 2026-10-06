"""Structured Cartesian 3-D spectral utilities for full-volume HIT benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartesianGrid3D:
    """Structured-grid metadata inferred from flat 3-D node coordinates.

    source_shape describes the stored mesh. shape describes the unique grid
    used for FFT analysis. For endpoint-inclusive periodic meshes such as the
    33^3 HIT mesh spanning a 32^3-cell periodic box, shape is 32^3 and the
    maximum-coordinate plane on every axis is excluded from the FFT.
    """

    source_shape: tuple[int, int, int]
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    linear_indices: np.ndarray
    k_nyquist_min: float
    periodic_endpoint_mode: str


def infer_cartesian_grid_3d(
    coords: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
    periodic_endpoint_mode: str = "none",
) -> CartesianGrid3D:
    """Infer a complete uniform Cartesian 3-D grid from flat coordinates."""

    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected coords with shape [N, 3], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Cartesian coordinates contain NaN or Inf")

    mode = str(periodic_endpoint_mode)
    if mode not in {"none", "drop_max"}:
        raise ValueError("periodic_endpoint_mode must be 'none' or 'drop_max'")

    tol = max(float(tolerance), 0.0)
    axes = [_unique_axis_values(arr[:, axis], tol) for axis in range(3)]
    source_shape = tuple(int(axis.size) for axis in axes)
    if any(size < 2 for size in source_shape):
        raise ValueError(
            f"Cartesian 3-D grid requires at least two points per axis, got {source_shape}"
        )
    if int(np.prod(source_shape)) != arr.shape[0]:
        raise ValueError(
            "coordinates do not form a complete Cartesian product: "
            f"shape={source_shape}, nodes={arr.shape[0]}"
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
            (
                np.clip(positions - 1, 0, axis_values.size - 1),
                np.clip(positions, 0, axis_values.size - 1),
            ),
            axis=1,
        )
        deltas = np.abs(axis_values[candidates] - arr[:, axis, None])
        indices = candidates[np.arange(arr.shape[0]), np.argmin(deltas, axis=1)]
        scale = np.maximum.reduce(
            (np.ones(arr.shape[0]), np.abs(axis_values[indices]), np.abs(arr[:, axis]))
        )
        allowed = np.maximum(
            tol * scale,
            float(np.finfo(np.float32).eps) * scale,
        )
        if np.any(np.abs(axis_values[indices] - arr[:, axis]) > allowed):
            raise ValueError("coordinates cannot be mapped to the inferred Cartesian grid")
        axis_indices.append(indices.astype(np.int64))

    linear_indices = np.ravel_multi_index(
        tuple(axis_indices),
        dims=source_shape,
        order="C",
    ).astype(np.int64)
    if np.unique(linear_indices).size != arr.shape[0]:
        raise ValueError("multiple nodes map to the same Cartesian grid point")

    if mode == "drop_max":
        if any(size < 3 for size in source_shape):
            raise ValueError(
                "drop_max periodic endpoint handling requires at least three stored points per axis"
            )
        shape = tuple(size - 1 for size in source_shape)
    else:
        shape = source_shape

    dx, dy, dz = spacing
    return CartesianGrid3D(
        source_shape=source_shape,
        shape=shape,
        spacing=(dx, dy, dz),
        linear_indices=linear_indices,
        k_nyquist_min=float(min(np.pi / dx, np.pi / dy, np.pi / dz)),
        periodic_endpoint_mode=mode,
    )


def scatter_to_grid_3d(values: np.ndarray, grid: CartesianGrid3D) -> np.ndarray:
    """Scatter one flat source-mesh field into the Cartesian FFT grid."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    expected = int(np.prod(grid.source_shape))
    if flat.size != expected:
        raise ValueError(f"expected {expected} source node values, got {flat.size}")

    full = np.empty(expected, dtype=np.float64)
    full[grid.linear_indices] = flat
    full = full.reshape(grid.source_shape, order="C")
    if grid.periodic_endpoint_mode == "drop_max":
        return full[:-1, :-1, :-1]
    return full


def sample_radial_spectra_3d(
    samples: np.ndarray,
    grid: CartesianGrid3D,
    *,
    num_k_bins: int,
    subtract_mean: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 3-D radial shell power for every sample as [S, C, K]."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected samples with shape [S, N, C], got {values.shape}")
    expected_nodes = int(np.prod(grid.source_shape))
    if values.shape[1] != expected_nodes:
        raise ValueError(f"expected {expected_nodes} nodes per sample, got {values.shape[1]}")
    bins = int(num_k_bins)
    if bins < 1:
        raise ValueError("num_k_bins must be positive")

    centers, valid, bin_index, active_bins = _radial_bins_3d(grid, bins)
    power = np.zeros((values.shape[0], values.shape[2], bins), dtype=np.float64)
    for sample_index, sample in enumerate(values):
        for channel in range(values.shape[2]):
            field = scatter_to_grid_3d(sample[:, channel], grid)
            if subtract_mean:
                field = field - float(np.mean(field))
            transformed = np.fft.fftn(field, norm="ortho")
            mode_power = np.abs(transformed) ** 2
            power[sample_index, channel] = np.bincount(
                bin_index,
                weights=mode_power[valid],
                minlength=bins,
            )[:bins]
    return centers[active_bins], power[..., active_bins]


def sample_velocity_energy_spectra_3d(
    samples: np.ndarray,
    grid: CartesianGrid3D,
    channel_names: tuple[str, ...],
    *,
    num_k_bins: int,
    subtract_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one 3-D velocity-based specific kinetic-energy spectrum per sample."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected samples with shape [S, N, C], got {values.shape}")
    expected_nodes = int(np.prod(grid.source_shape))
    if values.shape[1] != expected_nodes:
        raise ValueError(f"expected {expected_nodes} nodes per sample, got {values.shape[1]}")
    if values.shape[2] != len(channel_names):
        raise ValueError("channel_names do not match sample channels")

    base_to_index = {
        name.split(".", maxsplit=1)[0]: index for index, name in enumerate(channel_names)
    }
    required = ("rho", "rhou", "rhov", "rhow")
    missing = [name for name in required if name not in base_to_index]
    if missing:
        raise ValueError(
            f"velocity energy spectrum requires conservative channels {required}; missing {missing}"
        )

    rho = values[..., base_to_index["rho"]]
    if not np.isfinite(rho).all() or np.any(rho <= 0.0):
        raise ValueError("velocity energy spectrum requires finite strictly positive density")

    velocity = np.stack(
        (
            values[..., base_to_index["rhou"]] / rho,
            values[..., base_to_index["rhov"]] / rho,
            values[..., base_to_index["rhow"]] / rho,
        ),
        axis=-1,
    )
    if not np.isfinite(velocity).all():
        raise ValueError("velocity energy spectrum produced non-finite velocity values")

    bins = int(num_k_bins)
    if bins < 1:
        raise ValueError("num_k_bins must be positive")
    centers, valid, bin_index, active_bins = _radial_bins_3d(grid, bins)
    spectrum = np.zeros((values.shape[0], bins), dtype=np.float64)
    num_grid_points = float(np.prod(grid.shape))
    delta_k = float(grid.k_nyquist_min) / float(bins)

    for sample_index in range(values.shape[0]):
        mode_energy = np.zeros(grid.shape, dtype=np.float64)
        for component in range(3):
            field = scatter_to_grid_3d(velocity[sample_index, :, component], grid)
            if subtract_mean:
                field = field - float(np.mean(field))
            transformed = np.fft.fftn(field, norm="ortho")
            mode_energy += 0.5 * np.abs(transformed) ** 2 / num_grid_points
        shell_energy = np.bincount(
            bin_index,
            weights=mode_energy[valid],
            minlength=bins,
        )[:bins]
        spectrum[sample_index] = shell_energy / delta_k

    return centers[active_bins], spectrum[:, active_bins]


def nearest_neighbor_spatial_metrics_3d(
    values: np.ndarray,
    grid: CartesianGrid3D,
) -> tuple[float, float]:
    """Return axis-neighbour correlation and first-difference RMS on the FFT grid."""

    field = scatter_to_grid_3d(values, grid)
    first_parts: list[np.ndarray] = []
    second_parts: list[np.ndarray] = []
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        first_parts.append(field[tuple(left)].reshape(-1))
        second_parts.append(field[tuple(right)].reshape(-1))

        if grid.periodic_endpoint_mode == "drop_max":
            last = [slice(None)] * 3
            first = [slice(None)] * 3
            last[axis] = -1
            first[axis] = 0
            first_parts.append(field[tuple(last)].reshape(-1))
            second_parts.append(field[tuple(first)].reshape(-1))

    first_values = np.concatenate(first_parts)
    second_values = np.concatenate(second_parts)
    first_centered = first_values - float(np.mean(first_values))
    second_centered = second_values - float(np.mean(second_values))
    denominator = float(
        np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    )
    correlation = (
        float(np.sum(first_centered * second_centered) / denominator)
        if denominator > 0.0
        else float("nan")
    )
    difference_rms = float(np.sqrt(np.mean((second_values - first_values) ** 2)))
    return correlation, difference_rms


def _radial_bins_3d(
    grid: CartesianGrid3D,
    num_k_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = grid.shape
    dx, dy, dz = grid.spacing
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")
    kmag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

    edges = np.linspace(0.0, grid.k_nyquist_min, num_k_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = (kmag > 0.0) & (kmag <= grid.k_nyquist_min)
    raw_bin = np.searchsorted(edges, kmag[valid], side="right") - 1
    bin_index = np.clip(raw_bin, 0, num_k_bins - 1)
    mode_count = np.bincount(bin_index, minlength=num_k_bins)[:num_k_bins]
    active_bins = mode_count > 0
    if not np.any(active_bins):
        raise ValueError("3-D radial spectrum contains no non-zero Fourier modes")
    return centers, valid, bin_index, active_bins


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
