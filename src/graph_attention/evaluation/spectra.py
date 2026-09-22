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
            "coordinates do not form a complete Cartesian product: "
            f"shape={shape}, nodes={arr.shape[0]}"
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


def sample_radial_spectra(
    samples: np.ndarray,
    grid: CartesianGrid2D,
    *,
    num_k_bins: int,
    subtract_mean: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return radial shell power for every sample as ``[S, C, K]``."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected samples with shape [S, N, C], got {values.shape}")
    expected_nodes = grid.shape[0] * grid.shape[1]
    if values.shape[1] != expected_nodes:
        raise ValueError(f"expected {expected_nodes} nodes per sample, got {values.shape[1]}")
    bins = int(num_k_bins)
    if bins < 1:
        raise ValueError("num_k_bins must be positive")

    centers, valid, bin_index = _radial_bins(grid, bins)
    power = np.zeros((values.shape[0], values.shape[2], bins), dtype=np.float64)
    for sample_index, sample in enumerate(values):
        for channel in range(values.shape[2]):
            field = scatter_to_grid(sample[:, channel], grid)
            if subtract_mean:
                field = field - float(np.mean(field))
            transformed = np.fft.fft2(field, norm="ortho")
            mode_power = np.abs(transformed) ** 2
            power[sample_index, channel] = np.bincount(
                bin_index,
                weights=mode_power[valid],
                minlength=bins,
            )[:bins]
    return centers, power


def sample_velocity_energy_spectra(
    samples: np.ndarray,
    grid: CartesianGrid2D,
    channel_names: tuple[str, ...],
    *,
    num_k_bins: int,
    subtract_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one velocity-based kinetic-energy spectrum per snapshot.

    The conservative fields are converted to velocity first. With the
    orthonormal FFT and the explicit division by the number of grid points,
    shell sums have units of velocity squared and represent specific turbulent
    kinetic energy over the retained radial modes.
    """

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected samples with shape [S, N, C], got {values.shape}")
    expected_nodes = grid.shape[0] * grid.shape[1]
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
    centers, valid, bin_index = _radial_bins(grid, bins)
    spectrum = np.zeros((values.shape[0], bins), dtype=np.float64)
    num_grid_points = float(expected_nodes)

    for sample_index in range(values.shape[0]):
        mode_energy = np.zeros(grid.shape, dtype=np.float64)
        for component in range(3):
            field = scatter_to_grid(velocity[sample_index, :, component], grid)
            if subtract_mean:
                field = field - float(np.mean(field))
            transformed = np.fft.fft2(field, norm="ortho")
            mode_energy += 0.5 * np.abs(transformed) ** 2 / num_grid_points
        spectrum[sample_index] = np.bincount(
            bin_index,
            weights=mode_energy[valid],
            minlength=bins,
        )[:bins]

    return centers, spectrum


def energy_spectrum_population_rows(
    generated_sample_energy: np.ndarray,
    reference_sample_energy: np.ndarray,
    k_centers: np.ndarray,
    *,
    k_nyquist: float,
    eps: float,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> list[dict[str, float]]:
    """Summarize per-snapshot energy spectra with mean, median, and quantile bands."""

    generated = np.asarray(generated_sample_energy, dtype=np.float64)
    reference = np.asarray(reference_sample_energy, dtype=np.float64)
    centers = np.asarray(k_centers, dtype=np.float64)
    if generated.ndim != 2 or reference.ndim != 2:
        raise ValueError("energy spectra must have shape [S, K]")
    if generated.shape[1] != centers.size or reference.shape[1] != centers.size:
        raise ValueError("energy spectra do not match k centers")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("energy-spectrum quantiles must satisfy 0 <= lower < upper <= 1")
    if k_nyquist <= 0.0:
        raise ValueError("k_nyquist must be positive")

    generated_mean = np.mean(generated, axis=0)
    reference_mean = np.mean(reference, axis=0)
    generated_median = np.median(generated, axis=0)
    reference_median = np.median(reference, axis=0)
    generated_low = np.quantile(generated, lower_quantile, axis=0)
    generated_high = np.quantile(generated, upper_quantile, axis=0)
    reference_low = np.quantile(reference, lower_quantile, axis=0)
    reference_high = np.quantile(reference, upper_quantile, axis=0)

    rows: list[dict[str, float]] = []
    for index, k_value in enumerate(centers):
        rows.append(
            {
                "k": float(k_value),
                "k_over_k_nyquist": float(k_value / k_nyquist),
                "generated_mean": float(generated_mean[index]),
                "reference_mean": float(reference_mean[index]),
                "generated_mean_over_reference": (
                    float(generated_mean[index] / reference_mean[index])
                    if reference_mean[index] > eps
                    else float("nan")
                ),
                "generated_median": float(generated_median[index]),
                "reference_median": float(reference_median[index]),
                "generated_median_over_reference": (
                    float(generated_median[index] / reference_median[index])
                    if reference_median[index] > eps
                    else float("nan")
                ),
                "generated_q10": float(generated_low[index]),
                "generated_q90": float(generated_high[index]),
                "reference_q10": float(reference_low[index]),
                "reference_q90": float(reference_high[index]),
            }
        )
    return rows


def energy_spectrum_sample_rows(
    generated_sample_energy: np.ndarray,
    reference_sample_energy: np.ndarray,
    k_centers: np.ndarray,
    generated_ids: tuple[str, ...],
    reference_ids: tuple[str, ...],
    *,
    k_nyquist: float,
) -> list[dict[str, float | int | str]]:
    """Serialize every snapshot energy spectrum for generated and test populations."""

    generated = np.asarray(generated_sample_energy, dtype=np.float64)
    reference = np.asarray(reference_sample_energy, dtype=np.float64)
    centers = np.asarray(k_centers, dtype=np.float64)
    if generated.shape != (len(generated_ids), centers.size):
        raise ValueError("generated energy spectra do not match generated IDs or k centers")
    if reference.shape != (len(reference_ids), centers.size):
        raise ValueError("reference energy spectra do not match reference IDs or k centers")

    rows: list[dict[str, float | int | str]] = []
    for population, identifiers, values, id_role in (
        ("generated", generated_ids, generated, "sampling_key"),
        ("test_reference", reference_ids, reference, "test_sample_id"),
    ):
        for sample_index, identifier in enumerate(identifiers):
            for k_index, k_value in enumerate(centers):
                rows.append(
                    {
                        "population": population,
                        "sample_index": sample_index,
                        "sample_id": identifier,
                        "sample_id_role": id_role,
                        "k": float(k_value),
                        "k_over_k_nyquist": float(k_value / k_nyquist),
                        "energy": float(values[sample_index, k_index]),
                    }
                )
    return rows


def energy_spectral_band_rows(
    generated_sample_energy: np.ndarray,
    reference_sample_energy: np.ndarray,
    k_centers: np.ndarray,
    *,
    k_nyquist: float,
    bands: dict[str, tuple[float, float]],
    eps: float,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> list[dict[str, float | str]]:
    """Summarize per-snapshot integrated energy in normalized wavenumber bands."""

    generated = np.asarray(generated_sample_energy, dtype=np.float64)
    reference = np.asarray(reference_sample_energy, dtype=np.float64)
    centers = np.asarray(k_centers, dtype=np.float64)
    if generated.ndim != 2 or reference.ndim != 2:
        raise ValueError("energy spectra must have shape [S, K]")
    if generated.shape[1] != centers.size or reference.shape[1] != centers.size:
        raise ValueError("energy spectra do not match k centers")

    fraction = centers / float(k_nyquist)
    rows: list[dict[str, float | str]] = []
    for band_name, (lower, upper) in bands.items():
        mask = (fraction >= lower) & (fraction < upper)
        if upper == 1.0:
            mask = (fraction >= lower) & (fraction <= upper)
        generated_band = np.sum(generated[:, mask], axis=1)
        reference_band = np.sum(reference[:, mask], axis=1)

        generated_mean = float(np.mean(generated_band))
        reference_mean = float(np.mean(reference_band))
        generated_median = float(np.median(generated_band))
        reference_median = float(np.median(reference_band))
        rows.append(
            {
                "band": str(band_name),
                "k_fraction_min": float(lower),
                "k_fraction_max": float(upper),
                "generated_mean": generated_mean,
                "reference_mean": reference_mean,
                "generated_mean_over_reference": (
                    generated_mean / reference_mean if reference_mean > eps else float("nan")
                ),
                "generated_median": generated_median,
                "reference_median": reference_median,
                "generated_median_over_reference": (
                    generated_median / reference_median
                    if reference_median > eps
                    else float("nan")
                ),
                "generated_q10": float(np.quantile(generated_band, lower_quantile)),
                "generated_q90": float(np.quantile(generated_band, upper_quantile)),
                "reference_q10": float(np.quantile(reference_band, lower_quantile)),
                "reference_q90": float(np.quantile(reference_band, upper_quantile)),
            }
        )
    return rows


def population_radial_spectra(
    samples: np.ndarray,
    grid: CartesianGrid2D,
    *,
    num_k_bins: int,
    subtract_mean: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean radial shell power for samples shaped [S, N, C]."""

    centers, sample_power = sample_radial_spectra(
        samples,
        grid,
        num_k_bins=num_k_bins,
        subtract_mean=subtract_mean,
    )
    return centers, np.mean(sample_power, axis=0)


def spectral_band_rows(
    generated_power: np.ndarray,
    reference_power: np.ndarray,
    k_centers: np.ndarray,
    *,
    k_nyquist: float,
    channel_names: tuple[str, ...],
    bands: dict[str, tuple[float, float]],
    eps: float,
) -> list[dict[str, float | str]]:
    """Integrate generated/reference spectral power over normalized k bands."""

    generated = np.asarray(generated_power, dtype=np.float64)
    reference = np.asarray(reference_power, dtype=np.float64)
    centers = np.asarray(k_centers, dtype=np.float64)
    if generated.shape != reference.shape:
        raise ValueError("generated and reference spectra must have identical shapes")
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
            reference_band = float(np.sum(reference[channel, mask]))
            ratio = generated_band / reference_band if reference_band > eps else float("nan")
            rows.append(
                {
                    "channel": name,
                    "band": band_name,
                    "k_fraction_min": float(lower),
                    "k_fraction_max": float(upper),
                    "generated_power": generated_band,
                    "reference_power": reference_band,
                    "generated_over_reference": ratio,
                }
            )
    return rows


def _radial_bins(
    grid: CartesianGrid2D,
    num_k_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = grid.shape
    dx, dy = grid.spacing
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kmag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    edges = np.linspace(0.0, grid.k_nyquist_min, num_k_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = (kmag > 0.0) & (kmag <= grid.k_nyquist_min)
    raw_bin = np.searchsorted(edges, kmag[valid], side="right") - 1
    bin_index = np.clip(raw_bin, 0, num_k_bins - 1)
    return centers, valid, bin_index


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
