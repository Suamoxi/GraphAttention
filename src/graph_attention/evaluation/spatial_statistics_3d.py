"""Velocity-space spatial correlations and structure functions for periodic 3-D HIT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectra_3d import CartesianGrid3D, scatter_to_grid_3d


@dataclass(frozen=True)
class VelocitySpatialStatistics3D:
    """Per-snapshot spatial statistics on axis-aligned periodic separations."""

    lag: np.ndarray
    r_over_box: np.ndarray
    longitudinal_correlation: np.ndarray
    transverse_correlation: np.ndarray
    vector_correlation: np.ndarray
    longitudinal_s2: np.ndarray
    longitudinal_s3: np.ndarray
    longitudinal_s4: np.ndarray
    vector_s2: np.ndarray
    longitudinal_flatness: np.ndarray


def sample_velocity_spatial_statistics_3d(
    samples: np.ndarray,
    grid: CartesianGrid3D,
    channel_names: tuple[str, ...],
    *,
    max_lag: int | None = None,
) -> VelocitySpatialStatistics3D:
    """Compute periodic velocity correlations and structure functions.

    For each integer separation, the separation vector is taken along each of
    the three Cartesian axes in turn and the resulting statistics are averaged.
    This is an efficient isotropic-HIT diagnostic rather than a full all-direction
    radial pair enumeration.

    Longitudinal increments use the velocity component parallel to the chosen
    separation axis. vector_s2 uses the full velocity-increment magnitude.
    """

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected samples with shape [S, N, C], got {values.shape}")
    if values.shape[2] != len(channel_names):
        raise ValueError("channel_names do not match sample channels")
    expected_nodes = int(np.prod(grid.source_shape))
    if values.shape[1] != expected_nodes:
        raise ValueError(f"expected {expected_nodes} nodes per sample, got {values.shape[1]}")

    spacing = np.asarray(grid.spacing, dtype=np.float64)
    if not np.allclose(spacing, spacing[0], rtol=1.0e-6, atol=1.0e-12):
        raise ValueError(
            "axis-averaged HIT spatial statistics require equal Cartesian spacing"
        )
    if len(set(grid.shape)) != 1:
        raise ValueError(
            "axis-averaged HIT spatial statistics currently require a cubic FFT grid"
        )

    n = int(grid.shape[0])
    maximum = n // 2 if max_lag is None else int(max_lag)
    if maximum < 1 or maximum > n // 2:
        raise ValueError(f"max_lag must lie in [1, {n // 2}]")

    velocity = _velocity_samples(values, grid, channel_names)
    sample_count = velocity.shape[0]
    lag = np.arange(maximum + 1, dtype=np.int64)
    r_over_box = lag.astype(np.float64) / float(n)
    shape = (sample_count, maximum + 1)

    longitudinal_corr = np.empty(shape, dtype=np.float64)
    transverse_corr = np.empty(shape, dtype=np.float64)
    vector_corr = np.empty(shape, dtype=np.float64)
    longitudinal_s2 = np.empty(shape, dtype=np.float64)
    longitudinal_s3 = np.empty(shape, dtype=np.float64)
    longitudinal_s4 = np.empty(shape, dtype=np.float64)
    vector_s2 = np.empty(shape, dtype=np.float64)

    for sample_index in range(sample_count):
        field = velocity[sample_index]
        fluctuation = field - np.mean(field, axis=(0, 1, 2), keepdims=True)
        component_variance = np.mean(fluctuation**2, axis=(0, 1, 2))
        total_variance = float(np.sum(component_variance))
        if np.any(component_variance <= 0.0) or total_variance <= 0.0:
            raise ValueError(
                "velocity spatial statistics require non-zero variance in all components"
            )

        longitudinal_corr[sample_index, 0] = 1.0
        transverse_corr[sample_index, 0] = 1.0
        vector_corr[sample_index, 0] = 1.0
        longitudinal_s2[sample_index, 0] = 0.0
        longitudinal_s3[sample_index, 0] = 0.0
        longitudinal_s4[sample_index, 0] = 0.0
        vector_s2[sample_index, 0] = 0.0

        for separation in range(1, maximum + 1):
            long_corr_parts: list[float] = []
            transverse_corr_parts: list[float] = []
            vector_corr_parts: list[float] = []
            s2_long_parts: list[float] = []
            s3_long_parts: list[float] = []
            s4_long_parts: list[float] = []
            s2_vector_parts: list[float] = []

            for axis in range(3):
                shifted = np.roll(fluctuation, -separation, axis=axis)
                increment = shifted - fluctuation
                longitudinal_increment = increment[..., axis]

                long_corr_parts.append(
                    float(
                        np.mean(fluctuation[..., axis] * shifted[..., axis])
                        / component_variance[axis]
                    )
                )
                for component in range(3):
                    if component == axis:
                        continue
                    transverse_corr_parts.append(
                        float(
                            np.mean(
                                fluctuation[..., component] * shifted[..., component]
                            )
                            / component_variance[component]
                        )
                    )
                vector_corr_parts.append(
                    float(
                        np.mean(np.sum(fluctuation * shifted, axis=-1))
                        / total_variance
                    )
                )

                s2_long_parts.append(float(np.mean(longitudinal_increment**2)))
                s3_long_parts.append(float(np.mean(longitudinal_increment**3)))
                s4_long_parts.append(float(np.mean(longitudinal_increment**4)))
                s2_vector_parts.append(
                    float(np.mean(np.sum(increment**2, axis=-1)))
                )

            longitudinal_corr[sample_index, separation] = float(
                np.mean(long_corr_parts)
            )
            transverse_corr[sample_index, separation] = float(
                np.mean(transverse_corr_parts)
            )
            vector_corr[sample_index, separation] = float(np.mean(vector_corr_parts))
            longitudinal_s2[sample_index, separation] = float(np.mean(s2_long_parts))
            longitudinal_s3[sample_index, separation] = float(np.mean(s3_long_parts))
            longitudinal_s4[sample_index, separation] = float(np.mean(s4_long_parts))
            vector_s2[sample_index, separation] = float(np.mean(s2_vector_parts))

    flatness = np.divide(
        longitudinal_s4,
        longitudinal_s2**2,
        out=np.full_like(longitudinal_s4, np.nan),
        where=longitudinal_s2 > 0.0,
    )
    flatness[:, 0] = np.nan

    return VelocitySpatialStatistics3D(
        lag=lag,
        r_over_box=r_over_box,
        longitudinal_correlation=longitudinal_corr,
        transverse_correlation=transverse_corr,
        vector_correlation=vector_corr,
        longitudinal_s2=longitudinal_s2,
        longitudinal_s3=longitudinal_s3,
        longitudinal_s4=longitudinal_s4,
        vector_s2=vector_s2,
        longitudinal_flatness=flatness,
    )


def _velocity_samples(
    values: np.ndarray,
    grid: CartesianGrid3D,
    channel_names: tuple[str, ...],
) -> np.ndarray:
    base_to_index = {
        name.split(".", maxsplit=1)[0]: index for index, name in enumerate(channel_names)
    }
    required = ("rho", "rhou", "rhov", "rhow")
    missing = [name for name in required if name not in base_to_index]
    if missing:
        raise ValueError(
            f"velocity statistics require conservative channels {required}; missing {missing}"
        )

    rho = values[..., base_to_index["rho"]]
    if not np.isfinite(rho).all() or np.any(rho <= 0.0):
        raise ValueError("velocity statistics require finite strictly positive density")

    velocity_flat = np.stack(
        (
            values[..., base_to_index["rhou"]] / rho,
            values[..., base_to_index["rhov"]] / rho,
            values[..., base_to_index["rhow"]] / rho,
        ),
        axis=-1,
    )
    result = np.empty(
        (values.shape[0], *grid.shape, 3),
        dtype=np.float64,
    )
    for sample_index in range(values.shape[0]):
        for component in range(3):
            result[sample_index, ..., component] = scatter_to_grid_3d(
                velocity_flat[sample_index, :, component],
                grid,
            )
    return result
