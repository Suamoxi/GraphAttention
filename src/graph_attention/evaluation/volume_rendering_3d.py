"""CPU-based emission/absorption ray casting of complete 3-D CFD scalar fields.

All voxels intersected by each ray contribute color and opacity. No arbitrary
cross sections, surface thresholds, or extra plotting dependencies are needed.
The same transfer function and robust color limits apply to all compared models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .spectra_3d import CartesianGrid3D, scatter_to_grid_3d


def _camera_basis(
    azimuth_deg: float,
    elevation_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    camera = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )
    right = np.cross(np.array([0.0, 0.0, 1.0]), camera)
    right /= np.linalg.norm(right)
    up = np.cross(camera, right)
    return camera, right, up


def raycast_scalar_volume(
    field: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap: str = "RdBu_r",
    resolution: int = 220,
    steps: int = 84,
    azimuth: float = -58.0,
    elevation: float = 26.0,
    optical_depth: float = 2.0,
    extent: float = 0.93,
) -> np.ndarray:
    """Ray-march all interior voxels with trilinear interpolation.

    Orthographic parallel rays traverse the cube back-to-front. At every
    interior sample, trilinear scalar interpolation controls the chosen
    colormap and a nonzero opacity transfer function. This produces an actual
    *volume rendering* of the full field, not a 3-D view of planar slices.

    Returns an RGB float image [resolution, resolution, 3] on white.
    """

    from matplotlib import colormaps

    volume = np.asarray(field, dtype=np.float64)
    if volume.ndim != 3 or min(volume.shape) < 2:
        raise ValueError("field must be a 3-D scalar array with dimensions >= 2")
    if not np.isfinite(volume).all():
        raise ValueError("volume contains non-finite scalar values")
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
        raise ValueError("vmin/vmax must be finite with vmax > vmin")
    if resolution < 8 or steps < 2:
        raise ValueError("resolution must be >= 8 and steps must be >= 2")
    if optical_depth <= 0 or not np.isfinite(optical_depth):
        raise ValueError("optical_depth must be finite and positive")

    camera, right, up = _camera_basis(azimuth, elevation)
    u = np.linspace(-extent, extent, resolution, dtype=np.float64)
    v = np.linspace(extent, -extent, resolution, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    bases = uu.reshape(-1, 1) * right + vv.reshape(-1, 1) * up
    image = np.ones((bases.shape[0], 3), dtype=np.float64)
    color_map = colormaps[cmap]
    shape = np.asarray(volume.shape, dtype=np.int64)
    shape_float = shape.astype(np.float64)
    dt = 1.8 / float(steps - 1)

    for t in np.linspace(-0.9, 0.9, steps):
        xyz = bases + t * camera
        inside = np.all((xyz >= -0.5) & (xyz <= 0.5), axis=1)
        if not np.any(inside):
            continue

        positions = np.clip(
            (xyz[inside] + 0.5) * shape_float - 0.5,
            0.0,
            shape_float - 1.0,
        )
        low = np.minimum(np.floor(positions).astype(np.intp), shape - 2)
        fraction = positions - low
        ix, iy, iz = low[:, 0], low[:, 1], low[:, 2]
        fx, fy, fz = fraction[:, 0], fraction[:, 1], fraction[:, 2]

        # Linear interpolation in z, then y, then x. Every ray samples the
        # complete interior, including voxels away from all central planes.
        a00 = volume[ix, iy, iz] * (1.0 - fz) + volume[ix, iy, iz + 1] * fz
        a01 = volume[ix, iy + 1, iz] * (1.0 - fz) + volume[ix, iy + 1, iz + 1] * fz
        a10 = volume[ix + 1, iy, iz] * (1.0 - fz) + volume[ix + 1, iy, iz + 1] * fz
        a11 = volume[ix + 1, iy + 1, iz] * (1.0 - fz) + volume[ix + 1, iy + 1, iz + 1] * fz
        scalar = (
            ((1.0 - fy) * a00 + fy * a01) * (1.0 - fx)
            + ((1.0 - fy) * a10 + fy * a11) * fx
        )

        normalized = np.clip((scalar - vmin) / (vmax - vmin), 0.0, 1.0)
        rgba = color_map(normalized)
        # Nonzero density across the full domain. Larger scalar excursions
        # receive slightly more opacity; the transfer is shared by all models.
        density = 0.45 + 1.5 * np.abs(normalized - 0.5)
        alpha = -np.expm1(-optical_depth * density * dt)
        image[inside] = (
            image[inside] * (1.0 - alpha)[:, None]
            + rgba[:, :3] * alpha[:, None]
        )

    return image.reshape((resolution, resolution, 3))


def _draw_box_and_axes(
    axis: object,
    *,
    azimuth: float,
    elevation: float,
    extent: float,
) -> None:
    _, right, up = _camera_basis(azimuth, elevation)
    corners = np.array(
        [
            [x, y, z]
            for x in (-0.5, 0.5)
            for y in (-0.5, 0.5)
            for z in (-0.5, 0.5)
        ],
        dtype=np.float64,
    )
    projected = np.column_stack((corners @ right, corners @ up))
    for index in range(8):
        for axis_idx in range(3):
            other = index ^ (1 << axis_idx)
            if index < other:
                axis.plot(
                    [projected[index, 0], projected[other, 0]],
                    [projected[index, 1], projected[other, 1]],
                    color="#202020",
                    alpha=0.45,
                    linewidth=0.9,
                    zorder=10,
                )

    # An orientation triad makes the orthographic view interpretable.
    origin = np.array([-0.75 * extent, -0.70 * extent])
    for name, vector in (
        ("x", np.array([1.0, 0.0, 0.0])),
        ("y", np.array([0.0, 1.0, 0.0])),
        ("z", np.array([0.0, 0.0, 1.0])),
    ):
        displacement = 0.17 * np.array([vector @ right, vector @ up])
        axis.annotate(
            "",
            xy=origin + displacement,
            xytext=origin,
            arrowprops={"arrowstyle": "->", "color": "#222222", "lw": 1.4},
        )
        axis.text(
            *(origin + 1.18 * displacement),
            name,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )


def save_volume_population_comparison(
    populations: dict[str, np.ndarray],
    channel_names: tuple[str, ...],
    grid: CartesianGrid3D,
    output_dir: Path,
    *,
    num_examples: int = 3,
    dpi: int = 180,
    cmap: str = "RdBu_r",
    resolution: int = 220,
    steps: int = 84,
    azimuth: float = -58.0,
    elevation: float = 26.0,
    optical_depth: float = 2.0,
) -> None:
    """Export real 3-D volume renders with matched transfer functions.

    Limits are pooled across all compared populations and all rendered example
    indices for each CFD channel. Matching sample indices does not imply
    physical pairing between independently generated and reference snapshots.
    """

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps, colors, pyplot as plt

    if not populations:
        raise ValueError("no 3-D populations to render")
    if num_examples < 0:
        raise ValueError("num_examples must be nonnegative")
    expected_nodes = int(np.prod(grid.source_shape))
    checked: dict[str, np.ndarray] = {}
    for label, values in populations.items():
        array = np.asarray(values)
        if array.ndim != 3 or array.shape[1:] != (expected_nodes, len(channel_names)):
            raise ValueError(
                f"{label}: expected [samples, {expected_nodes}, {len(channel_names)}], "
                f"received {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{label}: contains nonfinite field values")
        checked[label] = array

    count = min(num_examples, *(arr.shape[0] for arr in checked.values()))
    if not count:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    extent = 0.93
    for channel, raw_name in enumerate(channel_names):
        # Fixed across comparison populations AND across selected examples.
        source_values = np.concatenate(
            [arr[:count, :, channel].ravel() for arr in checked.values()]
        )
        lower, upper = (float(v) for v in np.quantile(source_values, (0.01, 0.99)))
        if not upper > lower:
            lower, upper = float(np.min(source_values)), float(np.max(source_values))
        if not upper > lower:
            lower -= 0.5
            upper += 0.5
        norm = colors.Normalize(vmin=lower, vmax=upper, clip=True)

        for example in range(count):
            sample_dir = output_dir / f"example_{example:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            figure, axes = plt.subplots(
                1, len(checked), figsize=(5.8 * len(checked) + 0.8, 5.9),
                squeeze=False,
            )
            for axis, (label, arr) in zip(axes[0], checked.items(), strict=True):
                volume = scatter_to_grid_3d(arr[example, :, channel], grid)
                image = raycast_scalar_volume(
                    volume,
                    vmin=lower,
                    vmax=upper,
                    cmap=cmap,
                    resolution=resolution,
                    steps=steps,
                    azimuth=azimuth,
                    elevation=elevation,
                    optical_depth=optical_depth,
                    extent=extent,
                )
                axis.imshow(
                    image,
                    extent=(-extent, extent, -extent, extent),
                    origin="upper",
                    interpolation="nearest",
                )
                _draw_box_and_axes(
                    axis,
                    azimuth=azimuth,
                    elevation=elevation,
                    extent=extent,
                )
                axis.set_xlim(-extent, extent)
                axis.set_ylim(-extent, extent)
                axis.set_aspect("equal")
                axis.axis("off")
                axis.set_title(label, fontsize=17)

            colorbar = figure.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=colormaps[cmap]),
                ax=axes[0].tolist(),
                shrink=0.68,
                fraction=0.026,
                pad=0.02,
            )
            colorbar.set_label(raw_name, fontsize=15)
            figure.suptitle(
                f"{raw_name} | 3-D volume ray casting | unpaired populations",
                fontsize=16,
            )
            figure.subplots_adjust(left=0.015, right=0.90, bottom=0.03, top=0.89, wspace=0.01)
            safe_name = raw_name.replace(".", "_").replace("/", "_")
            figure.savefig(
                sample_dir / f"{safe_name}.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(figure)
