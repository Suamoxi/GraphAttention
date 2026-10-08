"""Plotting helpers for full-volume 3-D HIT generation diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .spectra_3d import CartesianGrid3D, scatter_to_grid_3d
from .spatial_statistics_3d import VelocitySpatialStatistics3D

_LINEWIDTH = 3.0
_LABEL_FONTSIZE = 18
_TICK_FONTSIZE = 16
_LEGEND_FONTSIZE = 14
_TITLE_FONTSIZE = 18


def save_3d_snapshot_slice_examples(
    generated: np.ndarray,
    reference: np.ndarray,
    generated_ids: tuple[str, ...],
    reference_ids: tuple[str, ...],
    channel_names: tuple[str, ...],
    grid: CartesianGrid3D,
    output_dir: Path,
    *,
    num_examples: int,
    dpi: int,
) -> None:
    """Save unpaired generated/reference central slices for a few 3-D snapshots."""

    generated_values = np.asarray(generated, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    count = min(int(num_examples), generated_values.shape[0], reference_values.shape[0])
    if count < 1:
        return

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_index in range(count):
        sample_dir = output_dir / f"example_{sample_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for channel, raw_name in enumerate(channel_names):
            generated_grid = scatter_to_grid_3d(
                generated_values[sample_index, :, channel],
                grid,
            )
            reference_grid = scatter_to_grid_3d(
                reference_values[sample_index, :, channel],
                grid,
            )
            combined = np.concatenate(
                (generated_grid.reshape(-1), reference_grid.reshape(-1))
            )
            lower, upper = np.quantile(combined, [0.01, 0.99])
            if not upper > lower:
                lower = float(np.min(combined))
                upper = float(np.max(combined))
            if not upper > lower:
                lower -= 0.5
                upper += 0.5

            centers = tuple(size // 2 for size in grid.shape)
            reference_slices = (
                reference_grid[centers[0], :, :].T,
                reference_grid[:, centers[1], :].T,
                reference_grid[:, :, centers[2]].T,
            )
            generated_slices = (
                generated_grid[centers[0], :, :].T,
                generated_grid[:, centers[1], :].T,
                generated_grid[:, :, centers[2]].T,
            )

            figure, axes = plt.subplots(2, 3, figsize=(12.4, 7.4))
            plane_names = ("x-mid", "y-mid", "z-mid")
            image = None
            for column, plane_name in enumerate(plane_names):
                image = axes[0, column].imshow(
                    reference_slices[column],
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=lower,
                    vmax=upper,
                )
                axes[1, column].imshow(
                    generated_slices[column],
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=lower,
                    vmax=upper,
                )
                axes[0, column].set_title(f"Test reference | {plane_name}")
                axes[1, column].set_title(f"Generated | {plane_name}")
                axes[0, column].tick_params(labelsize=_TICK_FONTSIZE)
                axes[1, column].tick_params(labelsize=_TICK_FONTSIZE)

            axes[0, 0].set_ylabel(reference_ids[sample_index], fontsize=11)
            axes[1, 0].set_ylabel(generated_ids[sample_index], fontsize=11)
            if image is not None:
                colorbar = figure.colorbar(
                    image,
                    ax=axes.ravel().tolist(),
                    fraction=0.025,
                    pad=0.02,
                )
                colorbar.set_label(raw_name, fontsize=_LABEL_FONTSIZE)
                colorbar.ax.tick_params(labelsize=_TICK_FONTSIZE)
            figure.suptitle(
                f"{raw_name} | unpaired examples",
                fontsize=_TITLE_FONTSIZE,
            )
            figure.subplots_adjust(top=0.90, right=0.90, wspace=0.25, hspace=0.30)
            safe_name = raw_name.replace(".", "_").replace("/", "_")
            figure.savefig(
                sample_dir / f"{safe_name}.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(figure)


def save_3d_orthogonal_plane_comparisons(
    populations: dict[str, np.ndarray],
    channel_names: tuple[str, ...],
    grid: CartesianGrid3D,
    output_dir: Path,
    *,
    num_examples: int,
    dpi: int,
    cmap: str = "RdBu_r",
) -> None:
    """Render three intersecting physical planes in 3-D for each CFD variable.

    Each image compares the same sample index from all populations, with one
    robust (1-99%) color normalization and one colorbar across every panel.
    A shared index is a visualization convention, not a physical pairing of
    unconditionally generated HIT volumes and test reference snapshots.
    """

    if not populations:
        raise ValueError("3-D cutaway plots require at least one population")
    if num_examples < 0:
        raise ValueError("num_examples must be non-negative")
    expected_nodes = int(np.prod(grid.source_shape))
    prepared: dict[str, np.ndarray] = {}
    for label, values in populations.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 3 or array.shape[1:] != (expected_nodes, len(channel_names)):
            raise ValueError(
                f"3-D population {label!r} must have shape "
                f"[samples, {expected_nodes}, {len(channel_names)}], got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"3-D population {label!r} contains nonfinite values")
        prepared[label] = array

    count = min(num_examples, *(values.shape[0] for values in prepared.values()))
    if count < 1:
        return

    plt = _pyplot()
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    color_map = plt.get_cmap(cmap)
    output_dir.mkdir(parents=True, exist_ok=True)
    plane_indices = tuple(size // 2 for size in grid.shape)

    # Use normalized Cartesian coordinates: the endpoint-inclusive source mesh
    # has its repeated periodic maximum planes dropped by scatter_to_grid_3d.
    coords = tuple(
        np.arange(size, dtype=np.float64) / size
        for size in grid.shape
    )
    x, y, z = coords
    yy, zz = np.meshgrid(y, z, indexing="ij")
    xx_y, zz_y = np.meshgrid(x, z, indexing="ij")
    xx_z, yy_z = np.meshgrid(x, y, indexing="ij")

    for sample_index in range(count):
        sample_dir = output_dir / f"example_{sample_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for channel, raw_name in enumerate(channel_names):
            scalar_grids = {
                label: scatter_to_grid_3d(values[sample_index, :, channel], grid)
                for label, values in prepared.items()
            }
            combined = np.concatenate([field.ravel() for field in scalar_grids.values()])
            lower, upper = (float(v) for v in np.quantile(combined, (0.01, 0.99)))
            if not upper > lower:
                lower = float(np.min(combined))
                upper = float(np.max(combined))
            if not upper > lower:
                lower -= 0.5
                upper += 0.5
            norm = Normalize(vmin=lower, vmax=upper, clip=True)

            figure = plt.figure(figsize=(6.6 * len(scalar_grids), 6.4))
            axes = []
            for index, (label, field) in enumerate(scalar_grids.items(), start=1):
                axis = figure.add_subplot(1, len(scalar_grids), index, projection="3d")
                axes.append(axis)

                # x = xmid, y = ymid and z = zmid. Coordinates and array
                # orientations are explicit, so each surface shows the correct
                # slice regardless of source mesh node ordering.
                surfaces = (
                    (
                        np.full_like(yy, x[plane_indices[0]]),
                        yy,
                        zz,
                        field[plane_indices[0], :, :],
                    ),
                    (
                        xx_y,
                        np.full_like(xx_y, y[plane_indices[1]]),
                        zz_y,
                        field[:, plane_indices[1], :],
                    ),
                    (
                        xx_z,
                        yy_z,
                        np.full_like(xx_z, z[plane_indices[2]]),
                        field[:, :, plane_indices[2]],
                    ),
                )
                for sx, sy, sz, values_on_plane in surfaces:
                    axis.plot_surface(
                        sx, sy, sz,
                        facecolors=color_map(norm(values_on_plane)),
                        rstride=1,
                        cstride=1,
                        shade=False,
                        alpha=0.88,
                        linewidth=0,
                        antialiased=False,
                    )

                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
                axis.set_zlim(0, 1)
                axis.set_box_aspect((1, 1, 1))
                axis.set_xlabel("x / L")
                axis.set_ylabel("y / L")
                axis.set_zlabel("z / L")
                axis.set_xticks((0, 0.5, 1))
                axis.set_yticks((0, 0.5, 1))
                axis.set_zticks((0, 0.5, 1))
                axis.tick_params(labelsize=10)
                axis.view_init(elev=26, azim=-58)
                axis.set_title(label, fontsize=_TITLE_FONTSIZE, pad=12)

            scalar_map = ScalarMappable(norm=norm, cmap=color_map)
            scalar_map.set_array([])
            colorbar = figure.colorbar(
                scalar_map, ax=axes, shrink=0.68, fraction=0.02, pad=0.05
            )
            colorbar.set_label(raw_name, fontsize=_LABEL_FONTSIZE)
            colorbar.ax.tick_params(labelsize=_TICK_FONTSIZE)
            figure.suptitle(
                f"{raw_name} | orthogonal 3-D cutaway | unpaired population",
                fontsize=_TITLE_FONTSIZE,
            )
            figure.subplots_adjust(left=0.01, right=0.90, bottom=0.02, top=0.91, wspace=0.02)
            safe_name = raw_name.replace(".", "_").replace("/", "_")
            figure.savefig(sample_dir / f"{safe_name}.png", dpi=dpi, bbox_inches="tight")
            plt.close(figure)


def save_velocity_spatial_correlation_plots(
    generated: VelocitySpatialStatistics3D,
    reference: VelocitySpatialStatistics3D,
    output_dir: Path,
    *,
    lower_quantile: float,
    upper_quantile: float,
    dpi: int,
) -> None:
    """Plot population velocity correlation functions versus normalized separation."""

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.testing.assert_allclose(generated.r_over_box, reference.r_over_box)
    x = generated.r_over_box

    metrics = (
        ("longitudinal_correlation", "Longitudinal velocity correlation", "R_LL(r)"),
        ("transverse_correlation", "Transverse velocity correlation", "R_NN(r)"),
        ("vector_correlation", "Vector velocity correlation", "R_uu(r)"),
    )
    for attribute, title, ylabel in metrics:
        gen = np.asarray(getattr(generated, attribute), dtype=np.float64)
        ref = np.asarray(getattr(reference, attribute), dtype=np.float64)
        figure, axis = plt.subplots(figsize=(7.4, 5.4))
        ref_line = axis.plot(
            x,
            np.mean(ref, axis=0),
            linewidth=_LINEWIDTH,
            label="Test reference mean",
        )[0]
        axis.fill_between(
            x,
            np.quantile(ref, lower_quantile, axis=0),
            np.quantile(ref, upper_quantile, axis=0),
            alpha=0.2,
            color=ref_line.get_color(),
        )
        gen_line = axis.plot(
            x,
            np.mean(gen, axis=0),
            linewidth=_LINEWIDTH,
            label="Generated mean",
        )[0]
        axis.fill_between(
            x,
            np.quantile(gen, lower_quantile, axis=0),
            np.quantile(gen, upper_quantile, axis=0),
            alpha=0.2,
            color=gen_line.get_color(),
        )
        axis.axhline(0.0, linewidth=1.2, linestyle="--")
        axis.set_xlabel("Separation r / L_box", fontsize=_LABEL_FONTSIZE)
        axis.set_ylabel(ylabel, fontsize=_LABEL_FONTSIZE)
        axis.set_title(title, fontsize=_TITLE_FONTSIZE)
        axis.tick_params(labelsize=_TICK_FONTSIZE)
        axis.legend(fontsize=_LEGEND_FONTSIZE)
        figure.tight_layout()
        figure.savefig(output_dir / f"{attribute}.png", dpi=dpi)
        plt.close(figure)


def save_velocity_structure_function_plots(
    generated: VelocitySpatialStatistics3D,
    reference: VelocitySpatialStatistics3D,
    output_dir: Path,
    *,
    lower_quantile: float,
    upper_quantile: float,
    dpi: int,
) -> None:
    """Plot second/third-order velocity structure functions and flatness."""

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.testing.assert_allclose(generated.r_over_box, reference.r_over_box)
    x = generated.r_over_box[1:]

    metrics = (
        ("longitudinal_s2", "S2 longitudinal", True),
        ("vector_s2", "S2 vector", True),
        ("longitudinal_s3", "S3 longitudinal", False),
        ("longitudinal_flatness", "Longitudinal flatness", False),
    )
    for attribute, ylabel, log_y in metrics:
        gen = np.asarray(getattr(generated, attribute), dtype=np.float64)[:, 1:]
        ref = np.asarray(getattr(reference, attribute), dtype=np.float64)[:, 1:]

        figure, axis = plt.subplots(figsize=(7.4, 5.4))
        ref_mean = np.nanmean(ref, axis=0)
        gen_mean = np.nanmean(gen, axis=0)
        ref_low = np.nanquantile(ref, lower_quantile, axis=0)
        ref_high = np.nanquantile(ref, upper_quantile, axis=0)
        gen_low = np.nanquantile(gen, lower_quantile, axis=0)
        gen_high = np.nanquantile(gen, upper_quantile, axis=0)

        if log_y:
            floor = np.finfo(np.float64).tiny
            ref_mean = np.maximum(ref_mean, floor)
            gen_mean = np.maximum(gen_mean, floor)
            ref_low = np.maximum(ref_low, floor)
            ref_high = np.maximum(ref_high, floor)
            gen_low = np.maximum(gen_low, floor)
            gen_high = np.maximum(gen_high, floor)
            ref_line = axis.loglog(
                x,
                ref_mean,
                linewidth=_LINEWIDTH,
                label="Test reference mean",
            )[0]
            gen_line = axis.loglog(
                x,
                gen_mean,
                linewidth=_LINEWIDTH,
                label="Generated mean",
            )[0]
        else:
            ref_line = axis.semilogx(
                x,
                ref_mean,
                linewidth=_LINEWIDTH,
                label="Test reference mean",
            )[0]
            gen_line = axis.semilogx(
                x,
                gen_mean,
                linewidth=_LINEWIDTH,
                label="Generated mean",
            )[0]

        axis.fill_between(
            x,
            ref_low,
            ref_high,
            alpha=0.2,
            color=ref_line.get_color(),
        )
        axis.fill_between(
            x,
            gen_low,
            gen_high,
            alpha=0.2,
            color=gen_line.get_color(),
        )
        if attribute == "longitudinal_s3":
            axis.axhline(0.0, linewidth=1.2, linestyle="--")
        axis.set_xlabel("Separation r / L_box", fontsize=_LABEL_FONTSIZE)
        axis.set_ylabel(ylabel, fontsize=_LABEL_FONTSIZE)
        axis.set_title("Velocity structure function", fontsize=_TITLE_FONTSIZE)
        axis.tick_params(labelsize=_TICK_FONTSIZE)
        axis.legend(fontsize=_LEGEND_FONTSIZE)
        figure.tight_layout()
        figure.savefig(output_dir / f"{attribute}.png", dpi=dpi)
        plt.close(figure)


def _pyplot():
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "benchmark plotting requires matplotlib; install the project dependencies"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
