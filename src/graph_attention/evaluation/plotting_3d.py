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
