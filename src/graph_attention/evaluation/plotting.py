"""Plotting helpers for the generative distribution benchmark."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .spectra import CartesianGrid2D, scatter_to_grid

_LINEWIDTH = 2.4
_LABEL_FONTSIZE = 14
_TICK_FONTSIZE = 12
_LEGEND_FONTSIZE = 11


def save_marginal_plots(
    generated: np.ndarray,
    reference: np.ndarray,
    channel_names: tuple[str, ...],
    output_dir: Path,
    *,
    bins: int,
    dpi: int,
) -> None:
    """Save pooled generated/reference marginal histograms."""

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    for channel, name in enumerate(channel_names):
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.hist(
            reference[..., channel].reshape(-1),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=_LINEWIDTH,
            label="test reference",
        )
        axis.hist(
            generated[..., channel].reshape(-1),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=_LINEWIDTH,
            label="generated",
        )
        axis.set_xlabel(name, fontsize=_LABEL_FONTSIZE)
        axis.set_ylabel("PDF", fontsize=_LABEL_FONTSIZE)
        _style_axis(axis)
        axis.legend(fontsize=_LEGEND_FONTSIZE)
        figure.tight_layout()
        figure.savefig(output_dir / f"{_safe_name(name)}.png", dpi=dpi)
        plt.close(figure)


def save_model_marginal_comparison(
    reference: np.ndarray,
    generated_by_model: dict[str, np.ndarray],
    channel_names: tuple[str, ...],
    output_path: Path,
    *,
    bins: int,
    dpi: int,
) -> None:
    """Save one multi-panel PDF comparison for several generative models."""

    if bins < 2:
        raise ValueError("bins must be at least 2")
    if not generated_by_model:
        raise ValueError("at least one generated model population is required")
    reference_values = np.asarray(reference, dtype=np.float64)
    if reference_values.ndim != 3:
        raise ValueError("reference must have shape [S, N, C]")
    if reference_values.shape[2] != len(channel_names):
        raise ValueError("channel_names do not match the reference channels")
    if not np.isfinite(reference_values).all():
        raise ValueError("reference values must be finite")

    populations: dict[str, np.ndarray] = {}
    for label, values in generated_by_model.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != reference_values.shape:
            raise ValueError(
                f"generated population '{label}' has shape {array.shape}; "
                f"expected {reference_values.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"generated population '{label}' contains non-finite values")
        populations[label] = array

    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channel_count = len(channel_names)
    column_count = min(3, max(channel_count, 1))
    row_count = math.ceil(channel_count / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.8 * column_count, 3.8 * row_count),
        squeeze=False,
    )
    flat_axes = axes.reshape(-1)

    for channel, name in enumerate(channel_names):
        axis = flat_axes[channel]
        arrays = [
            reference_values[..., channel].reshape(-1),
            *(values[..., channel].reshape(-1) for values in populations.values()),
        ]
        lower = min(float(np.min(values)) for values in arrays)
        upper = max(float(np.max(values)) for values in arrays)
        if not upper > lower:
            padding = 0.5 if lower == 0.0 else 0.05 * abs(lower)
            lower -= padding
            upper += padding

        edges = np.linspace(lower, upper, bins + 1, dtype=np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        reference_pdf, _ = np.histogram(arrays[0], bins=edges, density=True)
        axis.plot(
            centers,
            reference_pdf,
            linewidth=_LINEWIDTH + 0.4,
            color="black",
            label="Test reference",
        )
        for (label, _), values in zip(populations.items(), arrays[1:], strict=True):
            density, _ = np.histogram(values, bins=edges, density=True)
            axis.plot(centers, density, linewidth=_LINEWIDTH, label=label)

        axis.set_xlabel(name, fontsize=_LABEL_FONTSIZE)
        axis.set_ylabel("PDF", fontsize=_LABEL_FONTSIZE)
        _style_axis(axis)

    for axis in flat_axes[channel_count:]:
        axis.set_axis_off()

    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=_LEGEND_FONTSIZE,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def save_spectrum_plots(
    k_centers: np.ndarray,
    generated_power: np.ndarray,
    reference_power: np.ndarray,
    channel_names: tuple[str, ...],
    output_dir: Path,
    *,
    k_nyquist: float,
    eps: float,
    dpi: int,
) -> None:
    """Save spectra versus physical k with normalized k as a secondary coordinate."""

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    physical_k = np.asarray(k_centers, dtype=np.float64)
    for channel, name in enumerate(channel_names):
        reference_channel = np.asarray(reference_power[channel], dtype=np.float64)
        generated_channel = np.asarray(generated_power[channel], dtype=np.float64)
        ratio = np.divide(
            generated_channel,
            reference_channel,
            out=np.full_like(generated_channel, np.nan),
            where=reference_channel > eps,
        )

        figure, axes = plt.subplots(2, 1, figsize=(6.4, 7.0), sharex=True)
        axes[0].loglog(
            physical_k,
            np.maximum(reference_channel, eps),
            linewidth=_LINEWIDTH,
            label="test reference",
        )
        axes[0].loglog(
            physical_k,
            np.maximum(generated_channel, eps),
            linewidth=_LINEWIDTH,
            label="generated",
        )
        axes[0].set_ylabel("radial shell power", fontsize=_LABEL_FONTSIZE)
        _style_axis(axes[0])
        axes[0].legend(fontsize=_LEGEND_FONTSIZE)

        axes[1].semilogx(physical_k, ratio, linewidth=_LINEWIDTH)
        axes[1].axhline(1.0, linewidth=1.5, linestyle="--")
        axes[1].set_xlabel(r"Wavenumber $k$", fontsize=_LABEL_FONTSIZE)
        axes[1].set_ylabel("generated / reference", fontsize=_LABEL_FONTSIZE)
        axes[1].set_xlim(float(physical_k[0]), float(k_nyquist))
        _style_axis(axes[1])

        secondary = axes[0].secondary_xaxis(
            "top",
            functions=(
                lambda value: value / k_nyquist,
                lambda value: value * k_nyquist,
            ),
        )
        secondary.set_xlabel(r"$k/k_{Nyq}$", fontsize=_LABEL_FONTSIZE)
        secondary.tick_params(
            axis="x",
            which="both",
            labelsize=_TICK_FONTSIZE,
            width=1.4,
            length=5,
        )
        figure.suptitle(name, fontsize=_LABEL_FONTSIZE)
        figure.tight_layout()
        figure.savefig(output_dir / f"{_safe_name(name)}.png", dpi=dpi)
        plt.close(figure)


def save_energy_spectrum_population_plots(
    k_centers: np.ndarray,
    generated_sample_energy: np.ndarray,
    reference_sample_energy: np.ndarray,
    output_dir: Path,
    *,
    k_nyquist: float,
    eps: float,
    dpi: int,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> None:
    """Plot generated and test energy-spectrum populations on shared axes.

    Two figures are written: one using the population mean as the central
    curve and one using the population median. Both show the 10--90% snapshot
    interval for each population.
    """

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    physical_k = np.asarray(k_centers, dtype=np.float64)
    generated = np.asarray(generated_sample_energy, dtype=np.float64)
    reference = np.asarray(reference_sample_energy, dtype=np.float64)
    if generated.ndim != 2 or reference.ndim != 2:
        raise ValueError("energy spectra must have shape [S, K]")
    if generated.shape[1] != physical_k.size or reference.shape[1] != physical_k.size:
        raise ValueError("energy spectra do not match k centers")

    generated_low = np.quantile(generated, lower_quantile, axis=0)
    generated_high = np.quantile(generated, upper_quantile, axis=0)
    reference_low = np.quantile(reference, lower_quantile, axis=0)
    reference_high = np.quantile(reference, upper_quantile, axis=0)

    aggregations = {
        "mean": (np.mean(generated, axis=0), np.mean(reference, axis=0)),
        "median": (np.median(generated, axis=0), np.median(reference, axis=0)),
    }
    for aggregation, (generated_center, reference_center) in aggregations.items():
        figure, axis = plt.subplots(figsize=(6.8, 4.8))

        reference_line = axis.loglog(
            physical_k,
            np.maximum(reference_center, eps),
            linewidth=_LINEWIDTH,
            label=f"test reference {aggregation}",
        )[0]
        axis.fill_between(
            physical_k,
            np.maximum(reference_low, eps),
            np.maximum(reference_high, eps),
            alpha=0.2,
            color=reference_line.get_color(),
            label="test reference 10-90%",
        )

        generated_line = axis.loglog(
            physical_k,
            np.maximum(generated_center, eps),
            linewidth=_LINEWIDTH,
            label=f"generated {aggregation}",
        )[0]
        axis.fill_between(
            physical_k,
            np.maximum(generated_low, eps),
            np.maximum(generated_high, eps),
            alpha=0.2,
            color=generated_line.get_color(),
            label="generated 10-90%",
        )

        axis.set_xlabel(r"Wavenumber $k$", fontsize=_LABEL_FONTSIZE)
        axis.set_ylabel(r"$E_{2D}(k)$", fontsize=_LABEL_FONTSIZE)
        axis.set_xlim(float(physical_k[0]), float(k_nyquist))
        _style_axis(axis)
        axis.legend(fontsize=_LEGEND_FONTSIZE)

        secondary = axis.secondary_xaxis(
            "top",
            functions=(
                lambda value: value / k_nyquist,
                lambda value: value * k_nyquist,
            ),
        )
        secondary.set_xlabel(r"$k/k_{Nyq}$", fontsize=_LABEL_FONTSIZE)
        secondary.tick_params(
            axis="x",
            which="both",
            labelsize=_TICK_FONTSIZE,
            width=1.4,
            length=5,
        )
        axis.set_title(
            "Velocity-based 2-D kinetic-energy spectrum\n"
            f"{aggregation} with 10-90% snapshot interval",
            fontsize=_LABEL_FONTSIZE,
        )
        figure.tight_layout()
        figure.savefig(output_dir / f"energy_spectrum_{aggregation}.png", dpi=dpi)
        plt.close(figure)


def save_energy_spectrum_comparison(
    k_centers: np.ndarray,
    reference_energy: np.ndarray,
    generated_energy_by_model: dict[str, np.ndarray],
    output_path: Path,
    *,
    eps: float,
    dpi: int,
) -> None:
    """Save one mean energy-spectrum comparison in physical wavenumber space."""

    physical_k = np.asarray(k_centers, dtype=np.float64)
    reference = np.asarray(reference_energy, dtype=np.float64)
    if physical_k.ndim != 1 or reference.shape != physical_k.shape:
        raise ValueError("reference energy and k centers must be aligned 1-D arrays")
    if physical_k.size == 0 or np.any(np.diff(physical_k) <= 0.0):
        raise ValueError("k centers must be non-empty and strictly increasing")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if not generated_energy_by_model:
        raise ValueError("at least one generated model energy spectrum is required")

    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.loglog(
        physical_k,
        np.maximum(reference, eps),
        linewidth=_LINEWIDTH + 0.4,
        color="black",
        label="Test reference",
    )
    for label, values in generated_energy_by_model.items():
        energy = np.asarray(values, dtype=np.float64)
        if energy.shape != physical_k.shape:
            raise ValueError(
                f"energy spectrum '{label}' has shape {energy.shape}; expected {physical_k.shape}"
            )
        axis.loglog(
            physical_k,
            np.maximum(energy, eps),
            linewidth=_LINEWIDTH,
            label=label,
        )

    axis.set_xlabel(r"Wavenumber $k$", fontsize=_LABEL_FONTSIZE)
    axis.set_ylabel(r"$E_{2D}(k)$", fontsize=_LABEL_FONTSIZE)
    axis.set_xlim(float(physical_k[0]), float(physical_k[-1]))
    _style_axis(axis)
    axis.legend(fontsize=_LEGEND_FONTSIZE)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def save_nearest_reference_field_examples(
    generated: np.ndarray,
    reference: np.ndarray,
    generated_ids: tuple[str, ...],
    reference_ids: tuple[str, ...],
    nearest_reference_indices: np.ndarray,
    nearest_reference_distances: np.ndarray,
    channel_names: tuple[str, ...],
    grid: CartesianGrid2D,
    output_dir: Path,
    *,
    num_examples: int,
    max_channels: int,
    dpi: int,
) -> None:
    """Plot generated samples beside their descriptor-nearest real test snapshots."""

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(max(int(num_examples), 0), generated.shape[0])
    channel_count = min(max(int(max_channels), 0), generated.shape[2])
    for sample_index in range(sample_count):
        reference_index = int(nearest_reference_indices[sample_index])
        sample_dir = output_dir / f"generated_{sample_index:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for channel in range(channel_count):
            name = channel_names[channel]
            generated_field = scatter_to_grid(generated[sample_index, :, channel], grid)
            reference_field = scatter_to_grid(reference[reference_index, :, channel], grid)
            lower = float(min(np.min(reference_field), np.min(generated_field)))
            upper = float(max(np.max(reference_field), np.max(generated_field)))

            figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
            generated_image = axes[0].imshow(
                generated_field,
                origin="lower",
                vmin=lower,
                vmax=upper,
                cmap="RdBu_r",
            )
            axes[0].set_title(
                f"generated\nkey={generated_ids[sample_index]}",
                fontsize=_LABEL_FONTSIZE,
            )
            reference_image = axes[1].imshow(
                reference_field,
                origin="lower",
                vmin=lower,
                vmax=upper,
                cmap="RdBu_r",
            )
            axes[1].set_title(
                f"nearest test\n{reference_ids[reference_index]}",
                fontsize=_LABEL_FONTSIZE,
            )
            for axis in axes:
                _style_axis(axis)
            generated_colorbar = figure.colorbar(
                generated_image,
                ax=axes[0],
                fraction=0.046,
                pad=0.04,
            )
            reference_colorbar = figure.colorbar(
                reference_image,
                ax=axes[1],
                fraction=0.046,
                pad=0.04,
            )
            generated_colorbar.ax.tick_params(labelsize=_TICK_FONTSIZE, width=1.2)
            reference_colorbar.ax.tick_params(labelsize=_TICK_FONTSIZE, width=1.2)
            figure.suptitle(
                f"{name} | descriptor distance="
                f"{float(nearest_reference_distances[sample_index]):.4f}",
                fontsize=_LABEL_FONTSIZE,
            )
            figure.tight_layout()
            figure.savefig(sample_dir / f"{_safe_name(name)}.png", dpi=dpi)
            plt.close(figure)


def _style_axis(axis) -> None:
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=_TICK_FONTSIZE,
        width=1.4,
        length=5,
    )
    axis.tick_params(axis="both", which="minor", width=1.2, length=3)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"


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
