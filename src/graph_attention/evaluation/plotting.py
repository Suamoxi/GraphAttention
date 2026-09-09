"""Plotting helpers for the generative distribution benchmark."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .spectra import CartesianGrid2D, scatter_to_grid


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
            label="test reference",
        )
        axis.hist(
            generated[..., channel].reshape(-1),
            bins=bins,
            density=True,
            histtype="step",
            label="generated",
        )
        axis.set_xlabel(name)
        axis.set_ylabel("density")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{_safe_name(name)}.png", dpi=dpi)
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
            label="test reference",
        )
        axes[0].loglog(
            physical_k,
            np.maximum(generated_channel, eps),
            label="generated",
        )
        axes[0].set_ylabel("radial shell power")
        axes[0].legend()

        axes[1].semilogx(physical_k, ratio)
        axes[1].axhline(1.0, linewidth=1.0, linestyle="--")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("generated / reference")
        axes[1].set_xlim(float(physical_k[0]), float(k_nyquist))

        secondary = axes[0].secondary_xaxis(
            "top",
            functions=(
                lambda value: value / k_nyquist,
                lambda value: value * k_nyquist,
            ),
        )
        secondary.set_xlabel(r"$k/k_{Nyq}$")
        figure.suptitle(name)
        figure.tight_layout()
        figure.savefig(output_dir / f"{_safe_name(name)}.png", dpi=dpi)
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
            )
            axes[0].set_title(f"generated\nkey={generated_ids[sample_index]}")
            reference_image = axes[1].imshow(
                reference_field,
                origin="lower",
                vmin=lower,
                vmax=upper,
            )
            axes[1].set_title(f"nearest test\n{reference_ids[reference_index]}")
            figure.colorbar(generated_image, ax=axes[0], fraction=0.046, pad=0.04)
            figure.colorbar(reference_image, ax=axes[1], fraction=0.046, pad=0.04)
            figure.suptitle(
                f"{name} | descriptor distance="
                f"{float(nearest_reference_distances[sample_index]):.4f}"
            )
            figure.tight_layout()
            figure.savefig(sample_dir / f"{_safe_name(name)}.png", dpi=dpi)
            plt.close(figure)


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
