"""Plotting helpers for the generative distribution benchmark."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .spectra import CartesianGrid2D, scatter_to_grid


def save_marginal_plots(
    generated: np.ndarray,
    target: np.ndarray,
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
            target[..., channel].reshape(-1),
            bins=bins,
            density=True,
            histtype="step",
            label="target",
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
    target_power: np.ndarray,
    channel_names: tuple[str, ...],
    output_dir: Path,
    *,
    k_nyquist: float,
    eps: float,
    dpi: int,
) -> None:
    """Save mean radial spectra and generated/reference ratios."""

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_k = np.asarray(k_centers, dtype=np.float64) / float(k_nyquist)
    for channel, name in enumerate(channel_names):
        target_channel = np.asarray(target_power[channel], dtype=np.float64)
        generated_channel = np.asarray(generated_power[channel], dtype=np.float64)
        ratio = np.divide(
            generated_channel,
            target_channel,
            out=np.full_like(generated_channel, np.nan),
            where=target_channel > eps,
        )

        figure, axes = plt.subplots(2, 1, figsize=(6.4, 7.0), sharex=True)
        axes[0].loglog(
            normalized_k,
            np.maximum(target_channel, eps),
            label="target",
        )
        axes[0].loglog(
            normalized_k,
            np.maximum(generated_channel, eps),
            label="generated",
        )
        axes[0].set_ylabel("radial shell power")
        axes[0].legend()

        axes[1].plot(normalized_k, ratio)
        axes[1].axhline(1.0, linewidth=1.0, linestyle="--")
        axes[1].set_xlabel(r"$k/k_{Nyq}$")
        axes[1].set_ylabel("generated / target")
        axes[1].set_xlim(0.0, 1.0)
        figure.suptitle(name)
        figure.tight_layout()
        figure.savefig(output_dir / f"{_safe_name(name)}.png", dpi=dpi)
        plt.close(figure)


def save_field_examples(
    generated: np.ndarray,
    target: np.ndarray,
    sample_ids: tuple[str, ...],
    channel_names: tuple[str, ...],
    grid: CartesianGrid2D,
    output_dir: Path,
    *,
    num_examples: int,
    max_channels: int,
    dpi: int,
) -> None:
    """Save target/generated/difference panels for the first deterministic samples."""

    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(max(int(num_examples), 0), generated.shape[0])
    channel_count = min(max(int(max_channels), 0), generated.shape[2])
    for sample_index in range(sample_count):
        sample_dir = output_dir / _safe_name(sample_ids[sample_index])
        sample_dir.mkdir(parents=True, exist_ok=True)
        for channel in range(channel_count):
            name = channel_names[channel]
            target_field = scatter_to_grid(target[sample_index, :, channel], grid)
            generated_field = scatter_to_grid(generated[sample_index, :, channel], grid)
            difference = generated_field - target_field
            lower = float(min(np.min(target_field), np.min(generated_field)))
            upper = float(max(np.max(target_field), np.max(generated_field)))

            figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
            target_image = axes[0].imshow(target_field, origin="lower", vmin=lower, vmax=upper)
            axes[0].set_title("target")
            generated_image = axes[1].imshow(
                generated_field,
                origin="lower",
                vmin=lower,
                vmax=upper,
            )
            axes[1].set_title("generated")
            difference_image = axes[2].imshow(difference, origin="lower")
            axes[2].set_title("generated - target")
            figure.colorbar(target_image, ax=axes[0], fraction=0.046, pad=0.04)
            figure.colorbar(generated_image, ax=axes[1], fraction=0.046, pad=0.04)
            figure.colorbar(difference_image, ax=axes[2], fraction=0.046, pad=0.04)
            figure.suptitle(f"{sample_ids[sample_index]} | {name}")
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
