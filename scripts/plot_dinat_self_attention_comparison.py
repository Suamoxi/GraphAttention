"""Create direct M33/M39 generation-quality comparison plots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from graph_attention.evaluation.plotting import (
    save_energy_spectrum_comparison,
    save_model_marginal_comparison,
)


def _load_generation(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping generation artifact: {path}")

    generated = payload.get("generated_nondimensional")
    reference = payload.get("target_nondimensional")
    node_counts = tuple(int(value) for value in payload.get("node_counts", ()))
    channel_names = tuple(payload.get("channel_names", ()))
    if not isinstance(generated, torch.Tensor) or not isinstance(reference, torch.Tensor):
        raise TypeError(f"generation artifact requires tensor fields: {path}")
    if generated.shape != reference.shape or generated.ndim != 2:
        raise ValueError(
            f"generated/reference tensors must share shape [total_nodes, C]: {path}"
        )
    if not node_counts or len(set(node_counts)) != 1:
        raise ValueError(f"comparison requires one shared fixed-size mesh: {path}")
    if sum(node_counts) != generated.shape[0]:
        raise ValueError(f"node_counts do not match tensor rows: {path}")
    if len(channel_names) != generated.shape[1]:
        raise ValueError(f"channel_names do not match tensor columns: {path}")

    shape = (len(node_counts), node_counts[0], generated.shape[1])
    return (
        generated.to(torch.float64).numpy().reshape(shape),
        reference.to(torch.float64).numpy().reshape(shape),
        channel_names,
    )


def _load_energy_summary(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty energy-spectrum summary: {path}")

    return {
        "k": np.asarray([float(row["k"]) for row in rows], dtype=np.float64),
        "generated_mean": np.asarray(
            [float(row["generated_mean"]) for row in rows],
            dtype=np.float64,
        ),
        "reference_mean": np.asarray(
            [float(row["reference_mean"]) for row in rows],
            dtype=np.float64,
        ),
        "generated_median": np.asarray(
            [float(row["generated_median"]) for row in rows],
            dtype=np.float64,
        ),
        "reference_median": np.asarray(
            [float(row["reference_median"]) for row in rows],
            dtype=np.float64,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m33-generation", type=Path, required=True)
    parser.add_argument("--m39-generation", type=Path, required=True)
    parser.add_argument("--m33-energy-summary", type=Path, required=True)
    parser.add_argument("--m39-energy-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    m33_generated, reference, channel_names = _load_generation(args.m33_generation)
    m39_generated, m39_reference, m39_names = _load_generation(args.m39_generation)
    if m39_names != channel_names:
        raise ValueError("M33 and M39 channel names differ")
    np.testing.assert_allclose(m39_reference, reference, rtol=0.0, atol=0.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_model_marginal_comparison(
        reference,
        {
            "M33 no self": m33_generated,
            "M39 self": m39_generated,
        },
        channel_names,
        args.output_dir / "pdf",
        bins=args.bins,
        dpi=args.dpi,
    )

    m33_energy = _load_energy_summary(args.m33_energy_summary)
    m39_energy = _load_energy_summary(args.m39_energy_summary)
    np.testing.assert_allclose(m39_energy["k"], m33_energy["k"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        m39_energy["reference_mean"],
        m33_energy["reference_mean"],
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        m39_energy["reference_median"],
        m33_energy["reference_median"],
        rtol=1.0e-12,
        atol=1.0e-14,
    )

    save_energy_spectrum_comparison(
        m33_energy["k"],
        m33_energy["reference_mean"],
        {
            "M33 no self": m33_energy["generated_mean"],
            "M39 self": m39_energy["generated_mean"],
        },
        args.output_dir / "energy_spectrum_comparison_mean.png",
        aggregation="mean",
        eps=1.0e-30,
        dpi=args.dpi,
    )
    save_energy_spectrum_comparison(
        m33_energy["k"],
        m33_energy["reference_median"],
        {
            "M33 no self": m33_energy["generated_median"],
            "M39 self": m39_energy["generated_median"],
        },
        args.output_dir / "energy_spectrum_comparison_median.png",
        aggregation="median",
        eps=1.0e-30,
        dpi=args.dpi,
    )

    print(args.output_dir)


if __name__ == "__main__":
    main()
