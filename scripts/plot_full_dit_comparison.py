"""Plot Full-DiT EDM/DDPM/Flow Matching distribution comparisons."""

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
        raise ValueError(f"generated/reference tensors must share shape [total_nodes, C]: {path}")
    if not node_counts or len(set(node_counts)) != 1 or sum(node_counts) != generated.shape[0]:
        raise ValueError(f"comparison requires one shared fixed-size mesh: {path}")
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
    parser.add_argument("--edm-generation", type=Path, required=True)
    parser.add_argument("--ddpm-generation", type=Path, required=True)
    parser.add_argument("--flow-generation", type=Path, required=True)
    parser.add_argument("--edm-energy-summary", type=Path, required=True)
    parser.add_argument("--ddpm-energy-summary", type=Path, required=True)
    parser.add_argument("--flow-energy-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    generated_by_model: dict[str, np.ndarray] = {}
    reference: np.ndarray | None = None
    channel_names: tuple[str, ...] | None = None
    for label, path in (
        ("EDM", args.edm_generation),
        ("DDPM", args.ddpm_generation),
        ("Flow Matching", args.flow_generation),
    ):
        generated, current_reference, current_names = _load_generation(path)
        if reference is None:
            reference = current_reference
            channel_names = current_names
        else:
            if current_names != channel_names:
                raise ValueError(f"{label} channel names differ from the other Full-DiT runs")
            np.testing.assert_allclose(current_reference, reference, rtol=0.0, atol=0.0)
        generated_by_model[label] = generated

    assert reference is not None
    assert channel_names is not None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = args.output_dir / "pdf"

    save_model_marginal_comparison(
        reference,
        generated_by_model,
        channel_names,
        pdf_dir,
        bins=args.bins,
        dpi=args.dpi,
    )

    energy_payloads: dict[str, dict[str, np.ndarray]] = {}
    for label, path in (
        ("EDM", args.edm_energy_summary),
        ("DDPM", args.ddpm_energy_summary),
        ("Flow Matching", args.flow_energy_summary),
    ):
        energy_payloads[label] = _load_energy_summary(path)

    k_reference = energy_payloads["EDM"]["k"]
    reference_mean = energy_payloads["EDM"]["reference_mean"]
    reference_median = energy_payloads["EDM"]["reference_median"]

    for label, payload in energy_payloads.items():
        np.testing.assert_allclose(payload["k"], k_reference, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            payload["reference_mean"],
            reference_mean,
            rtol=1.0e-12,
            atol=1.0e-14,
            err_msg=f"{label} mean reference spectrum differs",
        )
        np.testing.assert_allclose(
            payload["reference_median"],
            reference_median,
            rtol=1.0e-12,
            atol=1.0e-14,
            err_msg=f"{label} median reference spectrum differs",
        )

    generated_mean = {
        label: payload["generated_mean"] for label, payload in energy_payloads.items()
    }
    generated_median = {
        label: payload["generated_median"] for label, payload in energy_payloads.items()
    }

    save_energy_spectrum_comparison(
        k_reference,
        reference_mean,
        generated_mean,
        args.output_dir / "energy_spectrum_comparison_mean.png",
        aggregation="mean",
        eps=1.0e-30,
        dpi=args.dpi,
    )
    save_energy_spectrum_comparison(
        k_reference,
        reference_median,
        generated_median,
        args.output_dir / "energy_spectrum_comparison_median.png",
        aggregation="median",
        eps=1.0e-30,
        dpi=args.dpi,
    )

    print(pdf_dir)
    print(args.output_dir / "energy_spectrum_comparison_mean.png")
    print(args.output_dir / "energy_spectrum_comparison_median.png")


if __name__ == "__main__":
    main()
