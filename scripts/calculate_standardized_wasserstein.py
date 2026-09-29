"""Compute per-variable Wasserstein-1 in frozen training-standardized space."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from graph_attention.evaluation.generation_benchmark import standardized_wasserstein_rows

_DISPLAY_NAMES = {
    "rho": "Density rho",
    "rhou": "X-momentum rhou",
    "rhov": "Y-momentum rhov",
    "rhow": "Z-momentum rhow",
    "rhoE": "Total energy rhoE",
}


def _display_name(channel: str) -> str:
    return _DISPLAY_NAMES.get(channel.split(".", maxsplit=1)[0], channel)


def _load_generation(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], torch.Tensor]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise TypeError(f"expected mapping generation artifact: {path}")

    required = {
        "generated_nondimensional",
        "target_nondimensional",
        "generated_standardized",
        "channel_names",
    }
    missing = sorted(required.difference(artifact))
    if missing:
        raise ValueError(f"generation artifact is missing required keys: {missing}")

    generated = artifact["generated_nondimensional"]
    reference = artifact["target_nondimensional"]
    generated_standardized = artifact["generated_standardized"]
    channel_names = tuple(artifact["channel_names"])
    if not all(isinstance(value, torch.Tensor) for value in (generated, reference, generated_standardized)):
        raise TypeError("generation fields must be tensors")
    if generated.shape != reference.shape or generated.shape != generated_standardized.shape:
        raise ValueError("generation tensors must have identical shape [total_nodes, C]")

    return (
        generated.to(torch.float64).numpy(),
        reference.to(torch.float64).numpy(),
        channel_names,
        generated_standardized.to(torch.float64),
    )


def _load_standardizers(path: Path) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping standardizer artifact: {path}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("standardizers.pt is missing the input standardizer")

    channel_names = tuple(inputs["channel_names"])
    mean = inputs["mean"]
    scale = inputs["scale"]
    if not isinstance(mean, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError("standardizer mean/scale must be tensors")
    return channel_names, mean.to(torch.float64), scale.to(torch.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--standardizers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generated, reference, channel_names, stored_generated_standardized = _load_generation(
        args.generation
    )
    standardizer_channels, mean, scale = _load_standardizers(args.standardizers)

    if standardizer_channels != channel_names:
        raise ValueError(
            "generation channels and training-standardizer channels differ: "
            f"{channel_names} != {standardizer_channels}"
        )

    reconstructed_generated_standardized = (
        torch.from_numpy(generated) - mean.reshape(1, -1)
    ) / scale.reshape(1, -1)
    torch.testing.assert_close(
        reconstructed_generated_standardized,
        stored_generated_standardized,
        rtol=1.0e-5,
        atol=1.0e-6,
    )

    rows = standardized_wasserstein_rows(
        generated,
        reference,
        channel_names,
        mean.numpy(),
        scale.numpy(),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        fieldnames = [
            "channel",
            "variable",
            "training_mean",
            "training_scale",
            "wasserstein_1_nondimensional",
            "wasserstein_1_standardized",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "variable": _display_name(str(row["channel"])),
                }
            )

    print("M29 Wasserstein-1 using the frozen training standardization")
    print("Variable                          W1 standardized     W1 nondimensional")
    print("-----------------------------------------------------------------------")
    for row in rows:
        variable = _display_name(str(row["channel"]))
        print(
            f"{variable:<32}"
            f"{float(row['wasserstein_1_standardized']):>16.8g}"
            f"{float(row['wasserstein_1_nondimensional']):>20.8g}"
        )
    print()
    print(f"CSV: {args.output}")


if __name__ == "__main__":
    main()
