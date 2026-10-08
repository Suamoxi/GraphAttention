"""Side-by-side true 3-D volumetric comparison for M37 and M40 HIT generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from graph_attention.evaluation.generation_benchmark import _fixed_mesh_samples
from graph_attention.evaluation.generation_benchmark_3d import _load_grid_3d
from graph_attention.evaluation.volume_rendering_3d import (
    save_volume_population_comparison,
)


def _load_run(path: Path):
    for name in ("generated_test.pt", "resolved_config.yaml", "summary.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"missing 3D generation artifact {path / name}")
    artifact = torch.load(path / "generated_test.pt", map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise TypeError(f"invalid 3D generation tensor artifact {path}")
    return _fixed_mesh_samples(artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m37-generation", type=Path, required=True)
    parser.add_argument("--m40-generation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=220)
    parser.add_argument("--steps", type=int, default=84)
    parser.add_argument("--optical-depth", type=float, default=2.0)
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    m37_generated, m37_reference, _, m37_test_ids, channel_names = _load_run(
        args.m37_generation
    )
    m40_generated, m40_reference, _, m40_test_ids, m40_channel_names = _load_run(
        args.m40_generation
    )
    if channel_names != m40_channel_names:
        raise ValueError("M37/M40 generation channels do not match")
    if m37_test_ids != m40_test_ids:
        raise ValueError("M37/M40 held-out reference sample IDs differ")
    if m37_reference.shape != m40_reference.shape:
        raise ValueError("M37/M40 reference population shapes differ")
    np.testing.assert_allclose(m37_reference, m40_reference, rtol=0.0, atol=0.0)

    cfg37 = OmegaConf.load(args.m37_generation / "resolved_config.yaml")
    cfg40 = OmegaConf.load(args.m40_generation / "resolved_config.yaml")
    if str(cfg37.data.mesh_file) != str(cfg40.data.mesh_file):
        raise ValueError("M37/M40 source 3D meshes differ")
    benchmark_cfg = OmegaConf.load("configs/benchmark_generation_3d.yaml")
    grid, _ = _load_grid_3d(cfg37, benchmark_cfg)

    expected = int(np.prod(grid.source_shape))
    if m37_reference.shape[1] != expected:
        raise ValueError("3D render mesh node count does not match stored samples")

    save_volume_population_comparison(
        {
            "DNS test reference": m37_reference,
            "M37 without self": m37_generated,
            "M40 local self": m40_generated,
        },
        channel_names,
        grid,
        args.output_dir,
        num_examples=args.num_examples,
        dpi=args.dpi,
        cmap=args.cmap,
        resolution=args.resolution,
        steps=args.steps,
        optical_depth=args.optical_depth,
    )
    print(f"3-D volume renders saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
