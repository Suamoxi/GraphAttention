"""Hydra launcher for full-volume 3-D generation distribution benchmarking."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from graph_attention.evaluation.generation_benchmark_3d import (
    run_generation_benchmark_3d,
)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="benchmark_generation_3d",
)
def main(cfg: DictConfig) -> None:
    summary = run_generation_benchmark_3d(cfg)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
