"""Hydra launcher for per-run generative distribution benchmarking."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from graph_attention.evaluation import run_generation_benchmark


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_generation")
def main(cfg: DictConfig) -> None:
    summary = run_generation_benchmark(cfg)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
