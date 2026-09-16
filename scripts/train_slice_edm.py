"""Backward-compatible EDM entry point for the common generative runner."""

from __future__ import annotations

from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from graph_attention.tasks import EDMDenoisingTask
from scripts.train_generative import run_generative_training


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_slice_edm(cfg)


def run_slice_edm(cfg: DictConfig) -> dict[str, Any]:
    """Train EDM through the dataset-agnostic common generative runner."""

    task = instantiate(cfg.task)
    if not isinstance(task, EDMDenoisingTask):
        raise TypeError("EDM training requires an EDMDenoisingTask")
    return run_generative_training(cfg)


if __name__ == "__main__":
    main()
