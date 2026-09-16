"""Backward-compatible DDPM entry point for the common generative runner."""

from __future__ import annotations

from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from graph_attention.tasks import DiffusionDenoisingTask
from scripts.train_generative import (
    _validate_generative_standardizers,
    run_generative_training,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_slice_diffusion(cfg)


def run_slice_diffusion(cfg: DictConfig) -> dict[str, Any]:
    """Train DDPM through the dataset-agnostic common generative runner."""

    task = instantiate(cfg.task)
    if not isinstance(task, DiffusionDenoisingTask):
        raise TypeError("DDPM training requires a DiffusionDenoisingTask")
    return run_generative_training(cfg)


def _validate_diffusion_standardizers(standardizers: Any) -> None:
    """Compatibility alias used by existing generation scripts."""

    _validate_generative_standardizers(standardizers)


if __name__ == "__main__":
    main()
