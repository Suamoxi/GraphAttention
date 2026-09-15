"""Train the M21 linear-beta DDPM schedule ablation on HIT slices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from graph_attention.tasks.linear_diffusion import LinearBetaDiffusionDenoisingTask
from scripts.train_slice_diffusion import run_slice_diffusion


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_slice_linear_diffusion(cfg)


def run_slice_linear_diffusion(cfg: DictConfig) -> dict[str, Any]:
    """Reuse the M13 runner while recording the linear schedule truthfully."""

    task = instantiate(cfg.task)
    if not isinstance(task, LinearBetaDiffusionDenoisingTask):
        raise TypeError("M21 training requires task=hit_diffusion_linear")

    summary = run_slice_diffusion(cfg)
    summary["noise_schedule"] = task.noise_schedule
    summary["beta_start"] = task.beta_start
    summary["beta_end"] = task.beta_end
    summary.pop("cosine_s", None)

    run_name = cfg.get("run_name")
    output_root = Path(str(cfg.generative.output_root)).expanduser().resolve()
    summary_path = output_root / str(run_name) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
