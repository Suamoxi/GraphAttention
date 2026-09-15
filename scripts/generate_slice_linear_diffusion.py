"""Generate HIT slices from an M21 linear-beta DDPM run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from scripts.generate_slice_diffusion import run_diffusion_generation

from graph_attention.tasks.linear_diffusion import LinearBetaDiffusionDenoisingTask


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="generate_slice_diffusion",
)
def main(cfg: DictConfig) -> None:
    run_slice_linear_diffusion_generation(cfg)


def run_slice_linear_diffusion_generation(cfg: DictConfig) -> dict[str, Any]:
    """Reuse the M13 generation runner while recording the linear schedule."""

    run_dir = Path(str(cfg.run_dir)).expanduser().resolve()
    source_cfg_path = run_dir / "resolved_config.yaml"
    if not source_cfg_path.is_file():
        raise FileNotFoundError(f"source resolved_config.yaml does not exist: {source_cfg_path}")

    source_cfg = OmegaConf.load(source_cfg_path)
    task = instantiate(source_cfg.task)
    if not isinstance(task, LinearBetaDiffusionDenoisingTask):
        raise TypeError("M21 generation requires a linear-beta diffusion source run")

    summary = run_diffusion_generation(cfg)
    summary["noise_schedule"] = task.noise_schedule
    summary["beta_start"] = task.beta_start
    summary["beta_end"] = task.beta_end
    summary.pop("cosine_s", None)

    generation_name = str(summary["run_name"])
    summary_path = run_dir / "generations" / generation_name / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
