"""Compare two persisted generation benchmarks on dimensionless error magnitudes."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from graph_attention.evaluation.relative_benchmark import compare_generation_benchmarks


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="compare_generation_benchmarks",
)
def main(cfg: DictConfig) -> None:
    baseline_dir = Path(str(cfg.baseline_dir)).expanduser().resolve()
    candidate_dir = Path(str(cfg.candidate_dir)).expanduser().resolve()
    output_root = Path(str(cfg.output_root)).expanduser().resolve()
    comparison_name = cfg.comparison_name
    if comparison_name is None:
        name = f"{baseline_dir.name}__vs__{candidate_dir.name}"
    else:
        if not isinstance(comparison_name, str) or not comparison_name.strip():
            raise ValueError("comparison_name must be null or a non-empty string")
        name = comparison_name.strip()
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("comparison_name must be one directory name, not a path")

    summary = compare_generation_benchmarks(
        baseline_dir,
        candidate_dir,
        output_root / name,
        overwrite=bool(cfg.overwrite),
        eps=float(cfg.eps),
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
