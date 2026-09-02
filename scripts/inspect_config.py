"""Compose the default Hydra config and print minimal M1 provenance."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from graph_attention.utils.provenance import collect_runtime_provenance


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "configs"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config")

    print("Resolved configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("Runtime provenance:")
    print(OmegaConf.to_yaml(collect_runtime_provenance(repo_root)))


if __name__ == "__main__":
    main()
