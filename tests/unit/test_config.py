from pathlib import Path

from hydra import compose, initialize_config_dir


def test_default_hydra_config_composes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base=None):
        cfg = compose(config_name="config")

    assert cfg.seed == 42
    assert cfg.data.name == "synthetic"
    assert cfg.model.name == "baseline"
    assert cfg.task.name == "regression"
    assert cfg.optimizer._target_ == "torch.optim.AdamW"
    assert cfg.trainer._target_ == "lightning.pytorch.Trainer"
