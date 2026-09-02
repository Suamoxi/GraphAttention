from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from graph_attention.data import SyntheticMeshDataset


def _config(overrides: list[str] | None = None):
    repo_root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base=None):
        return compose(config_name="config", overrides=overrides or [])


def test_default_hydra_config_composes() -> None:
    cfg = _config()

    assert cfg.seed == 42
    assert cfg.data._target_ == "graph_attention.data.SyntheticMeshDataset"
    assert cfg.data.seed == cfg.seed
    assert cfg.model.name == "baseline"
    assert cfg.task.name == "regression"
    assert cfg.optimizer._target_ == "torch.optim.AdamW"
    assert cfg.trainer._target_ == "lightning.pytorch.Trainer"


def test_default_synthetic_data_config_instantiates() -> None:
    dataset = instantiate(_config().data)

    assert isinstance(dataset, SyntheticMeshDataset)
    assert len(dataset) == 9


def test_avbp_hdf5_config_composes() -> None:
    cfg = _config(["data=avbp_hdf5"])

    assert cfg.data._target_ == "graph_attention.data.AVBPHDF5Dataset"
    assert list(cfg.data.field_names) == ["rho", "rhou", "rhov", "rhow", "rhoE"]
    assert cfg.data.connectivity_path == "Connectivity/hex->node"
    assert cfg.data.connectivity_indexing == "auto"
