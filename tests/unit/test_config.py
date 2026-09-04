from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from graph_attention.data import SyntheticMeshDataset
from graph_attention.models import NodeLinearBaseline, SparseGraphTransformer
from graph_attention.tasks import NodeRegressionTask


def _config(overrides: list[str] | None = None):
    repo_root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base=None):
        return compose(config_name="config", overrides=overrides or [])


def test_default_hydra_config_composes() -> None:
    cfg = _config()

    assert cfg.seed == 42
    assert cfg.data._target_ == "graph_attention.data.SyntheticMeshDataset"
    assert cfg.data.seed == cfg.seed
    assert cfg.model._target_ == "graph_attention.models.NodeLinearBaseline"
    assert cfg.model.in_channels == 2
    assert cfg.model.out_channels == 1
    assert cfg.task._target_ == "graph_attention.tasks.NodeRegressionTask"
    assert list(cfg.task.input_fields) == ["momentum"]
    assert list(cfg.task.target_fields) == ["rho"]
    assert cfg.optimizer._target_ == "torch.optim.AdamW"
    assert cfg.trainer._target_ == "lightning.pytorch.Trainer"


def test_default_synthetic_data_config_instantiates() -> None:
    dataset = instantiate(_config().data)

    assert isinstance(dataset, SyntheticMeshDataset)
    assert len(dataset) == 9


def test_default_task_and_model_config_instantiate_and_connect() -> None:
    cfg = _config()
    dataset = instantiate(cfg.data)
    task = instantiate(cfg.task)
    model = instantiate(cfg.model)

    assert isinstance(task, NodeRegressionTask)
    assert isinstance(model, NodeLinearBaseline)

    batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    output = model(
        batch.inputs,
        edge_index=batch.edge_index,
        batch_index=batch.batch_index,
        conditioning=batch.conditioning,
    )

    assert output.shape == batch.targets.shape


def test_sparse_transformer_config_instantiates_and_connects() -> None:
    cfg = _config(["model=sparse_transformer"])
    dataset = instantiate(cfg.data)
    task = instantiate(cfg.task)
    model = instantiate(cfg.model)

    assert isinstance(model, SparseGraphTransformer)
    assert model.hidden_dim == 64
    assert model.num_heads == 4
    assert model.num_layers == 2

    batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    output = model(
        batch.inputs,
        edge_index=batch.edge_index,
        batch_index=batch.batch_index,
        conditioning=batch.conditioning,
    )

    assert output.shape == batch.targets.shape


def test_avbp_hdf5_config_composes() -> None:
    cfg = _config(["data=avbp_hdf5"])

    assert cfg.data._target_ == "graph_attention.data.AVBPHDF5Dataset"
    assert list(cfg.data.samples) == []
    assert dict(cfg.data.case_files) == {}
    assert list(cfg.data.field_names) == ["rho", "rhou", "rhov", "rhow", "rhoE"]
    assert cfg.data.connectivity_path == "Connectivity/hex->node"
    assert cfg.data.connectivity_indexing == "auto"
