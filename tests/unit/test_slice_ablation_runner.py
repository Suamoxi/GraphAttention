import torch
from omegaconf import OmegaConf
from scripts.train_slice_ablation import _instantiate_model

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks import NodeRegressionTask


def _probe_batch():
    dataset = SyntheticMeshDataset(num_samples=1, spatial_dim=2, seed=11)
    task = NodeRegressionTask(input_fields=("momentum",), target_fields=("rho",))
    return task.pack_and_prepare([dataset[0]], dataset.field_catalog)


def test_m8_m9_ablation_initialization_matches_all_shared_parameters() -> None:
    probe = _probe_batch()
    common = {
        "in_channels": 2,
        "out_channels": 1,
        "hidden_dim": 16,
        "num_heads": 4,
        "num_layers": 2,
        "mlp_ratio": 2,
        "conditioning_channels": 0,
    }
    m8_cfg = OmegaConf.create(
        {"_target_": "graph_attention.models.SparseGraphTransformer", **common}
    )
    m9_cfg = OmegaConf.create(
        {
            "_target_": "graph_attention.models.GeometricSparseGraphTransformer",
            **common,
            "spatial_dim": 2,
        }
    )

    m8, m8_initialization = _instantiate_model(m8_cfg, probe, seed=42)
    m9, m9_initialization = _instantiate_model(m9_cfg, probe, seed=42)

    m8_state = m8.state_dict()
    m9_state = m9.state_dict()
    for name, value in m8_state.items():
        torch.testing.assert_close(m9_state[name], value, rtol=0.0, atol=0.0)

    geometry_names = sorted(name for name in m9_state if ".geometry_mlp." in name)
    assert geometry_names
    assert m8_initialization["shared_parameter_seed"] == 42
    assert m8_initialization["geometry_parameter_seed"] is None
    assert m9_initialization["shared_parameter_seed"] == 42
    assert m9_initialization["geometry_parameter_seed"] == 43
    assert m9_initialization["geometry_parameter_names"] == geometry_names
