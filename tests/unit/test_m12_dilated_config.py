from pathlib import Path

from hydra import compose, initialize_config_dir


def test_m12_dinat_model_and_geometry_configs_compose_independently() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "geometry=dinat_exact2",
                "model=alternating_dilated_geometric_sparse_transformer",
            ],
        )

    assert dict(cfg.geometry.attention_edge_indices) == {"dilated": "exact_two_hop"}
    assert cfg.model._target_ == (
        "graph_attention.models.AlternatingDilatedGeometricSparseGraphTransformer"
    )
    assert cfg.model.num_layers == 4
