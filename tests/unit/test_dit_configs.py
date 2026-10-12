from pathlib import Path

from hydra import compose, initialize_config_dir


def test_full_and_local_dit_configs_match_except_target() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base=None):
        full = compose(config_name="config", overrides=["model=full_dit_transformer"])
        local = compose(config_name="config", overrides=["model=local_dit_transformer"])

    assert full.model._target_ == (
        "graph_attention.models.full_dit.FullDiTGraphTransformer"
    )
    assert local.model._target_ == (
        "graph_attention.models.local_dit.LocalDiTGraphTransformer"
    )

    full_values = dict(full.model)
    local_values = dict(local.model)
    full_values.pop("_target_")
    local_values.pop("_target_")
    assert full_values == local_values
    assert full.model.coordinate_normalization == "centered_bbox"
    assert full.model.condition_embed_dim == 128


def test_dinat_dit_config_matches_full_dit_base_settings() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base=None):
        full = compose(config_name="config", overrides=["model=full_dit_transformer"])
        dinat = compose(config_name="config", overrides=["model=dinat_dit"])

    assert dinat.model._target_ == (
        "graph_attention.models.dinat_dit.AlternatingDilatedGeometricDiT"
    )

    full_values = dict(full.model)
    dinat_values = dict(dinat.model)
    full_values.pop("_target_")
    dinat_values.pop("_target_")

    # DiNAT has its own sparse-attention and self-loop settings. They are
    # not part of the shared Full-DiT backbone and cannot be expected in
    # the Full-DiT configuration.
    assert set(dinat_values) - set(full_values) == {
        "sparse_attention_backend",
        "include_self_attention",
    }
    assert {key: dinat_values[key] for key in full_values} == full_values
    assert dinat_values["sparse_attention_backend"] in {
        "scatter", "torch_sparse", "dgl",
    }
    assert dinat_values["include_self_attention"] is False
