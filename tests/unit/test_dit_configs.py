from pathlib import Path

from hydra import compose, initialize_config_dir


def test_full_and_local_dit_configs_match_except_target() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base=None):
        full = compose(config_name="config", overrides=["model=full_dit_transformer"])
        local = compose(config_name="config", overrides=["model=local_dit_transformer"])

    assert full.model._target_ == "graph_attention.models.FullDiTGraphTransformer"
    assert local.model._target_ == "graph_attention.models.LocalDiTGraphTransformer"

    full_values = dict(full.model)
    local_values = dict(local.model)
    full_values.pop("_target_")
    local_values.pop("_target_")
    assert full_values == local_values
    assert full.model.coordinate_normalization == "centered_bbox"
    assert full.model.condition_embed_dim == 128
