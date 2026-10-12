"""Standalone Physics-Token DiT: global-state math, batch isolation, gradients."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from graph_attention.data import SyntheticMeshDataset
from graph_attention.models import PhysicsTokenDiTGraphTransformer
from graph_attention.models.full_dit import FullDiTGraphTransformer
from graph_attention.models.physics_token_dit import PhysicsTokenMultiheadAttention
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.model_factory import instantiate_controlled_model


def _attention(*, tokens=5, chunk=None, sdpa=False):
    return PhysicsTokenMultiheadAttention(
        hidden_dim=24, num_heads=4, num_tokens=tokens,
        slice_temperature=0.5, token_chunk_size=chunk,
        use_sdpa=sdpa, dropout=0.0, qkv_bias=True,
    )


def _model(*, chunks=None):
    return PhysicsTokenDiTGraphTransformer(
        in_channels=5, out_channels=5,
        hidden_dim=32, num_heads=4, num_layers=3, spatial_dim=3,
        conditioning_channels=1, condition_embed_dim=32,
        num_tokens=8, token_chunk_size=chunks,
        use_sdpa=False, dropout=0.0,
    )


def test_source_formula_matches_reference():
    torch.manual_seed(501)
    attention = _attention(tokens=7).eval()
    x = torch.randn(1, 11, attention.hidden_dim)
    with torch.no_grad():
        features, weights = attention._features_and_weights(x)
        assert weights.shape == (1, 4, 11, 7)
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones_like(weights[..., 0])
        )
        tokens = torch.einsum("bhng,bhnd->bhgd", weights, features)
        tokens = tokens / (weights.sum(2)[..., None] + 1e-5)
        q, k, v = (
            attention.to_q(tokens), attention.to_k(tokens),
            attention.to_v(tokens)
        )
        score = q @ k.transpose(-2, -1) * attention.head_dim ** -0.5
        updated = score.softmax(dim=-1) @ v
        decoded = torch.einsum("bhng,bhgd->bhnd", weights, updated)
        expected = attention.out_proj(
            decoded.transpose(1, 2).contiguous().reshape(1, 11, 24)
        )
        actual = attention(
            x.squeeze(0), edge_index=torch.empty(2, 0, dtype=torch.long),
            batch_index=None,
        )
    torch.testing.assert_close(actual, expected.squeeze(0), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("sdpa", [False, True])
def test_chunked_matches_full_attention_and_all_gradients(sdpa):
    torch.manual_seed(502)
    full = _attention(tokens=6, sdpa=sdpa)
    chunk = _attention(tokens=6, chunk=4, sdpa=sdpa)
    chunk.load_state_dict(full.state_dict(), strict=True)
    x = torch.randn(13, 24, requires_grad=True)
    xx = x.detach().clone().requires_grad_(True)
    edges = torch.empty((2, 0), dtype=torch.long)

    y = full(x, edge_index=edges, batch_index=None)
    yy = chunk(xx, edge_index=edges, batch_index=None)
    torch.testing.assert_close(y, yy, atol=2e-6, rtol=2e-5)

    weights = torch.linspace(0.5, 2, 13)[:, None]
    (y.square() * weights).sum().backward()
    (yy.square() * weights).sum().backward()
    torch.testing.assert_close(x.grad, xx.grad, atol=3e-6, rtol=3e-5)
    for (name, parameter), (other_name, other) in zip(
        full.named_parameters(), chunk.named_parameters(), strict=True
    ):
        assert name == other_name
        assert parameter.grad is not None and other.grad is not None, name
        torch.testing.assert_close(
            parameter.grad, other.grad, atol=5e-6, rtol=5e-5
        )


def test_tokens_are_global_and_permutation_equivariant():
    torch.manual_seed(503)
    attention = _attention().eval()
    x = torch.randn(17, 24)
    perm = torch.randperm(17)
    edges = torch.empty(2, 0, dtype=torch.long)
    with torch.no_grad():
        out = attention(x, edge_index=edges, batch_index=None)
        permuted = attention(
            x[perm], edge_index=edges, batch_index=None
        )
        changed = x.clone()
        changed[-1] += 12.0
        changed_out = attention(
            changed, edge_index=edges, batch_index=None
        )
    torch.testing.assert_close(permuted, out[perm], atol=2e-6, rtol=2e-5)
    assert not torch.allclose(out[0], changed_out[0])


@pytest.mark.parametrize("batch_index", [
    [0, 0, 0, 1, 1, 1],
    [0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1],
])
def test_packed_physical_samples_never_share_global_tokens(batch_index):
    torch.manual_seed(504)
    attention = _attention().eval()
    x = torch.randn(6, 24)
    owners = torch.tensor(batch_index, dtype=torch.long)
    edges = torch.empty(2, 0, dtype=torch.long)

    with torch.no_grad():
        together = attention(x, edge_index=edges, batch_index=owners)
        separate = torch.empty_like(together)
        for graph in (0, 1):
            ix = torch.where(owners == graph)[0]
            alone = attention(
                x[ix], edge_index=edges, batch_index=None
            )
            separate.index_copy_(0, ix, alone)
        different = x.clone()
        different[owners == 1] += 100.0
        changed = attention(
            different, edge_index=edges, batch_index=owners
        )
    torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        together[owners == 0], changed[owners == 0], atol=1e-6, rtol=1e-5
    )


@pytest.mark.parametrize("bad", [
    {"num_tokens": 0},
    {"slice_temperature": 0},
    {"token_chunk_size": 0},
    {"dropout": 1.0},
    {"hidden_dim": 23},
])
def test_invalid_architecture_options_are_rejected(bad):
    kwargs = {
        "hidden_dim": 24, "num_heads": 4,
        "num_tokens": 8, "slice_temperature": 0.5,
        "token_chunk_size": None, "dropout": 0.0,
    }
    kwargs.update(bad)
    with pytest.raises(ValueError):
        PhysicsTokenMultiheadAttention(**kwargs)


def test_dit_zero_init_and_training_gradients():
    torch.manual_seed(505)
    model = _model()
    x = torch.randn(12, 5)
    coords = torch.randn(12, 3)
    cond = torch.tensor([[180.0]])
    no_edges = torch.empty((2, 0), dtype=torch.long)
    with torch.no_grad():
        zero = model(
            x, edge_index=no_edges, coords=coords, conditioning=cond
        )
        assert torch.equal(zero, torch.zeros_like(zero))
        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.03)
            torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.05)
        torch.nn.init.normal_(model.final_layer.projection.weight, std=0.1)

    loss = model(
        x, edge_index=no_edges, coords=coords, conditioning=cond
    ).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for name in (
        "in_project_x", "in_project_fx", "in_project_slice",
        "to_q", "to_k", "to_v", "out_proj",
    ):
        parameter = getattr(model.blocks[0].attention, name).weight
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name
    assert model.condition_projection[0].weight.grad is not None
    assert torch.isfinite(model.condition_projection[0].weight.grad).all()


def test_hydra_config_and_factory_match_full_dit_shared_body():
    repo = Path(__file__).resolve().parents[2]
    with initialize_config_dir(
        config_dir=str(repo / "configs"), version_base=None
    ):
        cfg = compose(
            config_name="config", overrides=["model=physics_token_dit"]
        )
    assert cfg.model._target_ == (
        "graph_attention.models.physics_token_dit.PhysicsTokenDiTGraphTransformer"
    )
    assert cfg.model.num_tokens == 64
    assert cfg.model.token_chunk_size is None

    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=3, seed=21)
    task = FlowMatchingTask(
        state_fields=("rho", "momentum"), time_embedding_scale=1000.0
    )
    batch = task.pack_and_prepare(
        [dataset[0]], dataset.field_catalog
    )
    problem = task.make_validation_problem(batch)
    probe = task.make_model_probe(problem)
    model_cfg = OmegaConf.create(
        {
            "_target_": cfg.model._target_,
            "hidden_dim": 32, "num_heads": 4,
            "num_layers": 2, "mlp_ratio": 2,
            "condition_embed_dim": 32, "num_tokens": 8,
            "slice_temperature": 0.5, "token_chunk_size": 3,
            "use_sdpa": False,
        }
    )
    physics, meta = instantiate_controlled_model(model_cfg, probe, seed=42)
    repeat, repeat_meta = instantiate_controlled_model(model_cfg, probe, seed=42)
    assert meta == repeat_meta
    assert meta["policy"] == "matched_dit_backbone_independent_physics_attention"
    assert meta["num_tokens"] == 8
    assert meta["token_chunk_size"] == 3
    for key, weight in physics.state_dict().items():
        torch.testing.assert_close(weight, repeat.state_dict()[key], rtol=0, atol=0)

    full_cfg = OmegaConf.create(dict(model_cfg))
    full_cfg["_target_"] = "graph_attention.models.full_dit.FullDiTGraphTransformer"
    full, _ = instantiate_controlled_model(full_cfg, probe, seed=42)
    assert isinstance(full, FullDiTGraphTransformer)
    for key, weight in full.state_dict().items():
        if ".attention." not in key:
            torch.testing.assert_close(
                weight, physics.state_dict()[key], atol=0, rtol=0
            )


def test_50_step_heun_flow_matching_smoke_and_repeatability():
    torch.manual_seed(506)
    dataset = SyntheticMeshDataset(
        num_samples=1, spatial_dim=3, seed=41
    )
    task = FlowMatchingTask(
        state_fields=("rho", "momentum"), time_embedding_scale=1000.0
    )
    batch = task.pack_and_prepare([dataset[0]], dataset.field_catalog)
    model = PhysicsTokenDiTGraphTransformer(
        in_channels=batch.inputs.shape[1],
        out_channels=batch.inputs.shape[1],
        hidden_dim=24, num_heads=4, num_layers=2, spatial_dim=3,
        conditioning_channels=1, condition_embed_dim=24,
        num_tokens=6, token_chunk_size=8, use_sdpa=True,
    ).eval()
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
            torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.04)
        torch.nn.init.normal_(model.final_layer.projection.weight, std=0.05)

    calls = [0]
    def count(_model, _args):
        calls[0] += 1

    hook = model.register_forward_pre_hook(count)
    try:
        with torch.inference_mode():
            first = task.sample_standardized(
                model, batch, steps=50, sampling_seed=97
            )
            second = task.sample_standardized(
                model, batch, steps=50, sampling_seed=97
            )
    finally:
        hook.remove()
    assert calls[0] == 200
    assert torch.isfinite(first).all()
    assert first.shape == batch.inputs.shape
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_chunk_size_limits_per_head_assignment_creation():
    torch.manual_seed(507)
    attention = _attention(tokens=8, chunk=5)
    sizes = []
    hook = attention.in_project_slice.register_forward_pre_hook(
        lambda _m, args: sizes.append(args[0].shape[-2])
    )
    try:
        with torch.no_grad():
            attention(
                torch.randn(23, 24),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                batch_index=None,
            )
    finally:
        hook.remove()
    assert max(sizes) <= 5
    assert len(sizes) == 2 * 5
