"""M43-B1b: true M37 forward, backward and AdamW parity with checkpointing."""
from __future__ import annotations

import copy

import pytest
import torch
from omegaconf import OmegaConf

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks import EDMDenoisingTask
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m43_b1b_checkpoint_gate import compare_65_loss
from graph_attention.geometry.connectivity import exact_two_hop_edge_index
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT


def _graph():
    count = 12
    links = set()
    for i in range(count):
        # Irregular finite-valence graph with nonempty exact-2-hop sets.
        for delta in (1, 3):
            j = (i + delta) % count
            links.add((i, j))
            links.add((j, i))
    edges = torch.tensor(sorted(links), dtype=torch.long).T.contiguous()
    dilated = exact_two_hop_edge_index(edges, count)
    assert dilated.size(1) > 0
    gen = torch.Generator().manual_seed(416)
    coords = torch.rand(count, 3, generator=gen, dtype=torch.float32)
    return edges, dilated, coords


def _pair(*, backend: str, self_policy: str, dropout: float = 0.15):
    settings = {
        "in_channels": 5,
        "out_channels": 5,
        "hidden_dim": 16,
        "num_heads": 2,
        "num_layers": 4,
        "spatial_dim": 3,
        "mlp_ratio": 2,
        "dropout": dropout,
        "conditioning_channels": 1,
        "condition_embed_dim": 16,
        "sparse_attention_backend": backend,
        "include_self_attention": self_policy == "both",
        "include_local_self_attention": self_policy == "local",
        "activation_checkpointing": False,
    }
    torch.manual_seed(2026)
    original = AlternatingDilatedGeometricDiT(**settings)
    # M37's adaLN-Zero/final projections initialize to zero, causing
    # all early attention gradients to be exactly zero. Populate nonzero
    # test weights so the parity test covers actual geometric attention,
    # QKV, MLP, time embedding and input-coordinate gradient paths.
    with torch.no_grad():
        gen = torch.Generator().manual_seed(819)
        for block in original.blocks:
            torch.nn.init.normal_(
                block.adaLN_modulation[-1].weight,
                mean=0.0, std=0.035, generator=gen,
            )
            torch.nn.init.normal_(
                block.adaLN_modulation[-1].bias,
                mean=0.0, std=0.035, generator=gen,
            )
        torch.nn.init.normal_(
            original.final_layer.projection.weight,
            mean=0.0, std=0.075, generator=gen,
        )
        torch.nn.init.normal_(
            original.final_layer.projection.bias,
            mean=0.0, std=0.025, generator=gen,
        )
    checkpointed = copy.deepcopy(original)
    checkpointed.activation_checkpointing = True
    original.train()
    checkpointed.train()
    assert original.state_dict().keys() == checkpointed.state_dict().keys()
    return original, checkpointed


def _forward(model, fields, coordinates, time, edges, dilated):
    return model(
        fields, edge_index=edges, coords=coordinates,
        batch_index=torch.zeros(fields.size(0), dtype=torch.long),
        conditioning=time,
        attention_edge_indices={"dilated": dilated},
    )


@pytest.mark.parametrize("backend", ["scatter", "torch_sparse"])
@pytest.mark.parametrize("self_policy", ["none", "local", "both"])
def test_checkpointed_m37_forward_gradients_and_adamw_match(
    backend: str, self_policy: str,
) -> None:
    edges, dilated, coords = _graph()
    reference, checkpointed = _pair(backend=backend, self_policy=self_policy)
    torch.manual_seed(440)
    x_ref = torch.randn(len(coords), 5, requires_grad=True)
    x_ckpt = x_ref.detach().clone().requires_grad_(True)
    coord_ref = coords.detach().clone().requires_grad_(True)
    coord_ckpt = coords.detach().clone().requires_grad_(True)
    cond_ref = torch.tensor([[0.41]], requires_grad=True)
    cond_ckpt = cond_ref.detach().clone().requires_grad_(True)
    targets = torch.randn(len(coords), 5)
    optim_a = torch.optim.AdamW(reference.parameters(), lr=1.0e-4)
    optim_b = torch.optim.AdamW(checkpointed.parameters(), lr=1.0e-4)
    observed = []
    checkpointed.set_block_memory_observer(lambda layer: observed.append(layer))

    for step in range(2):
        optim_a.zero_grad(set_to_none=True)
        optim_b.zero_grad(set_to_none=True)
        for tensor in (x_ref, x_ckpt, coord_ref, coord_ckpt, cond_ref, cond_ckpt):
            tensor.grad = None
        torch.manual_seed(731 + step)
        out_a = _forward(reference, x_ref, coord_ref, cond_ref, edges, dilated)
        torch.manual_seed(731 + step)
        out_b = _forward(
            checkpointed, x_ckpt, coord_ckpt, cond_ckpt, edges, dilated,
        )
        torch.testing.assert_close(out_b, out_a, atol=2e-6, rtol=2e-6)
        loss_a = (out_a - targets).square().mean()
        loss_b = (out_b - targets).square().mean()
        loss_a.backward()
        loss_b.backward()

        for name, a in reference.named_parameters():
            b = dict(checkpointed.named_parameters())[name]
            assert a.grad is not None, f"baseline missing gradient: {name}"
            assert b.grad is not None, f"checkpoint missing gradient: {name}"
            torch.testing.assert_close(b.grad, a.grad, atol=5e-5, rtol=5e-4)
        for a, b in (
            (x_ref, x_ckpt),
            (coord_ref, coord_ckpt),
            (cond_ref, cond_ckpt),
        ):
            torch.testing.assert_close(b.grad, a.grad, atol=5e-5, rtol=5e-4)
        assert reference.blocks[0].attention.qkv.weight.grad.abs().max().item() > 0
        assert (reference.blocks[0].attention.geometry_mlp[0].weight.grad
                .abs().max().item() > 0)

        optim_a.step()
        optim_b.step()
        for name, tensor in reference.state_dict().items():
            torch.testing.assert_close(
                checkpointed.state_dict()[name], tensor, atol=5e-6, rtol=5e-5,
            )
    # Checkpoint recomputation must not generate observer events.
    assert observed == list(range(len(checkpointed.blocks))) * 2


def test_checkpointing_bypassed_in_eval_and_no_grad(monkeypatch) -> None:
    edges, dilated, coords = _graph()
    reference, checkpointed = _pair(backend="scatter", self_policy="none", dropout=0.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint must not run outside training grad mode")

    monkeypatch.setattr("graph_attention.models.dinat_dit.checkpoint", forbidden)
    x = torch.randn(len(coords), 5)
    time = torch.tensor([[0.2]])
    reference.eval()
    checkpointed.eval()
    with torch.no_grad():
        normal = _forward(reference, x, coords, time, edges, dilated)
        checkpoint_mode = _forward(checkpointed, x, coords, time, edges, dilated)
        torch.testing.assert_close(checkpoint_mode, normal, rtol=0, atol=0)
    checkpointed.train()
    with torch.no_grad():
        training_without_grad = _forward(
            checkpointed, x, coords, time, edges, dilated,
        )
        torch.testing.assert_close(training_without_grad, normal, rtol=0, atol=0)


def test_checkpoint_configuration_default_is_disabled() -> None:
    original, candidate = _pair(backend="scatter", self_policy="none")
    assert not original.activation_checkpointing
    assert candidate.activation_checkpointing
    assert original.state_dict().keys() == candidate.state_dict().keys()
    with pytest.raises(TypeError, match="callable"):
        candidate.set_block_memory_observer("not-a-callback")


def test_factory_option_preserves_weights_and_reuses_m37_initialization() -> None:
    dataset = SyntheticMeshDataset(num_samples=1, spatial_dim=3, seed=416)
    task = EDMDenoisingTask(
        state_fields=("rho", "momentum"),
        physical_nondimensionalization=False,
        validation_seed=91,
    )
    batch = task.pack_and_prepare([dataset[0]], dataset.field_catalog)
    probe = task.make_model_probe(task.make_validation_problem(batch))
    cfg = OmegaConf.create({
        "_target_": "graph_attention.models.dinat_dit.AlternatingDilatedGeometricDiT",
        "hidden_dim": 16, "num_heads": 4, "num_layers": 4,
        "mlp_ratio": 2, "condition_embed_dim": 16,
        "sparse_attention_backend": "torch_sparse",
    })
    default, metadata_default = instantiate_controlled_model(cfg, probe, seed=42)
    cfg.activation_checkpointing = True
    active, metadata_active = instantiate_controlled_model(cfg, probe, seed=42)
    assert not default.activation_checkpointing
    assert active.activation_checkpointing
    assert metadata_default["activation_checkpointing"] is False
    assert metadata_active["activation_checkpointing"] is True
    assert default.state_dict().keys() == active.state_dict().keys()
    for name, value in default.state_dict().items():
        torch.testing.assert_close(
            active.state_dict()[name], value, atol=0, rtol=0,
        )


def test_real_probe_loss_comparison_contract() -> None:
    a = {
        "status": "ok", "per_step": [
            {"step": 1, "loss_probe_only": 2.01},
            {"step": 2, "loss_probe_only": 1.99},
        ]
    }
    b = {
        "status": "ok", "per_step": [
            {"step": 1, "loss_probe_only": 2.010001},
            {"step": 2, "loss_probe_only": 1.989999},
        ]
    }
    assert compare_65_loss(a, b)["status"] == "passed"
    b["per_step"][0]["loss_probe_only"] = 2.1
    assert compare_65_loss(a, b)["status"] == "numerical_mismatch"
    b["status"] = "cuda_oom"
    assert compare_65_loss(a, b)["status"] == "unavailable"
