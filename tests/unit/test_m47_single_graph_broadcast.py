"""CPU parity gates for benchmark-only native single-graph conditioning broadcast."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT
from graph_attention.models.dit_common import _broadcast_graph_features
from scripts.m43_b1_training_step import _model_forward


def test_graph_feature_expand_matches_packed_index_and_graph_gradient():
    torch.manual_seed(109)
    x = torch.randn(1, 17, dtype=torch.float32, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    batch_index = torch.zeros(43, dtype=torch.long)
    expanded = _broadcast_graph_features(x, batch_index=None, num_nodes=43)
    packed = _broadcast_graph_features(y, batch_index=batch_index, num_nodes=43)
    torch.testing.assert_close(expanded, packed, rtol=0.0, atol=0.0)
    upstream = torch.randn_like(expanded)
    (expanded * upstream).sum().backward()
    (packed * upstream).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1.0e-5, atol=1.0e-5)


def test_single_graph_mode_reaches_native_nonpacked_model_path():
    class Recorder:
        def __init__(self):
            self.batch_index = object()

        def __call__(self, inputs, *, batch_index, **kwargs):
            self.batch_index = batch_index
            return inputs

    model = Recorder()
    problem = SimpleNamespace(
        inputs=torch.ones(4, 3),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        coords=torch.zeros(4, 2),
        batch_index=torch.zeros(4, dtype=torch.long),
        conditioning=torch.ones(1, 1),
        attention_edge_indices={"dilated": torch.empty((2, 0), dtype=torch.long)},
    )
    _model_forward(model, problem, single_graph_forward=False)
    assert model.batch_index is problem.batch_index
    _model_forward(model, problem, single_graph_forward=True)
    assert model.batch_index is None


def test_checkpointed_dinat_forward_input_condition_and_weight_grad_parity():
    torch.manual_seed(27)
    reference = AlternatingDilatedGeometricDiT(
        in_channels=3, out_channels=3, hidden_dim=16, num_heads=2,
        num_layers=2, spatial_dim=2, mlp_ratio=2,
        conditioning_channels=1, condition_embed_dim=8,
        sparse_attention_backend="torch_sparse",
        activation_checkpointing=True,
        dropout=0.0,
    ).train()
    with torch.no_grad():
        reference.final_layer.projection.weight.normal_(std=0.04)
        for block in reference.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=0.04)
            block.adaLN_modulation[-1].bias.normal_(std=0.04)
    candidate = copy.deepcopy(reference)

    coords = torch.tensor([
        [0.0, 0.0], [0.5, 0.0], [1.1, 0.2],
        [1.2, 0.9], [0.1, 1.0],
    ])
    local = torch.tensor([
        [0, 1, 2, 3, 4, 0, 2, 4],
        [1, 2, 3, 4, 0, 2, 4, 1],
    ], dtype=torch.long)
    two_hop = torch.tensor([
        [0, 1, 2, 3, 4, 0, 1],
        [2, 3, 4, 0, 1, 3, 4],
    ], dtype=torch.long)
    original_inputs = torch.randn(5, 3, requires_grad=True)
    expanded_inputs = original_inputs.detach().clone().requires_grad_(True)
    original_condition = torch.randn(1, 1, requires_grad=True)
    expanded_condition = original_condition.detach().clone().requires_grad_(True)
    common = {
        "edge_index": local,
        "coords": coords,
        "attention_edge_indices": {"dilated": two_hop},
    }

    out_packed = reference(
        original_inputs, batch_index=torch.zeros(5, dtype=torch.long),
        conditioning=original_condition, **common,
    )
    out_single = candidate(
        expanded_inputs, batch_index=None,
        conditioning=expanded_condition, **common,
    )
    torch.testing.assert_close(out_single, out_packed, rtol=2e-5, atol=2e-5)
    upstream = torch.randn_like(out_packed)
    (out_packed * upstream).sum().backward()
    (out_single * upstream).sum().backward()
    torch.testing.assert_close(expanded_inputs.grad, original_inputs.grad, rtol=5e-4, atol=5e-5)
    torch.testing.assert_close(expanded_condition.grad, original_condition.grad, rtol=5e-4, atol=5e-5)
    for (name1, weight1), (name2, weight2) in zip(
        reference.named_parameters(), candidate.named_parameters(),
    ):
        assert name1 == name2
        assert (weight1.grad is None) == (weight2.grad is None)
        if weight1.grad is not None:
            torch.testing.assert_close(
                weight2.grad, weight1.grad, rtol=5e-4, atol=5e-5,
                msg=f"gradient mismatch for {name1}",
            )
