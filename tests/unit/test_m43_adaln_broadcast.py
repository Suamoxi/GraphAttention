"""Reference-parity gates for M43 single-graph adaLN broadcast optimization.

Exact previous implementation is defined locally as _legacy_broadcast. The
model/attention/time embeddings remain identical in both modes.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from graph_attention.geometry.connectivity import exact_two_hop_edge_index
from graph_attention.models import dit_common
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT


def _legacy_broadcast(
    features: torch.Tensor, *, batch_index: torch.Tensor | None, num_nodes: int,
) -> torch.Tensor:
    """Byte-for-byte prior mathematical implementation for comparisons."""
    if features.ndim != 2:
        raise ValueError("graph features must have shape [B, C]")
    if batch_index is None:
        if features.shape[0] != 1:
            raise ValueError(
                "single-graph inputs require exactly one graph feature row"
            )
        return features.expand(num_nodes, -1)
    return features[batch_index]


@contextmanager
def _variant(*, legacy: bool):
    if legacy:
        with patch.object(dit_common, "_broadcast_graph_features", _legacy_broadcast):
            yield
    else:
        yield


def _mesh_pair(device: torch.device, *, packed: bool):
    # Deliberately different unstructured chain/shortcut connectivity, node
    # counts and coordinate extents, not two images on a common regular grid.
    sizes = (4, 7) if packed else (7,)
    xyz = [
        torch.tensor([[0., 0.], [1., .4], [1.8, 1.1], [2.2, .9]], device=device),
        torch.tensor([
            [10., -1.], [10.4, 0.3], [11.5, .1], [12., .9],
            [13.1, .2], [13.3, 1.1], [14., 1.5],
        ], device=device),
    ]
    graph_coords = xyz if packed else xyz[1:]
    coords = torch.cat(graph_coords, dim=0)
    onehop = []
    offset = 0
    for count in sizes:
        undirected = [(i, i+1) for i in range(count-1)]
        if count == 7:
            undirected += [(0, 2), (3, 5)]
        source = torch.tensor(
            [u for u,v in undirected] + [v for u,v in undirected],
            device=device, dtype=torch.long,
        ) + offset
        target = torch.tensor(
            [v for u,v in undirected] + [u for u,v in undirected],
            device=device, dtype=torch.long,
        ) + offset
        onehop.append(torch.stack([source,target]))
        offset += count
    edge_index = torch.cat(onehop, dim=1)
    twohop = exact_two_hop_edge_index(edge_index.cpu(), sum(sizes)).to(device)
    idx = torch.cat([
        torch.full((count,), graph_id, device=device, dtype=torch.long)
        for graph_id, count in enumerate(sizes)
    ])
    # FlowMatchingTask scales t by 1000 before passing to time embedding.
    flow_times = (
        torch.tensor([[250.], [790.]], dtype=torch.float32, device=device)
        if packed else torch.tensor([[790.]], dtype=torch.float32, device=device)
    )
    generator = torch.Generator(device="cpu").manual_seed(121)
    inputs = torch.randn(sum(sizes), 5, generator=generator).to(device)
    return {
        "inputs": inputs, "coords": coords,
        "edge_index": edge_index, "batch_index": idx,
        "conditioning": flow_times,
        "attention_edge_indices": {"dilated": twohop},
    }


def _model(device: torch.device, *, checkpointing: bool):
    torch.manual_seed(2026)
    model = AlternatingDilatedGeometricDiT(
        in_channels=5, out_channels=5, hidden_dim=16,
        num_heads=2, num_layers=4, spatial_dim=2, mlp_ratio=2,
        dropout=0., conditioning_channels=1, condition_embed_dim=16,
        sparse_attention_backend="scatter",
        include_self_attention=False,
        activation_checkpointing=checkpointing,
    )
    with torch.no_grad():
        # Turn on the initially zero DiT gates and final readout, so
        # every layer's shared conditioning gradient is actually exercised.
        model.final_layer.projection.weight.normal_(0., .04)
        model.final_layer.adaLN_modulation[-1].weight.normal_(0., .02)
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.normal_(0., .02)
            block.adaLN_modulation[-1].bias[2*model.hidden_dim:3*model.hidden_dim] = .1
            block.adaLN_modulation[-1].bias[5*model.hidden_dim:6*model.hidden_dim] = .1
    return model.to(device)


def _loss_and_grads(model, problem, legacy: bool):
    model.zero_grad(set_to_none=True)
    with _variant(legacy=legacy):
        prediction = model(**problem)
        loss = prediction.square().mean()
        loss.backward()
    grads = {
        name: None if p.grad is None else p.grad.detach().clone()
        for name,p in model.named_parameters()
    }
    return prediction.detach().clone(), loss.detach().clone(), grads


@pytest.mark.parametrize("num_nodes", [1, 7, 187])
def test_single_graph_direct_broadcast_forward_backward_exact_cpu(num_nodes: int):
    torch.manual_seed(77)
    features = torch.randn(1, 6, requires_grad=True)
    weights = torch.randn(num_nodes, 6)
    index = torch.zeros(num_nodes, dtype=torch.long)
    original = _legacy_broadcast(features, batch_index=index, num_nodes=num_nodes)
    optimized = dit_common._broadcast_graph_features(
        features, batch_index=index, num_nodes=num_nodes
    )
    assert optimized._base is features  # no materialized gather, only expand view
    torch.testing.assert_close(original, optimized, atol=0., rtol=0.)
    ref_grad = torch.autograd.grad((original*weights).sum(), features)[0]
    fast_grad = torch.autograd.grad((optimized*weights).sum(), features)[0]
    torch.testing.assert_close(ref_grad, fast_grad, atol=3e-5, rtol=1e-6)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_full_dinat_forward_and_backward_reference_parity(
    device_name: str, packed: bool, checkpointing: bool,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device_name)
    problem = _mesh_pair(device, packed=packed)
    model = _model(device, checkpointing=checkpointing)
    baseline_y, baseline_loss, baseline_grads = _loss_and_grads(
        model, problem, legacy=True,
    )
    fast_y, fast_loss, fast_grads = _loss_and_grads(
        model, problem, legacy=False,
    )
    tolerance = dict(rtol=3e-4, atol=2e-6)
    torch.testing.assert_close(fast_y, baseline_y, **tolerance)
    torch.testing.assert_close(fast_loss, baseline_loss, **tolerance)
    assert fast_grads.keys() == baseline_grads.keys()
    for name, old in baseline_grads.items():
        new = fast_grads[name]
        assert (old is None) == (new is None), name
        if old is not None:
            torch.testing.assert_close(new, old, **tolerance, msg=lambda m: f"{name}: {m}")

    # For packed 4+7, the fallback remains EXACTLY indexed broadcasting.
    if packed:
        f = torch.randn(2, 16, device=device)
        raw = dit_common._broadcast_graph_features(
            f, batch_index=problem["batch_index"], num_nodes=11,
        )
        torch.testing.assert_close(
            raw, f[problem["batch_index"]], rtol=0., atol=0.,
        )


def test_packed_heterogeneous_graphs_receive_distinct_time_modulations():
    device = torch.device("cpu")
    p = _mesh_pair(device, packed=True)
    model = _model(device, checkpointing=False).eval()
    with torch.no_grad():
        left = model(**p)
        swapped = dict(p, conditioning=torch.tensor([[250.], [250.]]))
        right = model(**swapped)
    torch.testing.assert_close(left[:4], right[:4], rtol=0., atol=0.)
    assert not torch.allclose(left[4:], right[4:], rtol=0., atol=1e-7)


def test_packed_invalid_indices_rejected_before_single_graph_fast_path():
    p = _mesh_pair(torch.device("cpu"), packed=False)
    model = _model(torch.device("cpu"), checkpointing=False)
    broken = dict(p, batch_index=torch.ones_like(p["batch_index"]))
    with pytest.raises(ValueError, match="start at graph 0"):
        model(**broken)
    truncated = dict(p, batch_index=p["batch_index"][:-1])
    with pytest.raises(ValueError, match=r"shape \[N\]"):
        model(**truncated)


def test_full_optimizer_step_unchanged_for_two_different_meshes():
    problem = _mesh_pair(torch.device("cpu"), packed=True)
    reference = _model(torch.device("cpu"), checkpointing=True)
    optimized = _model(torch.device("cpu"), checkpointing=True)
    optimized.load_state_dict(reference.state_dict())
    a = torch.optim.AdamW(reference.parameters(), lr=3e-4, weight_decay=0.)
    b = torch.optim.AdamW(optimized.parameters(), lr=3e-4, weight_decay=0.)
    for model, optimizer, legacy in ((reference, a, True), (optimized, b, False)):
        with _variant(legacy=legacy):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**problem).square().mean()
            loss.backward()
            optimizer.step()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(
            value, optimized.state_dict()[name], rtol=1e-5, atol=2e-7,
        )
    assert len(a.state) == len(b.state)
