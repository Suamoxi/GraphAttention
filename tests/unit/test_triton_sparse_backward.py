"""Triton CSR backward numerical parity on directed, irregular CFD-like graphs."""
from __future__ import annotations

import pytest
import torch

from graph_attention.geometry import edge_relative_displacement
from graph_attention.models.dinat_dit import (
    AlternatingDilatedGeometricDiT, DiNATDiTMultiheadAttention,
)
from graph_attention.models.sparse_transformer import (
    _TorchSparseSDDMM, _TorchSparseSpMM, _build_torch_sparse_topology,
)
from graph_attention.models.triton_sparse_backward import (
    TritonSparseSDDMM, TritonSparseSpMM, require_triton,
)


def _gpu() -> torch.device:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("Triton sparse backward is CUDA-only")
    return torch.device("cuda")


def _edges(nodes: int, *, long_row: bool) -> torch.Tensor:
    if long_row:
        # One target has degree 49 (> the 32-edge Triton tile).
        edges = [(j, 0) for j in range(1, 50)]
        edges += [(0, 2), (3, 4), (4, 3), (7, 1), (2, 2)]
    else:
        # Directed, unsorted, self-loop, zero-indegree and zero-outdegree nodes.
        edges = [
            (4, 2), (0, 2), (1, 2), (3, 2), (5, 2), (6, 2),
            (7, 2), (9, 2), (2, 0), (2, 4), (8, 1), (1, 7),
            (0, 10), (5, 8), (7, 7),
        ]
    assert max(max(pair) for pair in edges) < nodes
    return torch.tensor(edges, dtype=torch.long).T.contiguous()


@pytest.mark.parametrize("nodes,long_row", [(12, False), (53, True)])
@pytest.mark.parametrize("head_dim", [4, 32])
def test_triton_sddmm_backward_matches_chunked_irregular(nodes, long_row, head_dim):
    device = _gpu()
    torch.manual_seed(101)
    edges = _edges(nodes, long_row=long_row).to(device)
    topo = _build_torch_sparse_topology(edges, num_nodes=nodes)
    heads = 4
    # Unbind creates the noncontiguous node stride used by real DiNAT QKV.
    qkv = torch.randn(nodes, 3, heads, head_dim, device=device)
    q1 = qkv[:, 0].detach().requires_grad_(True)
    k1 = qkv[:, 1].detach().requires_grad_(True)
    q2 = q1.detach().clone().requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    expected = _TorchSparseSDDMM.apply(
        q1, k1, topo.crow_indices, topo.row_indices, topo.col_indices, 0.15,
    )
    actual = TritonSparseSDDMM.apply(q2, k2, topo, 0.15)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(expected)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(q2.grad, q1.grad, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(k2.grad, k1.grad, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("nodes,long_row", [(12, False), (53, True)])
@pytest.mark.parametrize("head_dim", [4, 32])
def test_triton_spmm_backward_matches_chunked_irregular(nodes, long_row, head_dim):
    device = _gpu()
    torch.manual_seed(102)
    edges = _edges(nodes, long_row=long_row).to(device)
    topo = _build_torch_sparse_topology(edges, num_nodes=nodes)
    values = torch.randn(nodes, 4, head_dim, device=device)
    w1 = torch.randn(edges.shape[1], 4, device=device, requires_grad=True)
    v1 = values.detach().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)
    v2 = v1.detach().clone().requires_grad_(True)
    reference = _TorchSparseSpMM.apply(w1, v1, topo.row_indices, topo.col_indices)
    candidate = TritonSparseSpMM.apply(w2, v2, topo)
    torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(reference)
    (reference * upstream).sum().backward()
    (candidate * upstream).sum().backward()
    torch.testing.assert_close(w2.grad, w1.grad, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(v2.grad, v1.grad, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("nodes,long_row", [(12, False), (53, True)])
def test_triton_geometric_attention_outputs_parameters_and_bias(nodes, long_row):
    device = _gpu()
    torch.manual_seed(201)
    edges = _edges(nodes, long_row=long_row).to(device)
    xyz = torch.randn(nodes, 3, device=device)
    delta = edge_relative_displacement(xyz, edges)
    baseline = DiNATDiTMultiheadAttention(
        128, 4, 3, sparse_attention_backend="torch_sparse",
    ).to(device)
    candidate = DiNATDiTMultiheadAttention(
        128, 4, 3, sparse_attention_backend="torch_sparse_triton",
    ).to(device)
    candidate.load_state_dict(baseline.state_dict())
    x1 = torch.randn(nodes, 128, device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    d1 = delta.detach().clone().requires_grad_(True)
    d2 = delta.detach().clone().requires_grad_(True)
    opts = {"edge_index": edges, "batch_index": None}
    out1 = baseline(x1, edge_displacement=d1, **opts)
    out2 = candidate(x2, edge_displacement=d2, **opts)
    torch.testing.assert_close(out2, out1, rtol=1e-5, atol=1e-5)
    upstream = torch.randn_like(out1)
    (out1 * upstream).sum().backward()
    (out2 * upstream).sum().backward()
    torch.testing.assert_close(x2.grad, x1.grad, rtol=5e-4, atol=5e-5)
    torch.testing.assert_close(d2.grad, d1.grad, rtol=5e-4, atol=5e-5)
    for name, p1 in baseline.named_parameters():
        p2 = dict(candidate.named_parameters())[name]
        torch.testing.assert_close(
            p2.grad, p1.grad, rtol=5e-4, atol=5e-5,
            msg=lambda m, name=name: f"parameter {name}: {m}",
        )


def test_triton_full_dinat_with_activation_checkpointing():
    device = _gpu()
    torch.manual_seed(303)
    args = dict(
        in_channels=5, out_channels=5, hidden_dim=32, num_heads=4,
        num_layers=2, spatial_dim=3, mlp_ratio=2,
        conditioning_channels=1, condition_embed_dim=16,
        activation_checkpointing=True,
    )
    reference = AlternatingDilatedGeometricDiT(
        **args, sparse_attention_backend="torch_sparse",
    ).to(device)
    candidate = AlternatingDilatedGeometricDiT(
        **args, sparse_attention_backend="torch_sparse_triton",
    ).to(device)
    candidate.load_state_dict(reference.state_dict())
    local = _edges(12, long_row=False).to(device)
    dilated = torch.tensor(
        [[0, 2, 4, 6, 8, 10], [3, 5, 7, 9, 11, 0]],
        dtype=torch.long, device=device,
    )
    kwargs = dict(
        edge_index=local, coords=torch.randn(12, 3, device=device),
        attention_edge_indices={"dilated": dilated},
        conditioning=torch.tensor([[0.2]], device=device),
    )
    x1 = torch.randn(12, 5, device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    reference.train()
    candidate.train()
    out1 = reference(x1, **kwargs)
    out2 = candidate(x2, **kwargs)
    torch.testing.assert_close(out2, out1, rtol=1e-5, atol=1e-5)
    upstream = torch.randn_like(out1)
    (out1 * upstream).sum().backward()
    (out2 * upstream).sum().backward()
    torch.testing.assert_close(x2.grad, x1.grad, rtol=5e-4, atol=5e-5)


def test_triton_backend_refuses_missing_cuda_or_triton():
    from graph_attention.models.triton_sparse_backward import triton
    if triton is not None and torch.cuda.is_available():
        require_triton()
        return
    with pytest.raises(RuntimeError, match="requires"):
        require_triton()
