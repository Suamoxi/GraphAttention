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


def _assert_parity(name: str, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float) -> None:
    """Print reproducible error magnitudes before enforcing numerical parity."""
    difference = (actual.detach() - expected.detach()).abs()
    max_absolute = difference.max().item() if difference.numel() else 0.0
    # Near-zero reference values make unregularized relative errors misleading.
    relative = difference / expected.detach().abs().clamp_min(1e-6)
    max_relative = relative.max().item() if relative.numel() else 0.0
    print(
        f"[TRITON PARITY] {name}: max_abs={max_absolute:.6e}, "
        f"max_rel(ref_floor=1e-6)={max_relative:.6e}",
        flush=True,
    )
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _edges(nodes: int, *, long_row: bool) -> torch.Tensor:
    if long_row:
        # Both target- and source-oriented CSR have degree 49 (> 32-edge tile).
        edges = [(j, 0) for j in range(1, 50)]
        edges += [(0, j) for j in range(1, 50)]
        edges += [(3, 4), (4, 3), (7, 1), (2, 2)]
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
    # Cloning a sliced [N, H, D] view makes it contiguous. Clone the packed
    # [N, 3, H, D] storage first so both backends exercise the same Q/K strides.
    qkv2 = qkv.detach().clone()
    q2 = qkv2[:, 0].detach().requires_grad_(True)
    k2 = qkv2[:, 1].detach().requires_grad_(True)
    assert not q2.is_contiguous() and not k2.is_contiguous()
    assert q2.stride() == q1.stride() and k2.stride() == k1.stride()
    expected = _TorchSparseSDDMM.apply(
        q1, k1, topo.crow_indices, topo.row_indices, topo.col_indices, 0.15,
    )
    actual = TritonSparseSDDMM.apply(q2, k2, topo, 0.15)
    _assert_parity("sddmm forward", actual, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(expected)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    _assert_parity("sddmm dQ", q2.grad, q1.grad, rtol=3e-4, atol=3e-5)
    _assert_parity("sddmm dK", k2.grad, k1.grad, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("nodes,long_row", [(12, False), (53, True)])
@pytest.mark.parametrize("head_dim", [4, 32])
def test_triton_spmm_backward_matches_chunked_irregular(nodes, long_row, head_dim):
    device = _gpu()
    torch.manual_seed(102)
    edges = _edges(nodes, long_row=long_row).to(device)
    topo = _build_torch_sparse_topology(edges, num_nodes=nodes)
    # Exercise noncontiguous weight/value strides, including the transposed
    # CSR reduction of dV, rather than only the contiguous benchmark layout.
    weight_storage = torch.randn(edges.shape[1], 4, 2, device=device)
    value_storage = torch.randn(nodes, 2, 4, head_dim, device=device)
    w1 = weight_storage[:, :, 0].detach().requires_grad_(True)
    v1 = value_storage[:, 0].detach().requires_grad_(True)
    w2 = weight_storage.detach().clone()[:, :, 0].detach().requires_grad_(True)
    v2 = value_storage.detach().clone()[:, 0].detach().requires_grad_(True)
    assert not w2.is_contiguous() and not v2.is_contiguous()
    assert w2.stride() == w1.stride() and v2.stride() == v1.stride()
    reference = _TorchSparseSpMM.apply(w1, v1, topo.row_indices, topo.col_indices)
    candidate = TritonSparseSpMM.apply(w2, v2, topo)
    _assert_parity("spmm forward", candidate, reference, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(reference)
    (reference * upstream).sum().backward()
    (candidate * upstream).sum().backward()
    _assert_parity("spmm dP", w2.grad, w1.grad, rtol=3e-4, atol=3e-5)
    _assert_parity("spmm dV", v2.grad, v1.grad, rtol=3e-4, atol=3e-5)


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
    if long_row:
        # Both backends use the same chunked forward. CUDA index_add_ may
        # accumulate edges in different orders, especially for high-degree rows.
        # Measure within-backend variation to contextualize cross-backend parity.
        with torch.no_grad():
            repeats_baseline = [
                baseline(x1.detach(), edge_displacement=d1.detach(), **opts)
                for _ in range(5)
            ]
            repeats_candidate = [
                candidate(x2.detach(), edge_displacement=d2.detach(), **opts)
                for _ in range(5)
            ]
        baseline_jitter = max(
            (value - out1.detach()).abs().max().item()
            for value in repeats_baseline
        )
        candidate_jitter = max(
            (value - out2.detach()).abs().max().item()
            for value in repeats_candidate
        )
        cross_backend_error = (out2.detach() - out1.detach()).abs().max().item()
        print(
            f"[TRITON FORWARD REPEATABILITY] "
            f"baseline_max_abs={baseline_jitter:.6e} "
            f"candidate_max_abs={candidate_jitter:.6e} "
            f"cross_backend_max_abs={cross_backend_error:.6e}",
            flush=True,
        )
    # GH200 repeatability (job 510265): baseline, candidate and cross-backend
    # differences all reached 1.068413e-5 on this high-degree graph.
    # Both backends share the original CUDA index_add_ forward.
    # Keep the direct-kernel and backward gradient tolerances unchanged.
    torch.testing.assert_close(
        out2, out1, rtol=1e-5, atol=2e-5 if long_row else 1e-5,
    )
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
    coords1 = torch.randn(12, 3, device=device, requires_grad=True)
    coords2 = coords1.detach().clone().requires_grad_(True)
    conditioning1 = torch.tensor([[0.2]], device=device, requires_grad=True)
    conditioning2 = conditioning1.detach().clone().requires_grad_(True)
    kwargs = dict(
        edge_index=local, attention_edge_indices={"dilated": dilated},
    )
    x1 = torch.randn(12, 5, device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    reference.train()
    candidate.train()
    out1 = reference(x1, coords=coords1, conditioning=conditioning1, **kwargs)
    out2 = candidate(x2, coords=coords2, conditioning=conditioning2, **kwargs)
    torch.testing.assert_close(out2, out1, rtol=1e-5, atol=1e-5)
    upstream = torch.randn_like(out1)
    (out1 * upstream).sum().backward()
    (out2 * upstream).sum().backward()
    torch.testing.assert_close(x2.grad, x1.grad, rtol=5e-4, atol=5e-5)
    torch.testing.assert_close(coords2.grad, coords1.grad, rtol=5e-4, atol=5e-5)
    torch.testing.assert_close(conditioning2.grad, conditioning1.grad, rtol=5e-4, atol=5e-5)
    candidate_parameters = dict(candidate.named_parameters())
    for name, p1 in reference.named_parameters():
        p2 = candidate_parameters[name]
        if p1.grad is None or p2.grad is None:
            assert p1.grad is None and p2.grad is None, name
        else:
            torch.testing.assert_close(
                p2.grad, p1.grad, rtol=5e-4, atol=5e-5,
                msg=lambda m, name=name: f"checkpoint parameter {name}: {m}",
            )


def test_triton_backend_refuses_missing_cuda_or_triton():
    from graph_attention.models.triton_sparse_backward import triton
    if triton is not None and torch.cuda.is_available():
        require_triton()
        return
    with pytest.raises(RuntimeError, match="requires"):
        require_triton()
