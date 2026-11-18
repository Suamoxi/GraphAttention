"""Optional FP32 Triton backward for the existing CSR sparse attention.

The forward SDDMM, sparse softmax, and chunked SpMM are unchanged. During
backpropagation, bounded-size Triton programs reduce each CSR row into dQ,
dK and dV without Python edge-chunk loops, global atomics, or E x H x D
message buffers. Transposed CSR handles in-degree and out-degree imbalance.

This is experimental and must not be selected by the frozen M47.1/M47.4
training recipes until GH200 gradient/memory/latency parity is established.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def require_triton() -> None:
    """Fail explicitly: no silent fallback that would mislabel measurements."""
    if triton is None:
        raise RuntimeError(
            "torch_sparse_triton requires Triton in the active CUDA Python environment"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("torch_sparse_triton requires a CUDA device")


if triton is not None:

    @triton.jit
    def _csr_rows_reduce(
        ROW_PTR, COLS, ORDER, EDGE_WEIGHTS, FEATURES, OUTPUT,
        N: tl.constexpr, D: tl.constexpr,
        W_S0: tl.constexpr, W_S1: tl.constexpr,
        F_S0: tl.constexpr, F_S1: tl.constexpr, F_S2: tl.constexpr,
        O_S0: tl.constexpr, O_S1: tl.constexpr, O_S2: tl.constexpr,
        SCALE: tl.constexpr, TRANSPOSED: tl.constexpr,
        TILE_ROWS: tl.constexpr, TILE_EDGES: tl.constexpr, TILE_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
        head = tl.program_id(1)
        features = tl.arange(0, TILE_D)
        edge_offsets = tl.arange(0, TILE_EDGES)

        starts = tl.load(ROW_PTR + rows, rows < N, other=0)
        ends = tl.load(ROW_PTR + rows + 1, rows < N, other=0)
        max_degree = tl.max(ends - starts, 0)
        total = tl.full((TILE_ROWS, TILE_D), 0.0, tl.float32)

        # Dynamic row lengths work for unstructured graphs and isolated nodes.
        for edge_block in range(tl.cdiv(max_degree, TILE_EDGES)):
            positions = starts[:, None] + edge_block * TILE_EDGES + edge_offsets[None, :]
            active = (rows[:, None] < N) & (positions < ends[:, None])
            neighbours = tl.load(COLS + positions, active, other=0)
            if TRANSPOSED:
                weight_positions = tl.load(ORDER + positions, active, other=0)
            else:
                weight_positions = positions
            weights = tl.load(
                EDGE_WEIGHTS + weight_positions * W_S0 + head * W_S1,
                active, other=0.0,
            )
            values = tl.load(
                FEATURES + neighbours[:, :, None] * F_S0 + head * F_S1
                + features[None, None, :] * F_S2,
                active[:, :, None] & (features[None, None, :] < D),
                other=0.0,
            )
            total += tl.sum(weights[:, :, None] * values, 1)

        tl.store(
            OUTPUT + rows[:, None] * O_S0 + head * O_S1
            + features[None, :] * O_S2,
            total * SCALE,
            (rows[:, None] < N) & (features[None, :] < D),
        )


    @triton.jit
    def _csr_edge_dot(
        COLS, ROWS, GRAD_OUTPUT, VALUES, GRAD_WEIGHTS,
        E: tl.constexpr, D: tl.constexpr,
        G_S0: tl.constexpr, G_S1: tl.constexpr, G_S2: tl.constexpr,
        V_S0: tl.constexpr, V_S1: tl.constexpr, V_S2: tl.constexpr,
        W_S0: tl.constexpr, W_S1: tl.constexpr,
        TILE_EDGES: tl.constexpr, TILE_D: tl.constexpr,
    ):
        edges = tl.program_id(0) * TILE_EDGES + tl.arange(0, TILE_EDGES)
        head = tl.program_id(1)
        d = tl.arange(0, TILE_D)
        mask = edges < E
        src = tl.load(COLS + edges, mask, other=0)
        dst = tl.load(ROWS + edges, mask, other=0)
        grad = tl.load(
            GRAD_OUTPUT + dst[:, None] * G_S0 + head * G_S1 + d[None, :] * G_S2,
            mask[:, None] & (d[None, :] < D), other=0.0,
        )
        values = tl.load(
            VALUES + src[:, None] * V_S0 + head * V_S1 + d[None, :] * V_S2,
            mask[:, None] & (d[None, :] < D), other=0.0,
        )
        dot = tl.sum(grad * values, 1)
        tl.store(GRAD_WEIGHTS + edges * W_S0 + head * W_S1, dot, mask)


def _row_reduce(
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    permutation: torch.Tensor,
    weights: torch.Tensor,
    features: torch.Tensor,
    *,
    scale: float = 1.0,
    transposed: bool,
) -> torch.Tensor:
    require_triton()
    if features.dtype != torch.float32 or weights.dtype != torch.float32:
        raise TypeError("Triton sparse backward currently supports FP32 only")
    nodes, heads, dim = features.shape
    if nodes == 0 or dim < 1:
        raise ValueError("empty node/feature dimension not supported")
    # A small row tile bounds peak program storage independent of graph size.
    tile_rows, tile_edges = 4, 32
    output = torch.empty_like(features)
    _csr_rows_reduce[(triton.cdiv(nodes, tile_rows), heads)](
        row_ptr, col_indices, permutation, weights, features, output,
        nodes, dim, *weights.stride(), *features.stride(), *output.stride(),
        float(scale), transposed,
        tile_rows, tile_edges, triton.next_power_of_2(dim),
        num_warps=4,
    )
    return output


class TritonSparseSDDMM(torch.autograd.Function):
    """Use original CSR SDDMM forward and bounded Triton dQ/dK backward."""

    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, topology, scale: float):
        require_triton()
        from .sparse_transformer import _TorchSparseSDDMM

        scores = _TorchSparseSDDMM.forward(
            ctx, query, key,
            topology.crow_indices, topology.row_indices, topology.col_indices,
            scale,
        )
        ctx.save_for_backward(
            query, key,
            topology.crow_indices, topology.col_indices,
            topology.transpose_crow_indices, topology.transpose_col_indices,
            topology.transpose_order,
        )
        ctx.scale = float(scale)
        return scores

    @staticmethod
    def backward(ctx, grad_scores: torch.Tensor):
        (query, key, crow, cols, tcrow, tcols, order) = ctx.saved_tensors
        grad_scores = grad_scores.contiguous()
        grad_query = _row_reduce(
            crow, cols, order, grad_scores, key,
            scale=ctx.scale, transposed=False,
        )
        grad_key = _row_reduce(
            tcrow, tcols, order, grad_scores, query,
            scale=ctx.scale, transposed=True,
        )
        return grad_query, grad_key, None, None


class TritonSparseSpMM(torch.autograd.Function):
    """Use original bounded SpMM forward and Triton dWeights/dV backward."""

    @staticmethod
    def forward(ctx, weights: torch.Tensor, value: torch.Tensor, topology):
        require_triton()
        from .sparse_transformer import _TorchSparseSpMM

        output = _TorchSparseSpMM.forward(
            ctx, weights, value, topology.row_indices, topology.col_indices,
        )
        ctx.save_for_backward(
            weights, value,
            topology.row_indices, topology.col_indices,
            topology.transpose_crow_indices, topology.transpose_col_indices,
            topology.transpose_order,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weights, value, rows, cols, tcrow, tcols, order = ctx.saved_tensors
        nodes, heads, dim = value.shape
        edges = weights.shape[0]
        grad_weights = torch.empty_like(weights)
        if edges:
            tile_edges = 64
            _csr_edge_dot[(triton.cdiv(edges, tile_edges), heads)](
                cols, rows, grad_output, value, grad_weights,
                edges, dim, *grad_output.stride(), *value.stride(),
                *grad_weights.stride(), tile_edges, triton.next_power_of_2(dim),
                num_warps=4,
            )
        grad_value = _row_reduce(
            tcrow, tcols, order, weights, grad_output,
            transposed=True,
        )
        return grad_weights, grad_value, None
