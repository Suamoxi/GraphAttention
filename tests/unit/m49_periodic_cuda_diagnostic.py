"""Small-mesh diagnosis of M49.6 CUDA forward consistency.

Diagnostic only: compare identical model calls and existing sparse operators
against a direct FP64 edge reference. Never alter the production tolerance,
weights, topology, source, or precision policy. Used by the periodic tests.
"""
from collections.abc import Mapping
import json
import os

import torch

from graph_attention.models.sparse_transformer import (
    _build_torch_sparse_topology, _TorchSparseSDDMM, _TorchSparseSpMM,
)
from graph_attention.models.triton_sparse_backward import TritonSparseSDDMM, TritonSparseSpMM


def assert_identical_arguments(a, b, path="model"):
    """Check exact tensor values and scalar metadata, not output repeatability."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), path
        assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device, path
        assert torch.equal(a, b), f"different tensor argument: {path}"
    elif isinstance(a, Mapping):
        assert isinstance(b, Mapping) and set(a) == set(b), path
        for key in a:
            assert_identical_arguments(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_identical_arguments(x, y, f"{path}[{i}]")
    else:
        assert type(a) is type(b) and a == b, f"different argument: {path}"


def _error(actual, reference):
    actual, reference = actual.double(), reference.double()
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        return {"finite": False}
    error = actual - reference
    norm = torch.linalg.vector_norm(reference).item()
    return {
        "finite": True,
        "max_abs": error.abs().max().item(),
        "rms": error.square().mean().sqrt().item(),
        "relative_l2": torch.linalg.vector_norm(error).item()/norm if norm else None,
        "outside_original_gate": int((error.abs() > 3e-5+1e-4*reference.abs()).sum().item()),
        "elements": actual.numel(),
    }


def _repeat(outs):
    return [_error(out, outs[0]) for out in outs[1:]]


def _emit(kind, **payload):
    print("[M49 PERIODIC CUDA DIAGNOSTIC] " + json.dumps(
        dict(kind=kind, **payload), sort_keys=True, allow_nan=False,
    ), flush=True)


@torch.no_grad()
def diagnose_forward(model, call, original, sampled):
    """Inspect the exact failing model on its 512-node synthetic torus."""
    args, kwargs = call
    assert args[0].shape[0] == 512, "diagnostic uses bounded synthetic graph only"
    _emit("runtime", torch_version=str(torch.__version__),
          cuda_version=torch.version.cuda,
          gpu=torch.cuda.get_device_name(args[0].device),
          matmul_precision=torch.get_float32_matmul_precision(),
          allow_tf32=torch.backends.cuda.matmul.allow_tf32,
          cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
          deterministic=torch.are_deterministic_algorithms_enabled(),
          identical_model_arguments=True,
          original_pair=_error(sampled, original))

    # Fixed model/input, unchanged cached topology. Save small per-layer
    # projections and outputs to locate the first divergent operation.
    records = []
    captured = {}
    hooks = []
    for i, block in enumerate(model.blocks):
        for name, module in (
            ("qkv", block.attention.qkv),
            ("geometry", block.attention.geometry_mlp),
            ("output_projection", block.attention.out_proj),
            ("attention", block.attention),
            ("block", block),
        ):
            def save(_mod, _args, output, *, key=f"{i}.{name}"):
                captured[key] = output.detach().clone()
            hooks.append(module.register_forward_hook(save))
        # Compare the actual input to the output projection (the sparse
        # attention aggregation), separately from the dense projection.
        def save_aggregated(_mod, inputs, *, key=f"{i}.pre_output_projection"):
            captured[key] = inputs[0].detach().clone()
        hooks.append(block.attention.out_proj.register_forward_pre_hook(save_aggregated))
    outs = []
    try:
        for _ in range(3):
            captured.clear()
            outs.append(model(*args, **kwargs).detach().clone())
            records.append(dict(captured))
    finally:
        for hook in hooks:
            hook.remove()
    _emit("identical_forward_repeats", versus_first_forward=_repeat(outs),
          versus_original=[_error(v, original) for v in outs],
          layer_repeats={k:_repeat([r[k] for r in records]) for k in records[0]})

    # Test whether the first dense output projection amplifies tiny sparse
    # reduction differences. Hold the actual captured operands and weights
    # fixed across the precision arms, and use FP64 as a numerical reference.
    projection = model.blocks[0].attention.out_proj
    inputs = [r["0.pre_output_projection"] for r in records]
    weight = projection.weight.detach()
    bias = projection.bias.detach() if projection.bias is not None else None
    refs = [torch.nn.functional.linear(x.double(), weight.double(),
                                     None if bias is None else bias.double())
            for x in inputs]
    previous_precision = torch.get_float32_matmul_precision()
    try:
        for precision in ("high", "highest"):
            torch.set_float32_matmul_precision(precision)
            outputs = [torch.nn.functional.linear(x, weight, bias) for x in inputs]
            _emit("fixed_dense_projection", precision=precision,
                  allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  input_repeats=_repeat(inputs),
                  fp64_output_repeats=_repeat(refs),
                  output_repeats=_repeat(outputs),
                  output_vs_fp64=[_error(a,b) for a,b in zip(outputs,refs,strict=True)])
    finally:
        torch.set_float32_matmul_precision(previous_precision)

    # Both torch_sparse variants share the cuSPARSE SDDMM and bounded
    # index_add_ SpMM forward. Freeze Q/K/V to remove upstream variations.
    edge_names = {1: "local", 2: "dilated", 3: "dilated_3"}
    for i, block in enumerate(model.blocks):
        attention = block.attention
        hop = model.attention_hop_schedule[i % len(model.attention_hop_schedule)]
        edges = kwargs["edge_index"] if hop == 1 else kwargs["attention_edge_indices"][edge_names[hop]]
        top = _build_torch_sparse_topology(edges, num_nodes=args[0].shape[0])
        qkv = records[0][f"{i}.qkv"].reshape(
            top.num_nodes, 3, attention.num_heads, attention.head_dim,
        )
        q, k, v = qkv.unbind(1)
        row, col = top.row_indices, top.col_indices
        # The test graph is small; FP64 edge expansions are diagnostic
        # references only, not candidate production implementations.
        reference_scores = (q[row].double()*k[col].double()).sum(-1)*attention.scale
        native = [_TorchSparseSDDMM.apply(
            q, k, top.crow_indices, row, col, attention.scale
        ) for _ in range(3)]
        triton = [TritonSparseSDDMM.apply(q,k,top,attention.scale) for _ in range(3)]
        biased = reference_scores + records[0][f"{i}.geometry"][top.edge_order].double()
        by_head = row[:,None].expand(-1, attention.num_heads)
        maximum = torch.full((top.num_nodes, attention.num_heads), -torch.inf,
                             dtype=torch.float64, device=q.device)
        maximum.scatter_reduce_(0, by_head, biased, reduce="amax", include_self=True)
        exp = (biased-maximum[row]).exp()
        denominator = torch.zeros_like(maximum).scatter_add_(0, by_head, exp)
        exp32 = exp.float()
        denom32 = [torch.zeros_like(maximum, dtype=torch.float32).scatter_add_(0,by_head,exp32)
                   for _ in range(3)]
        # Use the very same fixed FP32 weights in each SpMM call.
        weights = (exp/denominator[row]).float()
        reference_agg = torch.zeros_like(v, dtype=torch.float64).index_add_(
            0, row, weights.double().unsqueeze(-1)*v[col].double(),
        )
        spmm_native = [_TorchSparseSpMM.apply(weights,v,row,col) for _ in range(3)]
        spmm_triton = [TritonSparseSpMM.apply(weights,v,top) for _ in range(3)]
        _emit("fixed_operand_primitives", layer=i, hop=hop, nodes=top.num_nodes,
              edges=edges.shape[1],
              sddmm_native_vs_fp64=[_error(a,reference_scores) for a in native],
              sddmm_triton_vs_fp64=[_error(a,reference_scores) for a in triton],
              sddmm_native_repeat=_repeat(native), sddmm_triton_repeat=_repeat(triton),
              softmax_denominator_repeat=_repeat(denom32),
              softmax_denominator_vs_fp64=[_error(a,denominator) for a in denom32],
              spmm_native_vs_fp64=[_error(a,reference_agg) for a in spmm_native],
              spmm_triton_vs_fp64=[_error(a,reference_agg) for a in spmm_triton],
              spmm_native_repeat=_repeat(spmm_native), spmm_triton_repeat=_repeat(spmm_triton))

    # Same model parameters and segmented modulation, with the reference
    # scatter attention. Restore all backend flags even if it raises.
    original_backend = model.sparse_attention_backend
    original_attn = [b.attention.sparse_attention_backend for b in model.blocks]
    try:
        model.sparse_attention_backend = "scatter"
        for b in model.blocks:
            b.attention.sparse_attention_backend = "scatter"
        reference = [model(*args, **kwargs).detach().clone() for _ in range(3)]
        _emit("scatter_reference", repeats=_repeat(reference),
              original_vs_scatter=_error(original,reference[0]),
              sampled_vs_scatter=_error(sampled,reference[0]),
              triton_vs_scatter=[_error(v,reference[0]) for v in outs])
    finally:
        model.sparse_attention_backend = original_backend
        for block, backend in zip(model.blocks,original_attn,strict=True):
            block.attention.sparse_attention_backend = backend

    # Probe strict deterministic CUDA execution as a diagnostic, not as an
    # authorization to change training or silently relax the original check.
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        values = [model(*args,**kwargs).detach().clone() for _ in range(3)]
        _emit("deterministic_mode", repeats=_repeat(values),
              versus_original=_error(values[0],original))
    except RuntimeError as exc:
        _emit("deterministic_mode_unavailable", error=str(exc))
    finally:
        torch.use_deterministic_algorithms(was_deterministic,warn_only=was_warn_only)
