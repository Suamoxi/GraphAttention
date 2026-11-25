"""Check the FP32 sparse-attention island and BF16 surrounding DiNAT blocks."""
import pytest
import torch

from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT


@pytest.mark.parametrize("checkpointing", [False, True])
def test_dinat_hybrid_amp_dtypes_and_gradients(checkpointing):
    pytest.importorskip("triton")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 unavailable")
    device = torch.device("cuda")
    torch.manual_seed(42)
    model = AlternatingDilatedGeometricDiT(
        5, 5, 32, 4, 2, 3,
        conditioning_channels=1,
        sparse_attention_backend="torch_sparse_triton",
        activation_checkpointing=checkpointing,
    ).to(device)
    with torch.no_grad():
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=.05)
        model.final_layer.projection.weight.normal_(std=.05)

    nodes = torch.arange(12, device=device)
    offset = (nodes // 6) * 6
    local = torch.stack((nodes, offset + (nodes + 1) % 6))
    dilated = torch.stack((nodes, offset + (nodes + 2) % 6))
    x = torch.randn(12, 5, device=device)
    kwargs = dict(
        coords=torch.randn(12, 3, device=device),
        edge_index=local, attention_edge_indices={"dilated": dilated},
        batch_index=nodes // 6, graph_sizes=(6, 6),
        conditioning=torch.randn(2, 1, device=device),
    )
    qkv_dtypes, mlp_dtypes = [], []
    def observe_qkv(module, args, output):
        qkv_dtypes.append(output.dtype)
        assert not torch.is_autocast_enabled("cuda")
    def observe_mlp(module, args, output):
        mlp_dtypes.append(output.dtype)
    handles = []
    for block in model.blocks:
        handles.append(block.attention.qkv.register_forward_hook(observe_qkv))
        handles.append(block.mlp[0].register_forward_hook(observe_mlp))
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x, **kwargs)
        assert pred.dtype == torch.bfloat16
        pred.float().square().mean().backward()
        assert qkv_dtypes and all(d == torch.float32 for d in qkv_dtypes)
        assert mlp_dtypes and all(d == torch.bfloat16 for d in mlp_dtypes)
        assert all(p.dtype == torch.float32 and p.grad is not None and
                   p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        assert model.blocks[0].attention.qkv.weight.grad.norm() > 0
    finally:
        for handle in handles:
            handle.remove()
    model.eval()
    with torch.no_grad():
        ref = model(x, **kwargs)
        assert ref.dtype == torch.float32
        assert torch.isfinite(ref).all()
