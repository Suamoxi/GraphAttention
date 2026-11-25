"""Check Full DiT autocast with nonzero weights and FP32 master gradients."""
import pytest
import torch

from graph_attention.models.full_dit import FullDiTGraphTransformer, FullDiTMultiheadAttention


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_dit_bf16_amp_finite(device):
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        pytest.skip("CUDA BF16 unavailable")
    torch.manual_seed(42)
    model = FullDiTGraphTransformer(5, 5, 32, 4, 2, 3, conditioning_channels=1).to(device)
    with torch.no_grad():
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=.05)
        model.final_layer.projection.weight.normal_(std=.05)
    x = torch.randn(12, 5, device=device)
    kwargs = dict(coords=torch.randn(12, 3, device=device),
                  edge_index=torch.empty(2, 0, dtype=torch.long, device=device),
                  batch_index=torch.arange(2, device=device).repeat_interleave(6),
                  conditioning=torch.randn(2, 1, device=device))
    with torch.autocast(device, dtype=torch.bfloat16):
        pred = model(x, **kwargs)
    pred.float().square().mean().backward()
    assert pred.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 and p.grad is not None and
               p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all() for p in model.parameters())


def test_full_attention_rejects_mixed_dtype_without_autocast():
    attn = FullDiTMultiheadAttention(32, 4)
    with pytest.raises(TypeError, match="expected model dtype"):
        attn(torch.randn(5, 32).bfloat16(), edge_index=torch.empty(2, 0, dtype=torch.long), batch_index=None)
