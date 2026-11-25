"""Full DiT block checkpointing parity on packed graphs."""
import pytest
import torch

from graph_attention.models.full_dit import FullDiTGraphTransformer


@pytest.mark.parametrize("sizes", [(7,), (5, 5), (4, 7)])
def test_full_checkpointing_parity(sizes):
    torch.manual_seed(42)
    common = dict(in_channels=5, out_channels=5, hidden_dim=16, num_heads=4,
                  num_layers=2, spatial_dim=3, conditioning_channels=1,
                  condition_embed_dim=16, dropout=0.1)
    plain = FullDiTGraphTransformer(**common)
    remat = FullDiTGraphTransformer(**common, activation_checkpointing=True)
    # Avoid zero-init masking attention/MLP gradients.
    with torch.no_grad():
        for block in plain.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=0.05)
            block.adaLN_modulation[-1].bias.normal_(std=0.05)
        plain.final_layer.projection.weight.normal_(std=0.05)
    remat.load_state_dict(plain.state_dict())
    n = sum(sizes)
    x = torch.randn(n, 5)
    kwargs = dict(coords=torch.randn(n, 3), edge_index=torch.empty(2, 0, dtype=torch.long),
                  batch_index=torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes)),
                  conditioning=torch.randn(len(sizes), 1))
    target = torch.randn_like(x)
    results = []
    for model in (plain, remat):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.manual_seed(123)
        inputs = x.clone().requires_grad_()
        out = model(inputs, **kwargs)
        (out-target).square().mean().backward()
        grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
        optimizer.step()
        results.append((out.detach(), inputs.grad, grads))
    for a, b in zip(results[0][:2], results[1][:2]):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for name in results[0][2]:
        torch.testing.assert_close(results[0][2][name], results[1][2][name], atol=1e-6, rtol=1e-5)
    for name, param in plain.named_parameters():
        torch.testing.assert_close(param, dict(remat.named_parameters())[name], atol=1e-6, rtol=1e-5)
    plain.eval()
    remat.eval()
    with torch.no_grad():
        torch.testing.assert_close(plain(x, **kwargs), remat(x, **kwargs))
