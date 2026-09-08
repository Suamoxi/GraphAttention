import torch

from graph_attention.geometry import exact_two_hop_edge_index
from graph_attention.models import (
    AlternatingDilatedGeometricSparseGraphTransformer,
    GeometricSparseGraphTransformer,
)


def _directed_chain(num_nodes: int) -> torch.Tensor:
    forward = torch.arange(num_nodes - 1, dtype=torch.long)
    reverse = forward + 1
    return torch.stack(
        (
            torch.cat((forward, reverse)),
            torch.cat((reverse, forward)),
        )
    )


def test_alternating_dilated_transformer_uses_local_then_dilated_layers() -> None:
    torch.manual_seed(3)
    model = AlternatingDilatedGeometricSparseGraphTransformer(
        in_channels=2,
        out_channels=1,
        hidden_dim=16,
        num_heads=4,
        num_layers=4,
        spatial_dim=2,
        mlp_ratio=2,
    )
    inputs = torch.randn(6, 2)
    coords = torch.randn(6, 2)
    local = _directed_chain(6)
    dilated = exact_two_hop_edge_index(local, num_nodes=6)

    seen: list[torch.Tensor] = []
    handles = [
        block.register_forward_pre_hook(
            lambda _module, args: seen.append(args[1].detach().clone())
        )
        for block in model.blocks
    ]
    try:
        output = model(
            inputs,
            edge_index=local,
            coords=coords,
            attention_edge_indices={"dilated": dilated},
        )
    finally:
        for handle in handles:
            handle.remove()

    assert output.shape == (6, 1)
    assert len(seen) == 4
    assert torch.equal(seen[0], local)
    assert torch.equal(seen[1], dilated)
    assert torch.equal(seen[2], local)
    assert torch.equal(seen[3], dilated)


def test_alternating_dilated_transformer_requires_external_dilated_topology() -> None:
    model = AlternatingDilatedGeometricSparseGraphTransformer(
        in_channels=2,
        out_channels=1,
        hidden_dim=8,
        num_heads=2,
        num_layers=2,
        spatial_dim=2,
    )
    inputs = torch.randn(4, 2)
    coords = torch.randn(4, 2)
    local = _directed_chain(4)

    try:
        model(inputs, edge_index=local, coords=coords)
    except ValueError as exc:
        assert "externally supplied" in str(exc)
    else:
        raise AssertionError("missing dilated topology should fail")


def test_alternating_model_has_same_parameterization_as_m9() -> None:
    kwargs = {
        "in_channels": 2,
        "out_channels": 3,
        "hidden_dim": 16,
        "num_heads": 4,
        "num_layers": 4,
        "spatial_dim": 2,
        "mlp_ratio": 2,
        "conditioning_channels": 1,
    }
    torch.manual_seed(11)
    m9 = GeometricSparseGraphTransformer(**kwargs)
    torch.manual_seed(12)
    dilated = AlternatingDilatedGeometricSparseGraphTransformer(**kwargs)

    incompatible = dilated.load_state_dict(m9.state_dict(), strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert sum(parameter.numel() for parameter in dilated.parameters()) == sum(
        parameter.numel() for parameter in m9.parameters()
    )


def test_alternating_dilated_transformer_is_translation_invariant() -> None:
    torch.manual_seed(17)
    model = AlternatingDilatedGeometricSparseGraphTransformer(
        in_channels=2,
        out_channels=1,
        hidden_dim=16,
        num_heads=4,
        num_layers=4,
        spatial_dim=2,
        mlp_ratio=2,
    )
    inputs = torch.randn(7, 2)
    coords = torch.randn(7, 2)
    local = _directed_chain(7)
    dilated = exact_two_hop_edge_index(local, num_nodes=7)

    reference = model(
        inputs,
        edge_index=local,
        coords=coords,
        attention_edge_indices={"dilated": dilated},
    )
    translated = model(
        inputs,
        edge_index=local,
        coords=coords + torch.tensor([4.0, -2.0]),
        attention_edge_indices={"dilated": dilated},
    )

    torch.testing.assert_close(reference, translated, rtol=1e-5, atol=1e-6)
