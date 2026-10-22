"""M43C 2-D absolute-coordinate vs relative-geometry-only DiNAT-DiT gates.

These are CPU-small, deterministic, unstructured-graph tests. Production
slice training/generation must run separately on Grace; this suite does not
claim any physical quality result.
"""
from __future__ import annotations

import torch

from graph_attention.geometry.connectivity import exact_two_hop_edge_index
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT


def _graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Two disconnected 3-node chains, deliberately not a Cartesian image.
    coords = torch.tensor([
        [0., 0.], [1., 0.], [2., 0.],
        [10., 0.], [11., 0.], [12., 0.],
    ])
    edges = torch.tensor([
        [0, 1, 1, 2, 3, 4, 4, 5],
        [1, 0, 2, 1, 4, 3, 5, 4],
    ], dtype=torch.long)
    return coords, edges, exact_two_hop_edge_index(edges, 6)


def _model(*, absolute: bool, explicit: bool = True):
    torch.manual_seed(2026)
    arguments = dict(
        in_channels=5, out_channels=5, hidden_dim=16,
        num_heads=2, num_layers=2, spatial_dim=2,
        mlp_ratio=2, conditioning_channels=1, condition_embed_dim=16,
        use_coord_mlp=True, coordinate_normalization="centered_bbox",
        sparse_attention_backend="scatter", dropout=0.0,
    )
    if explicit:
        arguments["use_absolute_position_encoding"] = absolute
    model = AlternatingDilatedGeometricDiT(**arguments)
    with torch.no_grad():
        # adaLN-Zero initialization makes all raw block gates and final
        # output projection zero at t=0. Activate the readout to make
        # an actual coordinate-dependence test rather than testing zeros.
        model.final_layer.projection.weight.normal_(0., 0.1)
        # Activate attention to test that geometric bias still participates.
        for block in model.blocks:
            model_dim = model.hidden_dim
            block.adaLN_modulation[-1].bias[2*model_dim:3*model_dim] = 0.2
    return model


def _forward(model, coords, edges, twohop):
    return model(
        torch.linspace(-0.5, 0.8, 30).reshape(6, 5),
        edge_index=edges,
        coords=coords,
        batch_index=torch.zeros(6, dtype=torch.long),
        conditioning=torch.tensor([[0.55]]),
        attention_edge_indices={"dilated": twohop},
    )


def test_default_is_original_absolute_coordinate_path() -> None:
    coords, edges, twohop = _graph()
    default = _model(absolute=True, explicit=False)
    explicit = _model(absolute=True)
    assert default.use_absolute_position_encoding
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for name, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(
        _forward(default, coords, edges, twohop),
        _forward(explicit, coords, edges, twohop),
        rtol=0, atol=0,
    )


def test_relative_only_retains_identical_initialized_parameters() -> None:
    baseline = _model(absolute=True)
    relative = _model(absolute=False)
    assert not relative.use_absolute_position_encoding
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, relative.state_dict()[name], rtol=0, atol=0)


def test_remote_component_cannot_change_relative_only_local_prediction() -> None:
    coords, edges, twohop = _graph()
    moved = coords.clone()
    moved[3:, 0] += 50.0  # left component's edge displacements unchanged
    absolute = _model(absolute=True).eval()
    relative = _model(absolute=False).eval()
    with torch.no_grad():
        before_relative = _forward(relative, coords, edges, twohop)
        after_relative = _forward(relative, moved, edges, twohop)
        before_absolute = _forward(absolute, coords, edges, twohop)
        after_absolute = _forward(absolute, moved, edges, twohop)
    torch.testing.assert_close(
        before_relative[:3], after_relative[:3], rtol=0, atol=1.e-6,
    )
    assert not torch.allclose(before_absolute[:3], after_absolute[:3],
                              atol=1.e-6, rtol=0)


def test_relative_only_keeps_geometry_attention_but_skips_position_gradients() -> None:
    coords, edges, twohop = _graph()
    model = _model(absolute=False).train()
    seen_displacements = []
    hook = model.blocks[0].attention.geometry_mlp.register_forward_hook(
        lambda _module, inputs, _outputs: seen_displacements.append(
            inputs[0].detach().clone()
        )
    )
    try:
        _forward(model, coords, edges, twohop).square().sum().backward()
    finally:
        hook.remove()
    assert len(seen_displacements) == 1
    assert seen_displacements[0].shape == (edges.shape[1], 2)
    assert not torch.allclose(
        seen_displacements[0], torch.zeros_like(seen_displacements[0])
    )
    assert all(p.grad is None for p in model.coordinate_projection.parameters())
    assert any(
        p.grad is not None for p in model.blocks[0].attention.parameters()
    )


def test_absolute_path_uses_coordinate_embedding_weights() -> None:
    coords, edges, twohop = _graph()
    model = _model(absolute=True).train()
    _forward(model, coords, edges, twohop).square().sum().backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.coordinate_projection.parameters()
    )


def test_relative_only_can_evaluate_with_missing_absolute_bbox() -> None:
    # A graph with constant coordinates has an invalid physical mesh, but the
    # coordinate pathway bypass should not depend on bounding-box spans.
    # Edge displacements are still valid finite vectors.
    coords, edges, twohop = _graph()
    constant = torch.zeros_like(coords)
    model = _model(absolute=False)
    output = _forward(model, constant, edges, twohop)
    assert output.shape == (6, 5)
    assert torch.isfinite(output).all()
