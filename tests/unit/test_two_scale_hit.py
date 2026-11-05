"""Small-graph qualification gates for static/dynamic two-scale coupling."""

import pytest
import torch

from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from graph_attention.geometry.two_scale_partition import two_scale_patch_dependencies
from graph_attention.models.two_scale_dinat import SparseCrossScaleAttention, TwoScaleDiNAT


@pytest.mark.parametrize("r", [2, 3])
def test_periodic_hit_mesh_and_cross_scale_support(r):
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=r)
    n_f = (3*r)**3
    assert mesh.fine_coords.shape == (n_f, 3)
    assert mesh.coarse_edges.shape[1] == 6*3**3
    assert mesh.fine_edges.shape[1] == 6*n_f
    target = mesh.coarse_to_fine[1]
    weights = torch.zeros(n_f).index_add_(0, target, mesh.cross_weights)
    torch.testing.assert_close(weights, torch.ones_like(weights))
    assert torch.equal(mesh.fine_to_coarse, mesh.coarse_to_fine.flip(0))
    # Exact coarse lattice nodes are carried exactly by the geometry operator.
    anchor = torch.arange(n_f).reshape(3*r, 3*r, 3*r)[::r, ::r, ::r].reshape(-1)
    anchored = torch.isin(target, anchor)
    assert torch.all(mesh.cross_weights[anchored] == 1)


def test_sparse_cross_attention_recovers_normalized_fixed_lifting():
    module = SparseCrossScaleAttention(4, 4, 4, heads=1)
    for layer in (module.q, module.k):
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    for layer in module.geo:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    for layer in (module.v, module.out):
        torch.nn.init.eye_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    keys = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    queries = torch.zeros((2, 4))
    edges = torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.long)
    weights = torch.tensor([0.25, 0.75, 1.])
    positions = torch.zeros((2, 3))
    y = module(queries, keys, query_coords=positions, key_coords=positions,
               edges=edges, weights=weights, coarse_spacing=1., periodic_length=1.)
    torch.testing.assert_close(y[0], .25 * keys[0] + .75 * keys[1])
    torch.testing.assert_close(y[1], keys[1])


def test_dynamic_patch_dependency_includes_cross_scale_both_directions():
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    a = two_scale_patch_dependencies(
        torch.tensor([0, 1, 2]),
        fine_local=mesh.fine_edges, fine_dilated=mesh.fine_dilated,
        coarse_local=mesh.coarse_edges, coarse_dilated=mesh.coarse_dilated,
        coarse_to_fine=mesh.coarse_to_fine, fine_to_coarse=mesh.fine_to_coarse,
        fine_pre_layers=1, fine_post_layers=1, coarse_static_layers=1,
        coarse_dynamic_layers=1,
    )
    s = two_scale_patch_dependencies(
        torch.tensor([0, 1, 2]),
        fine_local=mesh.fine_edges, fine_dilated=mesh.fine_dilated,
        coarse_local=mesh.coarse_edges, coarse_dilated=mesh.coarse_dilated,
        coarse_to_fine=mesh.coarse_to_fine, fine_to_coarse=mesh.fine_to_coarse,
        fine_pre_layers=1, fine_post_layers=1, coarse_static_layers=1,
    )
    assert torch.isin(a.owned_fine, a.fine_read).all()
    assert torch.isin(a.coarse_up_keys, a.coarse_read).all()
    assert len(a.fine_read) >= len(s.fine_read)
    assert len(a.coarse_read) >= len(s.coarse_read)


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_two_scale_forward_and_coarse_gradients(mode):
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    model = TwoScaleDiNAT(
        mode=mode, coarse_width=16, fine_width=16, num_heads=4,
        fine_pre_layers=1, fine_post_layers=1,
        coarse_static_layers=1, coarse_dynamic_layers=1,
    )
    c = torch.randn((mesh.coarse_coords.shape[0], 5), requires_grad=True)
    x = torch.randn((mesh.fine_coords.shape[0], 5), requires_grad=True)
    y = model(
        x, baseline=torch.randn_like(x), coarse_state=c,
        fine_coords=mesh.fine_coords, coarse_coords=mesh.coarse_coords,
        fine_edges=mesh.fine_edges, coarse_edges=mesh.coarse_edges,
        fine_dilated=mesh.fine_dilated, coarse_dilated=mesh.coarse_dilated,
        coarse_to_fine=mesh.coarse_to_fine, fine_to_coarse=mesh.fine_to_coarse,
        cross_weights=mesh.cross_weights, refinement=2, time=torch.tensor([0.5]),
        periodic_length=1.0, coarse_spacing=mesh.coarse_spacing,
    )
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert c.grad is not None and torch.isfinite(c.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert model.coarse_encoder.final_layer.projection.weight.grad is not None
    if mode == 'dynamic':
        assert model.down.q.weight.grad is not None
        assert model.coarse_dynamic.final_layer.projection.weight.grad is not None


def test_refinement_embedding_depends_only_on_ratio():
    time = torch.tensor([0.1, 0.9])
    torch.testing.assert_close(
        TwoScaleDiNAT._scale(torch.tensor([2, 4]), time=time),
        torch.tensor([[1.], [2.]]),
    )
    with pytest.raises(ValueError):
        TwoScaleDiNAT._scale(1, time=time)


def test_patch_extract_preserves_global_ids_and_bipartite_weights():
    from graph_attention.geometry.two_scale_partition import extract_two_scale_patch_geometry
    mesh = build_hit_two_scale_geometry(coarse_side=4, refinement=2)
    owned = torch.tensor([7, 8, 9], dtype=torch.long)
    plan = two_scale_patch_dependencies(
        owned, fine_local=mesh.fine_edges, fine_dilated=mesh.fine_dilated,
        coarse_local=mesh.coarse_edges, coarse_dilated=mesh.coarse_dilated,
        coarse_to_fine=mesh.coarse_to_fine, fine_to_coarse=mesh.fine_to_coarse,
        fine_pre_layers=1, fine_post_layers=1,
        coarse_static_layers=1, coarse_dynamic_layers=1,
    )
    patch = extract_two_scale_patch_geometry(mesh, plan)
    torch.testing.assert_close(patch.fine_global[patch.owned_fine_local], owned)
    torch.testing.assert_close(
        patch.coarse_global[patch.coarse_to_fine[0]],
        mesh.coarse_to_fine[0, torch.isin(mesh.coarse_to_fine[0], patch.coarse_global)
                               & torch.isin(mesh.coarse_to_fine[1], patch.fine_global)],
    )


@pytest.mark.parametrize("mode", ["static", "dynamic"])
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_owned_velocities_match_global_forward_after_dependency_closure(mode, depth):
    """Real M37 integration gate (requires the installed GraphAttention repo)."""
    from graph_attention.geometry.two_scale_partition import extract_two_scale_patch_geometry

    mesh = build_hit_two_scale_geometry(coarse_side=4, refinement=2)
    torch.manual_seed(17)
    model = TwoScaleDiNAT(
        mode=mode, coarse_width=16, fine_width=16, num_heads=4,
        fine_pre_layers=depth, fine_post_layers=depth,
        coarse_static_layers=depth, coarse_dynamic_layers=depth,
    ).eval()
    # Expose attention dependencies rather than only the initialized zero
    # velocity head or initially closed adaLN residual gates.
    with torch.no_grad():
        blocks = [model.coarse_encoder, model.fine_encoder, model.fine_decoder]
        if mode == 'dynamic':
            blocks.append(model.coarse_dynamic)
        for backbone in blocks:
            for block in backbone.blocks:
                torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
                torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)
            torch.nn.init.normal_(backbone.final_layer.projection.weight, std=0.02)
    state = torch.randn((mesh.fine_coords.shape[0], 5))
    baseline = torch.randn_like(state)
    coarse = torch.randn((mesh.coarse_coords.shape[0], 5))
    t = torch.tensor([0.37])
    common = dict(refinement=mesh.refinement, time=t, periodic_length=mesh.periodic_length,
                  coarse_spacing=mesh.coarse_spacing)
    whole = model(
        state, baseline=baseline, coarse_state=coarse,
        fine_coords=mesh.fine_coords, coarse_coords=mesh.coarse_coords,
        fine_edges=mesh.fine_edges, fine_dilated=mesh.fine_dilated,
        coarse_edges=mesh.coarse_edges, coarse_dilated=mesh.coarse_dilated,
        coarse_to_fine=mesh.coarse_to_fine, fine_to_coarse=mesh.fine_to_coarse,
        cross_weights=mesh.cross_weights, **common,
    )
    owned = torch.tensor([40, 41, 42], dtype=torch.long)
    plan = two_scale_patch_dependencies(
        owned, fine_local=mesh.fine_edges, fine_dilated=mesh.fine_dilated,
        coarse_local=mesh.coarse_edges, coarse_dilated=mesh.coarse_dilated,
        coarse_to_fine=mesh.coarse_to_fine, fine_to_coarse=mesh.fine_to_coarse,
        fine_pre_layers=depth, fine_post_layers=depth,
        coarse_static_layers=depth,
        coarse_dynamic_layers=depth * int(mode == 'dynamic'),
    )
    patch = extract_two_scale_patch_geometry(mesh, plan)
    local = model(
        state[patch.fine_global], baseline=baseline[patch.fine_global],
        coarse_state=coarse[patch.coarse_global],
        fine_coords=patch.fine_coords, coarse_coords=patch.coarse_coords,
        fine_edges=patch.fine_edges, fine_dilated=patch.fine_dilated,
        coarse_edges=patch.coarse_edges, coarse_dilated=patch.coarse_dilated,
        coarse_to_fine=patch.coarse_to_fine, fine_to_coarse=patch.fine_to_coarse,
        cross_weights=patch.cross_weights, **common,
    )
    torch.testing.assert_close(local[patch.owned_fine_local], whole[owned], atol=2e-4, rtol=2e-4)
    if depth == 2:
        # Equal owned-node outputs must also yield equal parameter gradients.
        # This catches missing halo dependencies that forward-only checks can miss.
        model.zero_grad(set_to_none=True)
        whole[owned].square().mean().backward()
        global_grads = {
            name: parameter.grad.detach().clone() if parameter.grad is not None else None
            for name, parameter in model.named_parameters()
        }
        model.zero_grad(set_to_none=True)
        local[patch.owned_fine_local].square().mean().backward()
        for name, parameter in model.named_parameters():
            reference = global_grads[name]
            if reference is None:
                assert parameter.grad is None, name
            else:
                assert parameter.grad is not None, name
                torch.testing.assert_close(parameter.grad, reference, atol=2e-5, rtol=2e-3)


def test_periodic_displacement_uses_minimum_image():
    from graph_attention.geometry.relative import edge_relative_displacement

    coords = torch.tensor([[0., 0., 0.], [0.75, 0., 0.]])
    edge = torch.tensor([[1], [0]], dtype=torch.long)
    displ = edge_relative_displacement(coords, edge, periodic_length=1.0)
    torch.testing.assert_close(displ, torch.tensor([[-0.25, 0., 0.]]))
