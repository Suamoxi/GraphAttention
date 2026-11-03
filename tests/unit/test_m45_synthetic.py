"""Synthetic HIT Q/B, nullspace and M45 patch-trained FM qualification."""

import pytest
import torch

from graph_attention.data.two_scale_synthetic import (
    SyntheticQB, fit_synthetic_scales, make_synthetic_pair, shared_mode,
)
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from graph_attention.geometry.two_scale_partition import (
    extract_two_scale_patch_geometry, two_scale_patch_dependencies,
)
from graph_attention.models.two_scale_dinat import TwoScaleDiNAT


@pytest.mark.parametrize("refinement", [2, 3])
@pytest.mark.parametrize("operator", ["block_average", "anchor_trilinear"])
def test_synthetic_qb_right_inverse_and_nullspace(refinement, operator):
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=refinement)
    qb = SyntheticQB(mesh, operator)
    torch.manual_seed(11)
    coarse = torch.randn(mesh.coarse_coords.shape[0], 5)
    fine = torch.randn(mesh.fine_coords.shape[0], 5)
    torch.testing.assert_close(qb.restrict(qb.reconstruct(coarse)), coarse, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(qb.restrict(qb.nullspace(fine)), torch.zeros_like(coarse), atol=1e-6, rtol=1e-6)
    template = shared_mode(qb)
    torch.testing.assert_close(qb.restrict(template), torch.zeros_like(coarse[:, :1]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(template.square().mean().sqrt(), torch.ones(()))


@pytest.mark.parametrize("operator", ["block_average", "anchor_trilinear"])
@pytest.mark.parametrize("case", ["local", "shared_mode"])
def test_synthetic_pairs_are_operator_consistent_and_reproducible(operator, case):
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    qb = SyntheticQB(mesh, operator)
    first = make_synthetic_pair(qb, case=case, sample_id=123, seed=2026)
    second = make_synthetic_pair(qb, case=case, sample_id=123, seed=2026)
    for key in ("coarse", "fine", "baseline", "residual", "mode"):
        torch.testing.assert_close(getattr(first, key), getattr(second, key))
    torch.testing.assert_close(first.fine, first.baseline + first.residual)
    torch.testing.assert_close(qb.restrict(first.fine), first.coarse, atol=1e-6, rtol=1e-6)
    assert first.fine.shape[1] == 5
    assert torch.all(first.fine[:, 0] > 0)
    assert torch.all(first.fine[:, -1] > 0)


def test_shared_random_mode_is_unobservable_at_coarse_resolution():
    mesh = build_hit_two_scale_geometry(coarse_side=4, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    local = make_synthetic_pair(qb, case="local", sample_id=1, seed=81)
    shared = make_synthetic_pair(qb, case="shared_mode", sample_id=1, seed=81)
    torch.testing.assert_close(local.coarse, shared.coarse)
    difference = shared.residual - local.residual
    assert difference.square().mean() > 1e-6
    torch.testing.assert_close(qb.restrict(difference), torch.zeros_like(shared.coarse), atol=1e-6, rtol=1e-6)
    # One scalar latent produces *spatially coherent* changes in every channel.
    template = local.mode[:, 0]
    projected = difference[:, 1] / 0.023
    amplitude = torch.dot(projected, template) / torch.dot(template, template)
    torch.testing.assert_close(projected, amplitude * template, atol=1e-5, rtol=1e-4)


def test_synthetic_statistics_are_train_only():
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    train = [make_synthetic_pair(qb, case="local", sample_id=i, seed=1000+i) for i in range(5)]
    scales = fit_synthetic_scales(train)
    assert scales.mean_coarse.shape == (5,)
    assert torch.all(scales.std_coarse > 0)
    assert torch.all(scales.rms_residual > 0)
    torch.testing.assert_close(scales.mean_coarse, torch.cat([p.coarse for p in train]).mean(0))
    with pytest.raises(ValueError):
        fit_synthetic_scales([])


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_synthetic_core_fm_optimizer_step_and_owned_loss(mode):
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    pair = make_synthetic_pair(qb, case="shared_mode", sample_id=1, seed=2026)
    scales = fit_synthetic_scales([pair, make_synthetic_pair(qb, case="shared_mode", sample_id=2, seed=2027)])
    model = TwoScaleDiNAT(
        mode=mode, coarse_width=16, fine_width=16, num_heads=4,
        fine_pre_layers=1, fine_post_layers=1,
        coarse_static_layers=1, coarse_dynamic_layers=1,
    )
    owned = torch.arange(27, dtype=torch.long)
    plan = two_scale_patch_dependencies(
        owned,
        fine_local=mesh.fine_edges, fine_dilated=mesh.fine_dilated,
        coarse_local=mesh.coarse_edges, coarse_dilated=mesh.coarse_dilated,
        coarse_to_fine=mesh.coarse_to_fine, fine_to_coarse=mesh.fine_to_coarse,
        fine_pre_layers=1, fine_post_layers=1, coarse_static_layers=1,
        coarse_dynamic_layers=int(mode == "dynamic"),
    )
    patch = extract_two_scale_patch_geometry(mesh, plan)
    coarse = (pair.coarse - scales.mean_coarse) / scales.std_coarse
    baseline = (pair.baseline - scales.mean_coarse) / scales.std_coarse
    clean = pair.residual / scales.rms_residual
    eps = torch.randn(clean.shape, generator=torch.Generator().manual_seed(42))
    time = torch.tensor([0.4])
    xt = (1-time) * eps + time * clean
    velocity = clean - eps
    result = model(
        xt[patch.fine_global], baseline=baseline[patch.fine_global],
        coarse_state=coarse[patch.coarse_global],
        fine_coords=patch.fine_coords, coarse_coords=patch.coarse_coords,
        fine_edges=patch.fine_edges, fine_dilated=patch.fine_dilated,
        coarse_edges=patch.coarse_edges, coarse_dilated=patch.coarse_dilated,
        coarse_to_fine=patch.coarse_to_fine, fine_to_coarse=patch.fine_to_coarse,
        cross_weights=patch.cross_weights, refinement=mesh.refinement,
        time=time, periodic_length=mesh.periodic_length,
        coarse_spacing=mesh.coarse_spacing,
    )
    loss = (result[patch.owned_fine_local]
            - velocity[patch.fine_global][patch.owned_fine_local]).square().mean()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.fine_decoder.final_layer.projection.weight.grad).all()
    opt.step()
    assert torch.isfinite(model.fine_decoder.final_layer.projection.weight).all()


@pytest.mark.parametrize("operator", ["block_average", "anchor_trilinear"])
def test_projected_flow_path_stays_in_q_nullspace(operator):
    from scripts.m45_synthetic_pilot import _path

    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    qb = SyntheticQB(mesh, operator)
    pair = make_synthetic_pair(qb, case="shared_mode", sample_id=1, seed=77)
    x1 = pair.residual / fit_synthetic_scales([
        pair, make_synthetic_pair(qb, case="shared_mode", sample_id=2, seed=78)
    ]).rms_residual
    xt, velocity, _ = _path(x1, 444, qb)
    torch.testing.assert_close(qb.restrict(xt), torch.zeros_like(pair.coarse), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(qb.restrict(velocity), torch.zeros_like(pair.coarse), atol=1e-6, rtol=1e-6)
    xt_full, _, _ = _path(x1, 444)
    assert qb.restrict(xt_full).abs().max() > 1e-4
