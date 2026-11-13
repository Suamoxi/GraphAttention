"""M48 qualification of target-driven global-ID geometry and CEDAR algebra.

Only the integration smoke needs the external MS_SR archive; these unit
tests work with local periodic meshes and lightweight mock operators.
"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from graph_attention.data.ms_sr_adapter import (
    _isotropic_box, component_max_relative, flat_channels,
    qualify_full_pair, select_pilot_record,
)
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from graph_attention.geometry.two_scale_partition import (
    extract_two_scale_patch_geometry, two_scale_patch_dependencies,
)
from graph_attention.geometry.two_scale_targeted import build_targeted_two_scale_patch
from graph_attention.models.two_scale_dinat import TwoScaleDiNAT


def _keys(edge, nsource, ntarget):
    # Order-independent global edge identity, used to compare geometry.
    return torch.sort(edge[0] * ntarget + edge[1]).values


def _global_edges(patch, edge_name):
    edge = getattr(patch, edge_name)
    if edge_name in ("coarse_to_fine", "fine_to_coarse"):
        if edge_name == "coarse_to_fine":
            return torch.stack((patch.coarse_global[edge[0]], patch.fine_global[edge[1]]))
        return torch.stack((patch.fine_global[edge[0]], patch.coarse_global[edge[1]]))
    ids = patch.fine_global if edge_name.startswith("fine") else patch.coarse_global
    return ids[edge]


@pytest.mark.parametrize("coarse_side,depth,mode", [
    (3, 1, "static"), (3, 1, "dynamic"),
    (4, 1, "static"), (4, 1, "dynamic"),
    (4, 2, "static"), (4, 2, "dynamic"),
    (6, 1, "static"), (6, 1, "dynamic"),
])
def test_targeted_support_equals_existing_global_qualification(coarse_side, depth, mode):
    full = build_hit_two_scale_geometry(coarse_side=coarse_side, refinement=2)
    nf = 2 * coarse_side
    owned = torch.tensor([nf**3 - 1, 0, nf**2, nf**2 + 1], dtype=torch.long)
    dynamic_depth = depth if mode == "dynamic" else 0
    kwargs = {
        "fine_pre_layers": depth, "fine_post_layers": depth,
        "coarse_static_layers": depth, "coarse_dynamic_layers": dynamic_depth,
    }
    reference_plan = two_scale_patch_dependencies(
        owned, fine_local=full.fine_edges, fine_dilated=full.fine_dilated,
        coarse_local=full.coarse_edges, coarse_dilated=full.coarse_dilated,
        coarse_to_fine=full.coarse_to_fine, fine_to_coarse=full.fine_to_coarse,
        **kwargs,
    )
    reference = extract_two_scale_patch_geometry(full, reference_plan)
    plan, candidate = build_targeted_two_scale_patch(
        coarse_side=coarse_side, refinement=2, owned_fine=owned,
        **kwargs,
    )
    for key in reference_plan.__dataclass_fields__:
        torch.testing.assert_close(getattr(plan, key), getattr(reference_plan, key), atol=0, rtol=0)
    torch.testing.assert_close(candidate.fine_global, reference.fine_global, atol=0, rtol=0)
    torch.testing.assert_close(candidate.coarse_global, reference.coarse_global, atol=0, rtol=0)
    for key in ("fine_edges", "fine_dilated", "coarse_edges", "coarse_dilated",
                "coarse_to_fine", "fine_to_coarse"):
        a = _global_edges(candidate, key)
        b = _global_edges(reference, key)
        if key == "coarse_to_fine":
            na, nb = coarse_side**3, nf**3
        elif key == "fine_to_coarse":
            na, nb = nf**3, coarse_side**3
        elif key.startswith("fine_"):
            na, nb = nf**3, nf**3
        else:
            na, nb = coarse_side**3, coarse_side**3
        torch.testing.assert_close(_keys(a, na, nb), _keys(b, na, nb), atol=0, rtol=0)
    # Inspect cross-scale weights by global edge key, not by edge enumeration.
    for x in (candidate, reference):
        edge = _global_edges(x, "coarse_to_fine")
        keys = edge[0] * nf**3 + edge[1]
        if x is candidate:
            order_a, key_a = torch.argsort(keys), torch.sort(keys).values
        else:
            order_b, key_b = torch.argsort(keys), torch.sort(keys).values
    torch.testing.assert_close(key_a, key_b, atol=0, rtol=0)
    torch.testing.assert_close(candidate.cross_weights[order_a], reference.cross_weights[order_b],
                               rtol=1e-6, atol=1e-7)


def _model_kwargs(geometry):
    return {
        "fine_coords": geometry.fine_coords,
        "coarse_coords": geometry.coarse_coords,
        "fine_edges": geometry.fine_edges,
        "fine_dilated": geometry.fine_dilated,
        "coarse_edges": geometry.coarse_edges,
        "coarse_dilated": geometry.coarse_dilated,
        "coarse_to_fine": geometry.coarse_to_fine,
        "fine_to_coarse": geometry.fine_to_coarse,
        "cross_weights": geometry.cross_weights,
        "refinement": geometry.refinement,
        "periodic_length": geometry.periodic_length,
        "coarse_spacing": geometry.coarse_spacing,
    }


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_targeted_owned_velocity_and_gradients_match_full_model(mode):
    torch.manual_seed(48)
    whole = build_hit_two_scale_geometry(coarse_side=4, refinement=2)
    _, patch = build_targeted_two_scale_patch(
        coarse_side=4, refinement=2,
        owned_fine=torch.tensor([1, 2, 3, 10]),
        fine_pre_layers=1, fine_post_layers=1, coarse_static_layers=1,
        coarse_dynamic_layers=int(mode == "dynamic"),
    )
    model = TwoScaleDiNAT(
        mode=mode, fine_width=8, coarse_width=8, num_heads=2,
        fine_pre_layers=1, fine_post_layers=1, coarse_static_layers=1,
        coarse_dynamic_layers=1,
    )
    with torch.no_grad():
        for backbone in (model.coarse_encoder, model.fine_encoder, model.fine_decoder):
            for block in backbone.blocks:
                torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
                torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)
            torch.nn.init.normal_(backbone.final_layer.projection.weight, std=0.02)
        if mode == "dynamic":
            for block in model.coarse_dynamic.blocks:
                torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
                torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)
            torch.nn.init.normal_(model.coarse_dynamic.final_layer.projection.weight, std=0.02)

    x = torch.randn((8**3, 5))
    b = torch.randn_like(x)
    c = torch.randn((4**3, 5))
    t = torch.tensor([0.33])
    model.eval()
    full_result = model(x, baseline=b, coarse_state=c, time=t, **_model_kwargs(whole))
    patch_result = model(
        x[patch.fine_global], baseline=b[patch.fine_global],
        coarse_state=c[patch.coarse_global],
        time=t, **_model_kwargs(patch),
    )
    owned = patch.fine_global[patch.owned_fine_local]
    torch.testing.assert_close(
        patch_result[patch.owned_fine_local], full_result[owned],
        rtol=2e-4, atol=2e-4,
    )
    model.zero_grad(set_to_none=True)
    full_result[owned].square().mean().backward()
    reference_grads = {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    patch_result[patch.owned_fine_local].square().mean().backward()
    for name, p in model.named_parameters():
        reference = reference_grads[name]
        if reference is None:
            assert p.grad is None, name
        else:
            assert p.grad is not None, name
            torch.testing.assert_close(p.grad, reference, atol=3e-5, rtol=3e-3)


def test_targeted_builder_rejects_invalid_ids_and_memory_cap():
    kwargs = dict(
        coarse_side=4, refinement=2, fine_pre_layers=1, fine_post_layers=1,
        coarse_static_layers=1, coarse_dynamic_layers=0,
    )
    with pytest.raises(ValueError, match="out of range"):
        build_targeted_two_scale_patch(owned_fine=torch.tensor([512]), **kwargs)
    with pytest.raises(ValueError, match="safety cap"):
        build_targeted_two_scale_patch(owned_fine=torch.tensor([1]), max_fine_read=1, **kwargs)


class _BlockAverage:
    """Small stand-in for the CEDAR interface, never used in real data."""
    def __init__(self, r=2):
        self.r = r

    def restrict(self, fine):
        b, ch, f1, f2, f3 = fine.shape
        r = self.r
        assert f1 % r == f2 % r == f3 % r == 0
        return fine.reshape(b, ch, f1//r, r, f2//r, r, f3//r, r).mean((3, 5, 7))

    def prolong(self, coarse, target_shape):
        assert tuple(x*self.r for x in coarse.shape[-3:]) == tuple(target_shape)
        return coarse.repeat_interleave(self.r, -3).repeat_interleave(self.r, -2).repeat_interleave(self.r, -1)

    def project(self, fine):
        return fine - self.prolong(self.restrict(fine), fine.shape[-3:])


def test_full_field_projection_contract_and_layout():
    qb = _BlockAverage()
    fine = torch.randn((1, 5, 6, 6, 6), dtype=torch.float64)
    coarse = qb.restrict(fine)
    base, residual, check = qualify_full_pair(coarse, fine, qb)
    assert max(r["max_component_scaled_linf"] for r in check.values()) < 1e-11
    torch.testing.assert_close(qb.restrict(base), coarse)
    torch.testing.assert_close(base+residual, fine)
    assert flat_channels(fine).shape == (216, 5)
    with pytest.raises(ValueError, match="expected an MS_SR"):
        flat_channels(fine[0])


def test_manifest_selection_does_not_admit_locked_test():
    record = {
        "role": "teacher", "source_role": "teacher",
        "case": "ReT060/Mt0.23_Xi0.0", "snapshot_id": 5,
        "physical_time_s": 0.5, "filter_id": "w1p25_l1",
        "fine_factor": 4, "coarse_factor": 8,
        "input_shape": [24, 24, 24], "target_shape": [48, 48, 48],
        "coarse_spacing": [0.1, 0.1, 0.1],
        "fine_spacing": [0.05, 0.05, 0.05],
    }
    manifest = {"schema": "ms-sr.campaign.v1", "pairs": [record]}
    assert select_pilot_record(manifest) == record
    length, ratio = _isotropic_box(record)
    assert length == pytest.approx(2.4)
    assert ratio == 2
    with pytest.raises(ValueError, match="locked_in_regime"):
        select_pilot_record(manifest, role="locked_in_regime")
    bad = deepcopy(record)
    bad["fine_spacing"] = [0.05, 0.05, 0.06]
    with pytest.raises(ValueError, match="isotropic"):
        _isotropic_box(bad)
