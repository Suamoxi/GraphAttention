"""M48.4c numerical patch-batching and exact-CEDAR regression tests."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from graph_attention.geometry.two_scale_targeted import build_targeted_two_scale_patch
from graph_attention.models.two_scale_dinat import TwoScaleDiNAT
from scripts.m48_ms_sr_train import _forward_args, _owned_cube
from scripts.m48_4c_patch_batch import (
    accumulated_batch_step, make_patch, pack_patches,
    packed_loss, preflight_halos,
)


def _mode_model(mode: str):
    model = TwoScaleDiNAT(
        mode=mode, fine_width=8, coarse_width=8, num_heads=2,
        fine_pre_layers=1, fine_post_layers=1,
        coarse_static_layers=1, coarse_dynamic_layers=1,
    )
    with torch.no_grad():
        torch.nn.init.normal_(model.fine_decoder.final_layer.projection.weight, std=0.02)
    return model


def _patches(mode: str, n: int = 2):
    torch.manual_seed(48)
    samples = []
    for i in range(n):
        owned = _owned_cube(8, 2, (i * 3, 6 if i else 2, 7))
        plan, geometry = build_targeted_two_scale_patch(
            coarse_side=4, refinement=2, owned_fine=owned,
            fine_pre_layers=1, fine_post_layers=1,
            coarse_static_layers=1,
            coarse_dynamic_layers=int(mode == "dynamic"),
        )
        samples.append({
            "state": torch.randn(len(plan.fine_read), 5),
            "baseline": torch.randn(len(plan.fine_read), 5),
            "coarse_state": torch.randn(len(plan.coarse_read), 5),
            "target": torch.randn(len(owned), 5),
            "geometry": geometry,
            "owned": owned, "fm_time": 0.25 + i * 0.2,
        })
    return samples


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_packed_patch_prediction_and_parameter_gradients_equal_independent(mode):
    patches = _patches(mode)
    model = _mode_model(mode)
    model.eval()

    packed = pack_patches(patches, torch.device("cpu"))
    packed_mean, packed_components = packed_loss(model, packed)
    model.zero_grad(set_to_none=True)
    packed_mean.backward()
    expected_grad = {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)

    separate = []
    for patch in patches:
        geometry = patch["geometry"]
        pred = model(
            patch["state"], baseline=patch["baseline"],
            coarse_state=patch["coarse_state"],
            time=torch.tensor([patch["fm_time"]]),
            **_forward_args(geometry),
        )
        separate.append(
            (pred[geometry.owned_fine_local] - patch["target"]).square().mean()
        )
    components = torch.stack(separate)
    torch.testing.assert_close(
        packed_components, components, rtol=2e-5, atol=2e-6,
    )
    components.mean().backward()
    for name, p in model.named_parameters():
        reference = expected_grad[name]
        if reference is None:
            assert p.grad is None, name
        else:
            torch.testing.assert_close(p.grad, reference, rtol=3e-3, atol=3e-5)

    # No attention edge in the packed graph may cross its graph-ID boundary.
    geom = packed["geometry"]
    f = packed["fine_batch_index"]
    c = packed["coarse_batch_index"]
    for edges, index in (
        (geom.fine_edges, f), (geom.fine_dilated, f),
        (geom.coarse_edges, c), (geom.coarse_dilated, c),
    ):
        assert torch.equal(index[edges[0]], index[edges[1]])
    for edge in (geom.coarse_to_fine, geom.fine_to_coarse):
        left, right = (c, f) if edge is geom.coarse_to_fine else (f, c)
        assert torch.equal(left[edge[0]], right[edge[1]])


class _BlockAverage:
    def restrict(self, fine):
        n = fine.shape[-1] // 2
        return fine.reshape(1, 5, n, 2, n, 2, n, 2).mean((3, 5, 7))

    def project(self, fine):
        coarse = self.restrict(fine)
        x = coarse.repeat_interleave(2, -3).repeat_interleave(2, -2).repeat_interleave(2, -1)
        return fine - x


def _prepared():
    torch.manual_seed(18)
    fine = torch.zeros((1, 5, 8, 8, 8), dtype=torch.float64)
    coarse = torch.zeros((1, 5, 4, 4, 4), dtype=torch.float64)
    pair = SimpleNamespace(
        fine=fine, coarse=coarse, refinement=2, periodic_length=1.0,
    )
    cedar = _BlockAverage()
    target = cedar.project(torch.randn_like(fine)).permute(0, 2, 3, 4, 1).reshape(-1, 5)
    return pair, cedar, torch.zeros(64, 5, dtype=torch.float64), (
        torch.zeros(512, 5, dtype=torch.float64)
    ), target


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_independent_global_projected_sources_and_patch_seams(mode):
    args = SimpleNamespace(
        core_side=2, depth=1, max_fine_read=10000, max_coarse_read=10000,
    )
    prep = _prepared()
    a = make_patch(
        prep, args, fm_time=0.2, core_origin=(7, 7, 7),
        source_seed=415, mode=mode,
    )
    b = make_patch(
        prep, args, fm_time=0.4, core_origin=(7, 7, 7),
        source_seed=416, mode=mode,
    )
    assert a["cedar_defect"] < 1e-12
    assert b["cedar_defect"] < 1e-12
    assert not torch.equal(a["state"], b["state"])
    assert a["owned_fine_nodes"] == b["owned_fine_nodes"] == 8
    assert a["fine_nodes_read"] >= a["owned_fine_nodes"]
    packed = pack_patches([a, b], torch.device("cpu"))
    assert packed["time"].tolist() == pytest.approx([0.2, 0.4])
    assert packed["geometry"].owned_fine_local.numel() == 16


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_effective_batch_gradient_is_independent_of_gpu_microbatch_size(
    monkeypatch, mode,
):
    # Use real packed two-scale geometry and actual model forward/backward.
    # Only the HDF5/CEDAR source factory is synthetic.
    patches = _patches(mode, n=4)
    index = iter(range(4))

    def fake_patch(*_args, **_kwargs):
        patch = patches[next(index)]
        return {
            **patch, "source_seed": 0, "core_origin": [0, 0, 0],
            "fine_nodes_read": patch["state"].shape[0],
            "coarse_nodes_read": patch["coarse_state"].shape[0],
            "owned_fine_nodes": patch["target"].shape[0],
            "cedar_defect": 0.0, "projection_seconds": 0.0,
            "geometry_seconds": 0.0,
        }

    from scripts import m48_4c_patch_batch as batch_module
    monkeypatch.setattr(batch_module, "make_patch", fake_patch)
    initial = _mode_model(mode).state_dict()
    # Recreate identical networks with one seed, then restore equal weights.
    results = []
    for micro in (1, 2, 4):
        index = iter(range(4))
        model = _mode_model(mode)
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        args = SimpleNamespace(microbatch_size=micro, core_side=2, grad_clip=1e8)
        result = accumulated_batch_step(
            model, optimizer, args, prepared=_prepared(),
            draws=[(0.2, (0, 0, 0), k) for k in range(4)],
            device=torch.device("cpu"),
        )
        results.append((result, {
            n: v.detach().clone() for n, v in model.state_dict().items()
        }))
    for (a, model_a), (b, model_b) in zip(results, results[1:]):
        assert a["effective_batch"] == b["effective_batch"] == 4
        assert a["loss"] == pytest.approx(b["loss"], rel=3e-5, abs=1e-6)
        for key in model_a:
            torch.testing.assert_close(model_a[key], model_b[key],
                                       rtol=5e-4, atol=3e-5)


def test_exact_halo_preflight_two_grid_sizes_and_seam_guard():
    records = [
        {"case": "ReT060/example", "input_shape": [4, 4, 4],
         "target_shape": [8, 8, 8]},
        {"case": "ReT100/example", "input_shape": [5, 5, 5],
         "target_shape": [10, 10, 10]},
    ]
    opts = SimpleNamespace(core_side=2, depth=1,
                           max_fine_read=10000, max_coarse_read=10000)
    report = preflight_halos(records, opts, "dynamic")
    assert report["sources"] == 2
    assert len(report["results"]) == 4
    for result in report["results"]:
        assert result["owned_fine_nodes"] == 8
        assert result["fine_nodes_read"] >= 8
        assert result["cross_edges"] > 0
    opts.max_fine_read = 1
    with pytest.raises(ValueError, match="safety cap"):
        preflight_halos(records, opts, "dynamic")
    records[0]["target_shape"] = [8, 8, 9]
    with pytest.raises(ValueError, match="cubic"):
        preflight_halos(records, opts, "dynamic")
