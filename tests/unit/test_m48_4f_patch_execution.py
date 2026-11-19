"""M48.4f: three-way TwoScaleDiNAT SR patch execution numerical gates."""
from __future__ import annotations

import copy

import pytest
import torch

from graph_attention.geometry.two_scale_targeted import build_targeted_two_scale_patch
from graph_attention.models.two_scale_dinat import TwoScaleDiNAT
from scripts.m48_4f_patch_execution import (
    MODES, execute_patch_update, make_gpu_batch, owned_losses, sample_sha256,
)
from scripts.m48_ms_sr_train import _owned_cube


def _patches(architecture: str, n: int = 4) -> list[dict]:
    torch.manual_seed(1601)
    patches = []
    for i in range(n):
        owned = _owned_cube(8, 2, (i * 3 % 8, (i * 2 + 1) % 8, 7))
        plan, geometry = build_targeted_two_scale_patch(
            coarse_side=4, refinement=2, owned_fine=owned,
            fine_pre_layers=1, fine_post_layers=1,
            coarse_static_layers=1,
            coarse_dynamic_layers=int(architecture == "dynamic"),
        )
        patches.append({
            "state": torch.randn(plan.fine_read.numel(), 5),
            "baseline": torch.randn(plan.fine_read.numel(), 5),
            "coarse_state": torch.randn(plan.coarse_read.numel(), 5),
            "target": torch.randn(owned.numel(), 5),
            "geometry": geometry,
            "fm_time": 0.15 + 0.18 * i,
        })
    return patches


def _model(architecture: str, device: torch.device):
    torch.manual_seed(1228)
    model = TwoScaleDiNAT(
        mode=architecture, fine_width=8, coarse_width=8, num_heads=2,
        fine_pre_layers=1, fine_post_layers=1,
        coarse_static_layers=1, coarse_dynamic_layers=1,
    ).to(device)
    # Keep gradients to intermediate network blocks observable from step one.
    with torch.no_grad():
        model.fine_decoder.final_layer.projection.weight.normal_(std=0.05)
    return model


@pytest.mark.parametrize("architecture", ("static", "dynamic"))
@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_three_modes_have_same_owned_loss_gradients_and_adamw(
    architecture: str, device_name: str,
):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    patches = _patches(architecture)
    reference = _model(architecture, device)
    audits = {}
    for mode in MODES:
        model = copy.deepcopy(reference).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        metrics, audit = execute_patch_update(
            model, optimizer, patches, mode=mode, device=device,
            grad_clip=1.0, capture_audit=True,
        )
        assert metrics["gpu_microbatch"] == (
            1 if mode == "sequential_single" else len(patches)
        )
        assert metrics["owned_fine_nodes"] == 4 * (2**3)
        assert len(metrics["patch_losses"]) == 4
        assert bool(torch.isfinite(audit["predictions"]).all())
        audits[mode] = (metrics, audit)
    seq = audits["sequential_single"]
    indexed = audits["packed_index"]
    segmented = audits["packed_segmented"]
    for left in (indexed, segmented):
        torch.testing.assert_close(
            left[1]["predictions"], seq[1]["predictions"],
            atol=1e-4, rtol=1e-3,
        )
        assert left[0]["loss"] == pytest.approx(seq[0]["loss"], abs=1e-4)
        for key, expected in seq[1]["gradients"].items():
            observed = left[1]["gradients"][key]
            if expected is None:
                assert observed is None, key
            else:
                torch.testing.assert_close(
                    observed, expected, atol=5e-4, rtol=5e-3, msg=key,
                )
        for key, expected in seq[1]["post_update_parameters"].items():
            torch.testing.assert_close(
                left[1]["post_update_parameters"][key], expected,
                atol=3e-6, rtol=2e-4, msg=key,
            )
    # The indexed and segmented arms have identical packed graph topology.
    torch.testing.assert_close(
        indexed[1]["predictions"], segmented[1]["predictions"],
        atol=0, rtol=0,
    )
    for category in ("gradients", "post_update_parameters"):
        for name, a in indexed[1][category].items():
            b = segmented[1][category][name]
            if a is None:
                assert b is None
            else:
                torch.testing.assert_close(
                    a, b, atol=0, rtol=0, msg=f"{category}: {name}",
                )


@pytest.mark.parametrize("architecture", ("static", "dynamic"))
def test_single_graph_fast_path_omits_both_batch_indices(architecture: str):
    patches = _patches(architecture, 2)
    device = torch.device("cpu")
    one = make_gpu_batch(patches[:1], device, "sequential_single")
    assert one["fine_batch_index"] is None
    assert one["coarse_batch_index"] is None
    assert one["fine_graph_sizes"] is None
    assert one["coarse_graph_sizes"] is None
    for mode in ("packed_index", "packed_segmented"):
        packed = make_gpu_batch(patches, device, mode)
        assert packed["fine_batch_index"].numel() == packed["state"].shape[0]
        assert packed["coarse_batch_index"].numel() == packed["coarse_state"].shape[0]
        if mode == "packed_segmented":
            assert sum(packed["fine_graph_sizes"]) == packed["state"].shape[0]
            assert sum(packed["coarse_graph_sizes"]) == packed["coarse_state"].shape[0]
        else:
            assert packed["fine_graph_sizes"] is None
            assert packed["coarse_graph_sizes"] is None
        assert len(owned_losses(_model(architecture, device), packed)[1]) == 2
    with pytest.raises(ValueError, match="one patch"):
        make_gpu_batch(patches, device, "sequential_single")


def test_sample_hash_covers_patch_state_and_geometry():
    patches = _patches("static", 2)
    before = sample_sha256(patches)
    assert before == sample_sha256(patches)
    patches[1]["target"] = patches[1]["target"].clone()
    patches[1]["target"][0, 0] += 1.0
    assert sample_sha256(patches) != before


def test_segmented_coarse_graph_sizes_are_validated():
    patches = _patches("static", 2)
    batch = make_gpu_batch(patches, torch.device("cpu"), "packed_segmented")
    model = _model("static", torch.device("cpu"))
    batch["coarse_graph_sizes"] = (1, 1)
    with pytest.raises(ValueError, match="graph_sizes"):
        owned_losses(model, batch)
