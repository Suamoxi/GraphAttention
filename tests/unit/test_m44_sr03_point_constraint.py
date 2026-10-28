"""M44 SR-0.3: intrinsic point-sample consistency, not physical LES restriction."""

from argparse import Namespace
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch

from graph_attention.tasks.sr0 import SR0FlowMatching, fit_sr0_scales
from graph_attention.tasks.sr0_evaluation import (
    per_field_mse, point_sample_anchors, project_point_sample,
)
from scripts.diagnose_m44_sr0 import _verified_point_proxy_free_mask, run as diagnose
from scripts.train_m44_sr0 import run as train
from test_m44_sr0 import _pair
from test_m44_sr0_training_runner import _archive


def _point_pair():
    pair = _pair()
    # A genuine point sample has zero residual at every retained coarse vertex.
    target = pair.fine_baseline.clone()
    target[1] += torch.tensor([0.03, -0.04, 0.07, -0.02, 0.05])
    source = replace(
        pair.fine.source,
        sample_metadata=({"sr0_pilot": "point_sampled_dns_not_les"},),
    )
    return replace(
        pair,
        fine=replace(pair.fine, source=source),
        fine_target=target,
    )


class _ConstantVelocity(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.ones_like(inputs[:, :5])


class _ZeroVelocity(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.zeros_like(inputs[:, :5])


def test_point_sample_constrained_training_keeps_original_free_node_fm_path():
    pair = _point_pair()
    scales = fit_sr0_scales([pair], train_sample_ids=("train-a",))
    constrained = SR0FlowMatching(
        scales, spatial_condition_mode="interpolated_only", point_sample_constraint=True
    )
    original = SR0FlowMatching(scales, spatial_condition_mode="interpolated_only")
    a = constrained.make_training_problem(
        pair, generator=torch.Generator().manual_seed(127)
    )
    b = original.make_training_problem(
        pair, generator=torch.Generator().manual_seed(127)
    )
    free = constrained.point_sample_free_mask(pair)
    assert free.tolist() == [False, True, False]
    torch.testing.assert_close(a.flow.inputs[free], b.flow.inputs[free])
    torch.testing.assert_close(a.flow.targets[free], b.flow.targets[free])
    torch.testing.assert_close(a.flow.conditioning, b.flow.conditioning)
    assert torch.equal(a.flow.inputs[~free], torch.zeros(2, 5))
    assert torch.equal(a.flow.targets[~free], torch.zeros(2, 5))
    assert a.flow.node_weights.tolist() == [0.0, 2.0, 0.0]
    assert constrained.model_probe(a).inputs.shape == (3, 10)
    objective = constrained.training_loss(_ZeroVelocity(), a).mean
    torch.testing.assert_close(
        objective, a.flow.targets[free].square().mean()
    )


def test_point_sample_constrained_generation_is_intrinsic_and_reproducible():
    pair = _point_pair()
    scales = fit_sr0_scales([pair], train_sample_ids=("train-a",))
    constrained = SR0FlowMatching(
        scales, spatial_condition_mode="interpolated_only", point_sample_constraint=True
    )
    original = SR0FlowMatching(scales, spatial_condition_mode="interpolated_only")
    inference = replace(pair, fine_target=None)
    generated = constrained.generate(
        _ConstantVelocity(), inference, steps=4, solver="heun", sampling_keys=("draw-a",)
    )
    generated_again = constrained.generate(
        _ConstantVelocity(), inference, steps=4, solver="heun", sampling_keys=("draw-a",)
    )
    anchors, coarse_ids = point_sample_anchors(pair.transfer)
    assert torch.equal(generated[anchors], pair.coarse_state[coarse_ids])
    assert torch.equal(generated_again, generated)
    assert torch.equal(
        project_point_sample(generated, pair.coarse_state, pair.transfer), generated
    )
    assert torch.isfinite(generated).all()
    legacy = original.generate(
        _ConstantVelocity(), inference, steps=4, solver="heun", sampling_keys=("draw-a",)
    )
    assert not torch.equal(legacy[anchors], pair.coarse_state[coarse_ids])
    assert not torch.equal(generated[1], pair.fine_baseline[1])


def test_point_sampling_constraint_rejects_nonpoint_metadata_and_inconsistent_targets():
    pair = _point_pair()
    scales = fit_sr0_scales([pair], train_sample_ids=("train-a",))
    task = SR0FlowMatching(scales, point_sample_constraint=True)
    missing_tag = replace(
        pair, fine=replace(
            pair.fine, source=replace(pair.fine.source, sample_metadata=({},))
        )
    )
    with pytest.raises(ValueError, match="explicitly tagged"):
        task.make_validation_problem(missing_tag)
    with pytest.raises(ValueError, match="fine target differs"):
        task.make_validation_problem(
            replace(pair, fine_target=pair.fine_target + 0.1)
        )
    with pytest.raises(ValueError, match="interpolation"):
        task.make_validation_problem(
            replace(pair, fine_baseline=pair.fine_baseline + 0.1)
        )


def test_common_support_mask_rejects_non_point_sample_proxy():
    pair = _point_pair()
    assert _verified_point_proxy_free_mask(pair).tolist() == [False, True, False]
    untagged = replace(
        pair,
        fine=replace(
            pair.fine,
            source=replace(pair.fine.source, sample_metadata=({},)),
        ),
    )
    with pytest.raises(ValueError, match="explicitly tagged"):
        _verified_point_proxy_free_mask(untagged)


def test_free_node_metric_and_m11_optional_sampling_mask():
    predictions = torch.zeros(3, 5)
    target = torch.ones(3, 5)
    free = torch.tensor([False, True, False])
    torch.testing.assert_close(
        per_field_mse(predictions, target, node_mask=free), torch.ones(5)
    )
    with pytest.raises(ValueError, match="node_mask"):
        per_field_mse(predictions, target, node_mask=free.float())


def test_synthetic_hdf5_constrained_train_diagnose_and_manifest_parity(tmp_path: Path):
    root, refs = _archive(tmp_path)
    output = tmp_path / "point_constrained"
    args = Namespace(
        root=root, reference_dir=refs, case="Re54/Ma0.3", side=7,
        patch_cells=4, refinement=2, patches_per_snapshot=1, boundary_margin=1,
        max_snapshots=8, train_fraction=0.70, validation_fraction=0.15,
        embargo_snapshots=1, epochs=1, resume=False,
        hidden_dim=16, layers=2, heads=2, lr=1.0e-3, seed=37,
        time_embedding_scale=1000.0, condition_mode="interpolated_only",
        absolute_coordinate_embedding=False, sparse_attention_backend="scatter",
        point_sample_constraint=True,
        checkpointing=False, sample_steps=2, device="cpu", output=output,
    )
    trained = train(args)
    assert trained["point_sample_constraint"] is True
    assert trained["flow_mse_support"] == "unobserved_fine_nodes"
    assert trained["model_configuration"]["in_channels"] == 10
    assert trained["coarse_total_nodes_per_patch"] == 27
    assert trained["validation_generation_diagnostic"]["num_coarse_nodes"] == 27
    assert max(trained["validation_generation_diagnostic"]["generated_point_sample_max_abs_by_field"].values()) < 1e-5
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["point_sample_constraint"] is True
    diagnosed = diagnose(Namespace(
        run_dir=output, output=tmp_path / "diagnosed", device="cpu",
        draws=2, sample_steps=2, max_generation_patches=1, path_repeats=2,
    ))
    assert diagnosed["best_validation_loss_reproduced"] is True
    assert diagnosed["point_sample_constraint"] is True
    assert diagnosed["flow_mse_support"] == "unobserved_fine_nodes"
    assert diagnosed["repeated_flow_matching"]["train"]["paths_per_patch"] == 2
    for split in ("train", "validation", "test"):
        native = diagnosed["repeated_flow_matching"][split]
        common = native["common_unobserved_node_support"]
        assert common["unobserved_nodes_per_patch"] == 98
        assert common["coarse_nodes_per_patch"] == 27
        assert common["matched_flow_mse"] == pytest.approx(native["matched_flow_mse"])
        assert common["zero_velocity_flow_mse"] == pytest.approx(native["zero_velocity_flow_mse"])
        if native["wrong_minus_matched_mse"] is not None:
            assert common["wrong_minus_matched_mse"] == pytest.approx(
                native["wrong_minus_matched_mse"]
            )
    assert math.isfinite(
        diagnosed["repeated_flow_matching"]["test"]["matched_flow_mse"]
    )
    for split in ("validation", "test"):
        assert max(
            diagnosed["generation_ensemble"][split][0][
                "raw_point_max_abs_by_field"
            ].values()
        ) < 1e-5
