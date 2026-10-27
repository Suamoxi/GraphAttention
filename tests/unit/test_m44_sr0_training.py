"""CPU invariants for M44 SR-0.1: snapshot grouping and optional M37 features."""

from dataclasses import replace

import pytest
import torch

from graph_attention.data.sr0_training import (
    choose_patch_origins,
    select_chronological_indices,
    split_chronological_indices,
)
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT
from graph_attention.tasks.sr0 import SR0FlowMatching, fit_sr0_scales


def test_even_chronological_snapshot_selection_is_unique_and_ordered() -> None:
    selected = select_chronological_indices(100, 24)
    assert len(selected) == 24
    assert selected == tuple(sorted(set(selected)))
    assert selected[0] > 0 and selected[-1] < 100
    assert select_chronological_indices(8, 24) == tuple(range(8))


def test_snapshot_level_embargo_blocks_are_disjoint() -> None:
    chosen = select_chronological_indices(100, 24)
    split = split_chronological_indices(chosen, embargo_snapshots=2)
    split.validate(chosen)
    assert len(split.embargo) == 4
    assert max(split.train) < min(split.validation) < max(split.validation) < min(split.test)
    assert set(split.train).isdisjoint(set(split.validation))
    assert set(split.test).isdisjoint(set(split.train) | set(split.validation))


def test_too_few_snapshots_with_embargo_rejected() -> None:
    with pytest.raises(ValueError, match="too few"):
        split_chronological_indices((0, 1, 2, 3), embargo_snapshots=1)
    with pytest.raises(ValueError, match="chronological"):
        split_chronological_indices((2, 1, 3), embargo_snapshots=0)


def test_patch_origins_are_fixed_unique_nonoverlapping_and_interior() -> None:
    origins = choose_patch_origins(
        side=65, patch_cells=8, refinement=2,
        patches_per_snapshot=10, seed=42, boundary_margin=1,
    )
    assert len(origins) == len(set(origins)) == 10
    assert origins == choose_patch_origins(
        side=65, patch_cells=8, refinement=2,
        patches_per_snapshot=10, seed=42, boundary_margin=1,
    )
    for a in origins:
        assert all(1 <= k <= 55 for k in a)
    for i, a in enumerate(origins):
        for b in origins[i + 1:]:
            assert any(abs(a[d] - b[d]) >= 8 for d in range(3))


def test_patch_selection_fails_if_not_enough_interior_space() -> None:
    with pytest.raises(ValueError, match="no valid interior"):
        choose_patch_origins(
            side=9, patch_cells=8, refinement=2,
            patches_per_snapshot=1, seed=0, boundary_margin=1,
        )
    with pytest.raises(ValueError, match="only"):
        choose_patch_origins(
            side=9, patch_cells=8, refinement=2,
            patches_per_snapshot=2, seed=0, boundary_margin=0,
        )


def test_m37_can_skip_absolute_coordinate_mlp_without_changing_parameters() -> None:
    def make(enabled: bool) -> AlternatingDilatedGeometricDiT:
        return AlternatingDilatedGeometricDiT(
            in_channels=10, out_channels=5, hidden_dim=8,
            num_heads=2, num_layers=2, spatial_dim=3,
            conditioning_channels=1,
            use_absolute_coordinate_embedding=enabled,
        )

    torch.manual_seed(13)
    with_coords = make(True)
    torch.manual_seed(13)
    without_coords = make(False)
    assert set(with_coords.state_dict()) == set(without_coords.state_dict())
    for name in with_coords.state_dict():
        torch.testing.assert_close(with_coords.state_dict()[name], without_coords.state_dict()[name])

    class FailIfCalled(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("absolute coordinate pathway must not execute")

    without_coords.coordinate_projection = FailIfCalled()
    inputs = torch.ones(3, 10)
    coords = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    dilated = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
    output = without_coords(
        inputs, coords=coords, edge_index=edge,
        batch_index=torch.zeros(3, dtype=torch.long),
        conditioning=torch.tensor([[0.5]]),
        attention_edge_indices={"dilated": dilated},
    )
    assert output.shape == (3, 5)
    assert torch.isfinite(output).all()


def test_m44_sr0_interpolated_only_condition_has_five_channels() -> None:
    # The original M44 single-pair fixture is kept as the source of truth.
    from test_m44_sr0 import _pair

    pair = _pair()
    scales = fit_sr0_scales([pair], train_sample_ids=("train-a",))
    task = SR0FlowMatching(scales, spatial_condition_mode="interpolated_only")
    problem = task.make_training_problem(pair, generator=torch.Generator().manual_seed(13))
    assert problem.spatial_condition.shape == (3, 5)
    probe = task.model_probe(problem)
    assert probe.inputs.shape == (3, 10)
    assert len(probe.input_channels) == 10

    original = SR0FlowMatching(scales, spatial_condition_mode="interpolated_plus_baseline")
    old_problem = original.make_training_problem(pair, generator=torch.Generator().manual_seed(13))
    assert original.model_probe(old_problem).inputs.shape == (3, 15)
    torch.testing.assert_close(problem.spatial_condition, old_problem.spatial_condition[:, :5])
