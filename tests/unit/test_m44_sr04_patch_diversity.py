"""M44 SR-0.4: reproduce fixed-budget spatial diversity without changing SR-0.3."""

from argparse import Namespace
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch

from graph_attention.data.sr0_training import (
    choose_epoch_patch_indices,
    choose_patch_origins,
)
from graph_attention.tasks.sr0_evaluation import ensemble_residual_moments_and_increments
from scripts.diagnose_m44_sr0 import run as diagnose
from scripts.train_m44_sr0 import run as train
from test_m44_sr0_training_runner import _archive


def test_patch_pool_is_deterministic_and_includes_original_fixed_origins() -> None:
    options = dict(side=65, patch_cells=8, refinement=2, seed=42, boundary_margin=1)
    original = choose_patch_origins(patches_per_snapshot=2, **options)
    pool = choose_patch_origins(patches_per_snapshot=16, **options)
    assert len(original) == 2 and len(pool) == 16
    assert set(original).issubset(set(pool))
    assert len(set(pool)) == 16


def test_per_snapshot_cycle_is_exact_and_does_not_consume_global_random_state() -> None:
    torch.manual_seed(31415)
    expected_state = torch.get_rng_state().clone()
    for snapshot_index in (3, 12, 59):
        schedule = [
            choose_epoch_patch_indices(
                pool_size=16, patches_per_epoch=2, seed=42,
                snapshot_index=snapshot_index, epoch=epoch,
            )
            for epoch in range(16)
        ]
        assert all(len(indices) == len(set(indices)) == 2 for indices in schedule)
        assert sorted(index for selection in schedule[:8] for index in selection) == list(range(16))
        assert schedule[:8] == schedule[8:]
    assert torch.equal(torch.get_rng_state(), expected_state)
    with pytest.raises(ValueError, match="multiple"):
        choose_epoch_patch_indices(
            pool_size=15, patches_per_epoch=2, seed=42,
            snapshot_index=1, epoch=0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        choose_epoch_patch_indices(
            pool_size=16, patches_per_epoch=2, seed=42,
            snapshot_index=1, epoch=-1,
        )


def test_ensemble_moments_and_local_increments_have_expected_values() -> None:
    baseline = torch.zeros(3, 5)
    reference = torch.tensor([0.0, 1.0, 2.0])[:, None].repeat(1, 5)
    samples = torch.stack((torch.ones(3, 5), -torch.ones(3, 5)))
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    got = ensemble_residual_moments_and_increments(
        samples, baseline, reference, coords, edge
    )
    assert got["ensemble_mean_residual_rms_by_field"]["rhou"] == 0.0
    assert got["ensemble_mean_residual_bias_by_field"]["rhou"] == pytest.approx(-1.0)
    assert got["ensemble_pointwise_std_rms_by_field"]["rhoE"] == pytest.approx(1.0)
    inc = got["nearest_neighbor_increments"]
    assert inc["graph_edge_count"] == 2
    assert inc["separation_over_L_ref"] == pytest.approx(1.0)
    assert inc["generated_rms_by_field"]["rho"] == 0.0
    assert inc["reference_rms_by_field"]["rho"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="out of bounds"):
        ensemble_residual_moments_and_increments(
            samples, baseline, reference, coords, torch.tensor([[0, 3], [1, 2]])
        )


def test_synthetic_frozen_scale_diverse_training_resume_and_checkpoint_audit(tmp_path: Path) -> None:
    root, refs = _archive(tmp_path, side=15)
    base_dir = tmp_path / "sr03_reference"
    args = Namespace(
        root=root, reference_dir=refs, case="Re54/Ma0.3", side=15,
        patch_cells=4, refinement=2, patches_per_snapshot=2,
        boundary_margin=1, max_snapshots=8, train_fraction=0.70,
        validation_fraction=0.15, embargo_snapshots=1,
        epochs=1, resume=False, hidden_dim=16, layers=2, heads=2,
        lr=1.0e-3, seed=37, time_embedding_scale=1000.0,
        condition_mode="interpolated_only",
        absolute_coordinate_embedding=False, sparse_attention_backend="scatter",
        point_sample_constraint=True, checkpointing=False,
        sample_steps=2, device="cpu", output=base_dir,
    )
    base = train(args)
    assert base["point_sample_constraint"] is True
    assert base["patch_counts"] == {"train": 8, "validation": 2, "test": 2}
    scale_reference = torch.load(base_dir / "scales.pt", weights_only=True)

    run_dir = tmp_path / "sr04_spatial_diversity"
    diverse_args = replace_namespace = Namespace(**{
        **vars(args), "output": run_dir, "train_patch_pool_size": 16,
        "frozen_scales_run": base_dir,
    })
    one_epoch = train(diverse_args)
    assert one_epoch["point_sample_constraint"] is True
    assert one_epoch["patch_counts"] == {"train": 64, "validation": 2, "test": 2}
    info = one_epoch["training_patch_diversity"]
    assert info["optimizer_updates_per_epoch"] == 8
    assert info["cycle_epochs"] == 8
    assert info["training_patch_pool_size"] == 64
    assert info["active_train_patches_per_snapshot_per_epoch"] == 2
    assert info["validation_test_patch_origins_unchanged"] is True
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert set(map(tuple, manifest["patch_origins"])).issubset(
        set(map(tuple, manifest["train_patch_pool_origins"]))
    )
    assert manifest["frozen_training_scales"]["source_run_dir"] == str(base_dir.resolve())
    assert manifest["snapshot_indices"] == json.loads(
        (base_dir / "manifest.json").read_text()
    )["snapshot_indices"]
    frozen = torch.load(run_dir / "scales.pt", weights_only=True)
    for field in ("coarse_mean", "coarse_std", "residual_rms"):
        assert torch.equal(frozen[field], scale_reference[field])
    assert frozen["train_sample_ids"] == scale_reference["train_sample_ids"]
    ckpt0 = torch.load(run_dir / "last.pt", weights_only=True)
    assert len(ckpt0["history"]) == 1
    assert len(ckpt0["train_scales"]["train_sample_ids"]) == 8
    resumed = train(Namespace(**{**vars(diverse_args), "resume": True, "epochs": 2}))
    assert resumed["epochs_completed"] == 2
    ckpt1 = torch.load(run_dir / "last.pt", weights_only=True)
    assert ckpt1["history"][0] == ckpt0["history"][0]
    assert ckpt1["epoch"] == 1
    assert all(math.isfinite(float(e["validation_loss"])) for e in ckpt1["history"])

    diag = diagnose(Namespace(
        run_dir=run_dir, output=tmp_path / "diagnostics", device="cpu",
        draws=2, sample_steps=2, max_generation_patches=1, path_repeats=2,
    ))
    assert diag["best_validation_loss_reproduced"] is True
    assert diag["train_evaluation_patch_scope"] == "original_fixed_train_origins_only"
    assert diag["flow_matching_counterfactual"]["train"]["num_patches"] == 8
    assert diag["flow_matching_counterfactual"]["validation"]["num_patches"] == 2
    assert diag["repeated_flow_matching"]["train"]["paths_per_patch"] == 2
    assert diag["generation_ensemble"]["test"][0]["residual_spatial_moments"][
        "nearest_neighbor_increments"
    ]["graph_edge_count"] > 0
    assert max(
        diag["generation_ensemble"]["test"][0]["raw_point_max_abs_by_field"].values()
    ) == 0.0
