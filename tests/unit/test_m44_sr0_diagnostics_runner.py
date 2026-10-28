"""Contract gates for SR-0.2 matching-by-origin and read-only checkpoint audit."""

from argparse import Namespace
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.diagnose_m44_sr0 import _wrong_condition_donors, run as diagnose
from scripts.train_m44_sr0 import run as train
from test_m44_sr0_training_runner import _archive


@dataclass
class _FakePair:
    fine: object


def _fake_pair(sample_id: str) -> _FakePair:
    return _FakePair(fine=SimpleNamespace(source=SimpleNamespace(sample_ids=(sample_id,))))


def test_wrong_condition_changes_snapshot_not_patch_origin():
    pairs = [
        _fake_pair("case:t0/sr0_patch_0_0_0"),
        _fake_pair("case:t0/sr0_patch_4_4_4"),
        _fake_pair("case:t1/sr0_patch_0_0_0"),
        _fake_pair("case:t1/sr0_patch_4_4_4"),
    ]
    assert _wrong_condition_donors(pairs) == {0: 2, 2: 0, 1: 3, 3: 1}
    assert _wrong_condition_donors(pairs[:2]) == {0: None, 1: None}
    with pytest.raises(ValueError, match="duplicate source snapshot"):
        _wrong_condition_donors([pairs[0], pairs[0]])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_synthetic_archive_checkpoint_diagnosis_does_not_modify_training_run(tmp_path: Path):
    root, refs = _archive(tmp_path)
    source = tmp_path / "source_run"
    args = Namespace(
        root=root, reference_dir=refs, case="Re54/Ma0.3", side=7,
        patch_cells=4, refinement=2, patches_per_snapshot=1, boundary_margin=1,
        max_snapshots=8, train_fraction=0.70, validation_fraction=0.15,
        embargo_snapshots=1, epochs=1, resume=False,
        hidden_dim=16, layers=2, heads=2, lr=1.0e-3, seed=37,
        time_embedding_scale=1000.0, condition_mode="interpolated_only",
        absolute_coordinate_embedding=False, sparse_attention_backend="scatter",
        checkpointing=False, sample_steps=2, device="cpu", output=source,
    )
    train(args)
    ckpt_hash_before = _sha(source / "best.pt")
    manifest_hash_before = _sha(source / "manifest.json")
    diag_output = tmp_path / "diagnostics"
    summary = diagnose(Namespace(
        run_dir=source, output=diag_output, device="cpu", draws=2,
        sample_steps=2, max_generation_patches=1, path_repeats=3,
    ))
    assert summary["status"] == "sr0_checkpoint_diagnostics_complete"
    assert summary["best_epoch"] == 0
    assert summary["flow_matching_counterfactual"]["train"]["counterfactual_patches"] == 4
    for split in ("train", "validation", "test"):
        native = summary["repeated_flow_matching"][split]
        common = native["common_unobserved_node_support"]
        assert common["fine_nodes_per_patch"] == 125
        assert common["coarse_nodes_per_patch"] == 27
        assert common["unobserved_nodes_per_patch"] == 98
        assert math.isfinite(common["matched_flow_mse"])
        assert math.isfinite(common["zero_velocity_flow_mse"])
        assert common["per_snapshot"]
    assert summary["flow_matching_counterfactual"]["validation"]["counterfactual_patches"] == 0
    assert summary["flow_matching_counterfactual"]["test"]["wrong_coarse_flow_mse"] is None
    assert math.isfinite(summary["flow_matching_counterfactual"]["test"]["matched_flow_mse"])
    repeated = summary["repeated_flow_matching"]
    assert repeated["train"]["paths_per_patch"] == 3
    assert repeated["train"]["counterfactual_paths"] == 12
    assert math.isfinite(repeated["train"]["wrong_minus_matched_mse"])
    assert repeated["train"]["num_snapshots"] == 4
    assert repeated["validation"]["counterfactual_paths"] == 0
    assert repeated["test"]["wrong_minus_matched_mse"] is None
    for split in ("train", "validation", "test"):
        assert repeated[split]["repeat_zero_matched_flow_mse"] == pytest.approx(
            summary["flow_matching_counterfactual"][split]["matched_flow_mse"], abs=1e-5
        )
    assert (diag_output / "diagnostic_summary.json").is_file()
    assert (diag_output / "flow_matching_per_patch.json").is_file()
    assert (diag_output / "ensemble_examples.pt").is_file()
    example = torch.load(diag_output / "ensemble_examples.pt", map_location="cpu", weights_only=True)
    assert example["test"]["raw_generations"].shape[0] == 2
    assert example["validation"]["raw_generations"].shape[0] == 2
    assert _sha(source / "best.pt") == ckpt_hash_before
    assert _sha(source / "manifest.json") == manifest_hash_before
    with pytest.raises(FileExistsError, match="overwrite"):
        diagnose(Namespace(
            run_dir=source, output=diag_output, device="cpu", draws=2,
            sample_steps=2, max_generation_patches=1,
        ))
