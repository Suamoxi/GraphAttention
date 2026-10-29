"""M44 SR-0.5 tests: immutable full-state continuation on synthetic DNS."""

from argparse import Namespace
import hashlib
from pathlib import Path

import pytest
import torch

from scripts.prepare_m44_sr05_continuation import prepare
from scripts.train_m44_sr0 import run as train
from test_m44_sr0_training_runner import _archive


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_continuation_preserves_original_checkpoint_optimizer_rng_and_resumes(tmp_path: Path):
    root, refs = _archive(tmp_path, side=15)
    baseline = tmp_path / "baseline_source"
    config = Namespace(
        root=root, reference_dir=refs, case="Re54/Ma0.3", side=15,
        patch_cells=4, refinement=2, patches_per_snapshot=2,
        boundary_margin=1, max_snapshots=8, train_fraction=0.70,
        validation_fraction=0.15, embargo_snapshots=1,
        epochs=1, resume=False, hidden_dim=16, layers=2, heads=2,
        lr=1.0e-3, seed=37, time_embedding_scale=1000.0,
        condition_mode="interpolated_only", point_sample_constraint=True,
        absolute_coordinate_embedding=False, sparse_attention_backend="scatter",
        checkpointing=False, sample_steps=2, device="cpu", output=baseline,
        tensorboard=False,
    )
    train(config)
    diverse = tmp_path / "diverse_source"
    varied_config = Namespace(**{
        **vars(config), "output": diverse,
        "train_patch_pool_size": 16, "frozen_scales_run": baseline,
    })
    train(varied_config)
    original = {
        (str(source), filename): _sha(source / filename)
        for source in (baseline, diverse)
        for filename in ("manifest.json", "history.json", "last.pt", "best.pt", "scales.pt")
    }
    with pytest.raises(ValueError, match="mode disagrees"):
        prepare(baseline, tmp_path / "incorrect", mode="diverse",
                initial_epochs=1, target_epochs=3, production_profile=False)
    with pytest.raises(ValueError, match="unexpected source"):
        prepare(baseline, tmp_path / "strict_profile", mode="baseline",
                initial_epochs=1, target_epochs=3)

    for mode, source, args in (
        ("baseline", baseline, config),
        ("diverse", diverse, varied_config),
    ):
        output = tmp_path / f"continued_{mode}"
        record = prepare(
            source, output, mode=mode, initial_epochs=1,
            target_epochs=3, production_profile=False,
        )
        assert record["protocol"] == "M44_SR05_IMMUTABLE_CONTINUATION_V1"
        assert record["initial_optimizer_updates"] == 8
        assert record["target_optimizer_updates"] == 24
        assert record["source_run_dir"] == str(source.resolve())
        for filename in ("manifest.json", "history.json", "last.pt", "best.pt", "scales.pt"):
            assert _sha(output / filename) == original[(str(source), filename)]
        assert (output / "continuation_provenance.json").is_file()
        prepare(
            source, output, mode=mode, initial_epochs=1,
            target_epochs=3, resume=True, production_profile=False,
        )
        with pytest.raises(ValueError, match="configuration changed"):
            prepare(
                source, output, mode=mode, initial_epochs=1,
                target_epochs=4, resume=True, production_profile=False,
            )
        result = train(Namespace(**{
            **vars(args), "output": output, "resume": True, "epochs": 3,
        }))
        assert result["epochs_completed"] == 3
        assert result["point_sample_constraint"] is True
        final = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        first = torch.load(source / "last.pt", map_location="cpu", weights_only=True)
        assert final["epoch"] == 2
        assert final["history"][0] == first["history"][0]
        assert final["manifest_sha256"] == first["manifest_sha256"]
        assert final["model_config"] == first["model_config"]
        assert _sha(source / "last.pt") == original[(str(source), "last.pt")]

    for (source_name, filename), digest in original.items():
        assert _sha(Path(source_name) / filename) == digest


def test_continuation_rejects_missing_optimizer_state_and_wrong_epoch(tmp_path: Path):
    root, refs = _archive(tmp_path, side=15)
    source = tmp_path / "source"
    config = Namespace(
        root=root, reference_dir=refs, case="Re54/Ma0.3", side=15,
        patch_cells=4, refinement=2, patches_per_snapshot=2,
        boundary_margin=1, max_snapshots=8, train_fraction=0.70,
        validation_fraction=0.15, embargo_snapshots=1,
        epochs=1, resume=False, hidden_dim=16, layers=2, heads=2,
        lr=1.0e-3, seed=37, time_embedding_scale=1000.0,
        condition_mode="interpolated_only", point_sample_constraint=True,
        absolute_coordinate_embedding=False, sparse_attention_backend="scatter",
        checkpointing=False, sample_steps=2, device="cpu", output=source,
    )
    train(config)
    with pytest.raises(ValueError, match="epoch and history"):
        prepare(source, tmp_path / "bad_epoch", mode="baseline",
                initial_epochs=2, target_epochs=3, production_profile=False)
    last = torch.load(source / "last.pt", map_location="cpu", weights_only=True)
    del last["optimizer_state_dict"]
    torch.save(last, source / "last.pt")
    with pytest.raises(ValueError, match="full optimizer/RNG"):
        prepare(source, tmp_path / "bad_optimizer", mode="baseline",
                initial_epochs=1, target_epochs=3, production_profile=False)
