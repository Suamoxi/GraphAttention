"""CPU-only contracts for opt-in M37 CUDA profiler diagnostics."""
from pathlib import Path

import pytest

from scripts.m43_b1_training_step import one_case_probe


@pytest.mark.parametrize("profile_step,profile_output", [
    (0, Path("prof.json")),
    (3, Path("prof.json")),
    (2, None),
    (None, Path("prof.json")),
])
def test_profile_flags_validate_before_any_gpu_or_hdf5_work(tmp_path, profile_step, profile_output):
    with pytest.raises(ValueError, match="profil"):
        one_case_probe(
            case="test_case", snapshot=tmp_path / "missing.h5",
            mesh=tmp_path / "missing_mesh.h5", expected_side=3,
            seed=42, num_steps=2, output=tmp_path / "report.json",
            operator_profile_step=profile_step,
            operator_profile_output=profile_output,
        )
    assert not (tmp_path / "report.json").exists()
