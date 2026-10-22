"""CPU-only tests for the independent M43 129^3 CUDA profile instrumentation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import types

import pytest
import torch

from graph_attention.training.m43_kernel_profile_report import (
    format_table,
    kernel_groups,
    operator_groups,
    phase_of_event,
)
from scripts.profile_m43_b_129_kernels import (
    _chosen_index,
    _validate_args,
    inspect,
)


def _event(name, parent=None, kernels=()):
    return types.SimpleNamespace(
        name=name, cpu_parent=parent, kernels=kernels
    )


def _kernel(name, us):
    return types.SimpleNamespace(name=name, duration=us)


def _op(name, n, gpu_us, cpu_us):
    return types.SimpleNamespace(
        key=name, count=n, self_device_time_total=gpu_us,
        self_cpu_time_total=cpu_us,
    )


def test_kernel_aggregation_attributed_to_nested_phase_and_sorted() -> None:
    backward = _event("m43_profile/backward_including_checkpoint_recompute")
    op = _event("aten::index_add_", backward, [
        _kernel("scatter_add_cuda_kernel", 1500),
        _kernel("scatter_add_cuda_kernel", 500),
    ])
    forward = _event("m43_profile/model_forward_loss")
    op2 = _event("aten::sparse_sampled_addmm", forward, [
        _kernel("sampled_addmm_cuda_kernel", 1000),
    ])
    summary = kernel_groups([backward, op, forward, op2], limit=10)
    assert summary["kernel_count"] == 3
    assert summary["sum_cuda_kernel_seconds"] == pytest.approx(.003)
    assert summary["kernel_time_seconds_by_phase"] == {
        "backward_including_checkpoint_recompute": pytest.approx(.002),
        "model_forward_loss": pytest.approx(.001),
    }
    top = summary["top_cuda_kernels"]
    assert top[0]["name"] == "scatter_add_cuda_kernel"
    assert top[0]["launches"] == 2
    assert top[0]["fraction_of_kernel_duration"] == pytest.approx(2/3)
    assert phase_of_event(op) == "backward_including_checkpoint_recompute"


def test_no_cuda_events_is_reported_without_fabricated_device_time() -> None:
    result = kernel_groups([_event("aten::sum")])
    assert result["kernel_count"] == 0
    assert result["sum_cuda_kernel_seconds"] == 0
    assert result["top_cuda_kernels"] == []


def test_operator_sorting_microsecond_to_second_conversion() -> None:
    group = operator_groups([
        _op("aten::scatter_add_", 100, 310000, 120),
        _op("aten::mm", 4, 20000, 1000),
    ])
    top = group["top_pytorch_operators"]
    assert group["sum_exclusive_gpu_operator_seconds"] == pytest.approx(.33)
    assert top[0]["name"] == "aten::scatter_add_"
    assert top[0]["self_gpu_seconds"] == pytest.approx(.31)
    assert "scatter_add" in format_table(top, name_key="name",
                                         seconds_key="self_gpu_seconds")


def test_snapshot_selection_uses_full_filename_and_129_side_only() -> None:
    from graph_attention.data.multicase_avbp import HITCaseSpec
    fake = types.SimpleNamespace(
        cases=(HITCaseSpec("Re94/Ma0.2", 129),
               HITCaseSpec("Re54/Ma0.2", 65)),
        indices_by_case={"Re94/Ma0.2": (0, 1), "Re54/Ma0.2": (2,)},
        sample_specs=(
            types.SimpleNamespace(
                snapshot_file=Path("/fake/solut_hit_00000001.h5")
            ),
            types.SimpleNamespace(
                snapshot_file=Path("/fake/solut_hit_00000051.h5")
            ),
            types.SimpleNamespace(
                snapshot_file=Path("/fake/solut_hit_00000051.h5")
            ),
        ),
    )
    assert _chosen_index(fake, case_id="Re94/Ma0.2", snapshot_number=51) == 1
    with pytest.raises(ValueError, match="129"):
        _chosen_index(fake, case_id="Re54/Ma0.2", snapshot_number=51)
    with pytest.raises(ValueError, match="not uniquely"):
        _chosen_index(fake, case_id="Re94/Ma0.2", snapshot_number=52)


def _synthetic_input(tmp_path):
    run_dir = tmp_path / "train"
    root_dir = tmp_path / "dns"
    ref_dir = tmp_path / "refs"
    out_dir = tmp_path / "profile_out"
    run_dir.mkdir()
    root_dir.mkdir()
    ref_dir.mkdir()
    from scripts.train_m43_b_multires import M37_RECIPE
    content = {
        "mode": "train",
        "revision": "ff851882091e4c4fd4ccfd5ecb69cb02b8bd0d90",
        "root": str(root_dir),
        "model": M37_RECIPE,
        "seed": 42,
        "accumulation": 6,
        "checkpointing": True,
        "references": {"case": {"file": str(ref_dir / "case.yaml")}},
        "physical_nondimensionalization": True,
        "time_embedding_scale": 1000.,
    }
    raw = json.dumps(content, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(raw.encode()).hexdigest()
    (run_dir / "run_signature.json").write_text(
        json.dumps({"content": content, "sha256": sha})
    )
    (run_dir / "last.pt").write_bytes(b"not-loaded-in-inspect")
    (run_dir / "standardizers.pt").write_bytes(b"not-loaded-in-inspect")
    args = types.SimpleNamespace(
        mode="inspect",
        root=root_dir,
        reference_dir=ref_dir,
        train_run_dir=run_dir,
        output_dir=out_dir,
        case_id="Re94/Ma0.2",
        snapshot_number=51,
    )
    return args


def test_inspect_is_read_only_and_checks_original_signature(tmp_path) -> None:
    args = _synthetic_input(tmp_path)
    original_bytes = (args.train_run_dir / "run_signature.json").read_bytes()
    payload = inspect(args)
    assert payload["status"] == "read_only_profile_inspected"
    assert payload["source_training_revision"].startswith("ff85188")
    assert payload["scientific_parameters"]["accumulation"] == 6
    assert not args.output_dir.exists()
    assert original_bytes == (args.train_run_dir / "run_signature.json").read_bytes()
    bad = json.loads((args.train_run_dir / "run_signature.json").read_text())
    bad["sha256"] = "incorrect"
    (args.train_run_dir / "run_signature.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="SHA256"):
        inspect(args)


def test_inspect_rejects_reference_and_data_root_mismatch(tmp_path) -> None:
    args = _synthetic_input(tmp_path)
    args.reference_dir = tmp_path / "another_ref_folder"
    with pytest.raises(ValueError, match="reference directory"):
        inspect(args)
    args.reference_dir = tmp_path / "refs"
    args.root = tmp_path / "another_root_folder"
    with pytest.raises(ValueError, match="archive path"):
        inspect(args)


def test_output_path_must_be_separate_from_training_dir(tmp_path) -> None:
    args = _synthetic_input(tmp_path)
    args.output_dir = args.train_run_dir / "profiles"
    with pytest.raises(ValueError, match="separate"):
        _validate_args(args)
    args.output_dir = args.train_run_dir
    with pytest.raises(ValueError, match="must not overwrite"):
        _validate_args(args)


def test_profiler_reports_cpu_scopes_without_requiring_gpu() -> None:
    from torch.profiler import ProfilerActivity, profile, record_function

    x = torch.randn(12, 12, requires_grad=True)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with record_function("m43_profile/model_forward_loss"):
            loss = x.square().sum()
        with record_function("m43_profile/backward_including_checkpoint_recompute"):
            loss.backward()
    event_phases = {phase_of_event(event) for event in prof.events()}
    assert "model_forward_loss" in event_phases
    assert "backward_including_checkpoint_recompute" in event_phases
    assert operator_groups(prof.key_averages())["top_pytorch_operators"]
