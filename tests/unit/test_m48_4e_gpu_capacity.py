"""Phase-A capacity and packed-graph safety tests (CUDA is not required)."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from scripts import m48_4e_gpu_capacity as bench


@dataclass(frozen=True)
class _Geometry:
    fine_global: torch.Tensor
    coarse_global: torch.Tensor
    owned_fine_local: torch.Tensor
    fine_coords: torch.Tensor
    coarse_coords: torch.Tensor
    fine_edges: torch.Tensor
    fine_dilated: torch.Tensor
    coarse_edges: torch.Tensor
    coarse_dilated: torch.Tensor
    coarse_to_fine: torch.Tensor
    fine_to_coarse: torch.Tensor
    cross_weights: torch.Tensor
    periodic_length: float
    coarse_spacing: float
    refinement: int


def _fixture_geometry():
    fine_edges = torch.tensor([[0, 1], [1, 2]])
    coarse_edges = torch.tensor([[0], [1]])
    up = torch.tensor([[0, 1], [0, 2]])
    return _Geometry(
        fine_global=torch.tensor([0, 1, 7]),
        coarse_global=torch.tensor([0, 3]),
        owned_fine_local=torch.tensor([0, 2]),
        fine_coords=torch.tensor([[0., 0., 0.], [0.1, 0., 0.], [0.9, 0., 0.]]),
        coarse_coords=torch.tensor([[0., 0., 0.], [0.5, 0., 0.]]),
        fine_edges=fine_edges, fine_dilated=fine_edges.flip(0).contiguous(),
        coarse_edges=coarse_edges, coarse_dilated=coarse_edges.flip(0).contiguous(),
        coarse_to_fine=up, fine_to_coarse=up.flip(0).contiguous(),
        cross_weights=torch.tensor([0.5, 0.5]),
        periodic_length=1.0, coarse_spacing=0.5, refinement=2,
    )


def test_parse_candidates_and_reject_invalid_values():
    assert bench.positive_ints("64,4,16") == [4, 16, 64]
    for bad in ("", "0,2", "-1,2", "4,4", "x,8"):
        with pytest.raises((ValueError, TypeError)):
            bench.positive_ints(bad)


def test_packed_edges_and_masks_preserve_disconnected_graphs():
    geometry = _fixture_geometry()
    packed, fbatch, cbatch = bench.pack_repeated_geometry(geometry, 3)
    assert fbatch.tolist() == [0]*3 + [1]*3 + [2]*3
    assert cbatch.tolist() == [0]*2 + [1]*2 + [2]*2
    assert packed.owned_fine_local.tolist() == [0, 2, 3, 5, 6, 8]
    torch.testing.assert_close(packed.fine_edges,
        torch.tensor([[0, 1, 3, 4, 6, 7], [1, 2, 4, 5, 7, 8]]))
    torch.testing.assert_close(packed.coarse_to_fine,
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 2, 3, 5, 6, 8]]))
    bench.verify_packed_graph(packed, fbatch, cbatch)
    # A cross-sample incoming edge must be rejected before CUDA execution.
    bad = replace(packed, fine_edges=torch.tensor([[2], [3]]))
    with pytest.raises(RuntimeError, match="cross-sample"):
        bench.verify_packed_graph(bad, fbatch, cbatch)


def test_repeated_geometry_counts_scale_linearly():
    geom = _fixture_geometry()
    assert bench.graph_tensor_bytes(geom) > 0
    for count in (1, 2, 5):
        packed, _, _ = bench.pack_repeated_geometry(geom, count)
        for name in bench.GRAPH_KEYS:
            assert getattr(packed, name).numel() == getattr(geom, name).numel() * count


def test_best_capacity_is_per_mesh_core_and_excludes_unqualified():
    trials = [
        dict(status="ok", mode="static", fine_side=48, owned_core_side=12,
             gpu_microbatch=4, owned_nodes_per_second=100., network_width=32,
             network_depth=1, peak_allocated_bytes=1, peak_reserved_bytes=2,
             gpu_total_memory_bytes=1000),
        dict(status="ok", mode="static", fine_side=48, owned_core_side=12,
             gpu_microbatch=8, owned_nodes_per_second=200., network_width=32,
             network_depth=1, peak_allocated_bytes=3, peak_reserved_bytes=4,
             gpu_total_memory_bytes=1000),
        dict(status="gpu_headroom_exceeded", mode="static", fine_side=48,
             owned_core_side=12, gpu_microbatch=16, owned_nodes_per_second=5000.),
        dict(status="ok", mode="dynamic", fine_side=80, owned_core_side=16,
             gpu_microbatch=2, owned_nodes_per_second=50., network_width=64,
             network_depth=2, peak_allocated_bytes=5, peak_reserved_bytes=7,
             gpu_total_memory_bytes=1000),
    ]
    winners = bench._best_configurations(trials)
    assert len(winners) == 2
    by_mode = {row["mode"]: row for row in winners}
    assert by_mode["static"]["gpu_microbatch"] == 8
    assert by_mode["static"]["owned_nodes_per_second"] == 200.
    assert by_mode["dynamic"]["gpu_microbatch"] == 2


def test_worker_missing_receipt_is_explicit_failure():
    result = bench._classify_worker(-9, "", "killed", None)
    assert result["status"] == "worker_crash"
    assert result["returncode"] == -9
    assert bench._classify_worker(0, "", "", {"status": "cuda_oom"}) == {"status": "cuda_oom"}


def test_worker_oom_writes_result_without_demanding_gpu(monkeypatch, tmp_path):
    def fail(_args):
        raise torch.cuda.OutOfMemoryError("expected capacity limit")
    monkeypatch.setattr(bench, "_gpu_trial", fail)
    path = tmp_path / "trial.json"
    bench.run_worker(SimpleNamespace(worker_output=path))
    data = json.loads(path.read_text())
    assert data["status"] == "cuda_oom"
    assert "expected capacity" in data["error"]


def test_authentication_rejects_mismatched_teacher_selection(tmp_path, monkeypatch):
    manifest = tmp_path / "dataset" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_bytes(b"sealed M48 manifest fixture")
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"manifest_sha256": sha, "source_count": 44,
                                  "selection_sha256": "right"}))
    source = ModuleType("graph_attention.data.ms_sr_training")
    adapter = ModuleType("graph_attention.data.ms_sr_adapter")
    adapter.MANIFEST_SHA256 = sha
    source.select_role_records = lambda *a, **kw: ([], "wrong")
    # Keep the local test independent of a GraphAttention installation.
    for key in ("graph_attention", "graph_attention.data"):
        if key not in sys.modules:
            monkeypatch.setitem(sys.modules, key, ModuleType(key))
    monkeypatch.setitem(sys.modules, "graph_attention.data.ms_sr_training", source)
    monkeypatch.setitem(sys.modules, "graph_attention.data.ms_sr_adapter", adapter)
    with pytest.raises(RuntimeError, match="44 authenticated"):
        bench.authentic_teacher_grids(tmp_path, frozen)


def _valid_teacher_records_with_variable_lengths():
    records = []
    for side, group in ((48, "ReT060"), (80, "ReT100")):
        for case_index in range(11):
            # Same grid topology; physically different periodic box lengths.
            length = (1 + 0.1 * case_index) * (0.001 if side == 48 else 0.002)
            for snapshot in (5, 6):
                records.append({
                    "case": f"{group}/Mt{case_index:02d}_Xi0.0",
                    "snapshot_id": snapshot,
                    "target_shape": [side] * 3,
                    "input_shape": [side // 2] * 3,
                    "fine_spacing": [length / side] * 3,
                    "coarse_spacing": [length / (side // 2)] * 3,
                })
    return records


def test_valid_teacher_metadata_accepts_distinct_periods_for_equal_resolution():
    grids = bench.summarize_teacher_grids(_valid_teacher_records_with_variable_lengths())
    assert sorted(grids) == [48, 80]
    for side in (48, 80):
        grid = grids[side]
        assert grid["teacher_snapshots"] == 22
        assert grid["teacher_case_count"] == 11
        assert len(grid["distinct_periodic_lengths"]) == 11
        assert grid["periodic_length_min"] < grid["periodic_length_max"]
        assert grid["periodic_length"] == grid["periodic_length_min"]
        assert grid["representative_teacher_case"].startswith("ReT")
        assert len(grid["case_periodic_lengths"]) == 11
        assert all(len(lengths) == 2 for lengths in grid["case_periodic_lengths"].values())


def test_representative_length_is_authenticated_case_not_grid_minimum():
    records = _valid_teacher_records_with_variable_lengths()
    # Deliberately put a longer ReT060 case first: the benchmark should
    # retain its actual period, not silently reinterpret it as grid minimum.
    index = next(i for i, rec in enumerate(records)
                 if rec["case"] == "ReT060/Mt10_Xi0.0")
    record = records.pop(index)
    records.insert(0, record)
    grids = bench.summarize_teacher_grids(records)
    assert grids[48]["periodic_length"] > grids[48]["periodic_length_min"]
    assert grids[48]["representative_teacher_case"] == record["case"]
    assert grids[48]["teacher_snapshots"] == 22


@pytest.mark.parametrize("tamper", ("fine_axis", "coarse_axis", "negative", "nonfinite", "shape"))
def test_each_teacher_still_requires_consistent_physical_geometry(tamper):
    records = _valid_teacher_records_with_variable_lengths()
    record = records[-1]
    if tamper == "fine_axis":
        record["fine_spacing"][1] *= 1.1
    elif tamper == "coarse_axis":
        record["coarse_spacing"][2] *= 1.1
    elif tamper == "negative":
        record["fine_spacing"] = [-1.] * 3
    elif tamper == "nonfinite":
        record["coarse_spacing"][0] = float("nan")
    else:
        record["input_shape"][0] += 1
    with pytest.raises(RuntimeError, match="(periodic lengths|grid dimensions)"):
        bench.summarize_teacher_grids(records)


def test_authenticated_metadata_accepts_variable_periods_and_checks_seal(
    tmp_path, monkeypatch,
):
    manifest = tmp_path / "dataset" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_bytes(b"sealed variable-length teacher fixture")
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"manifest_sha256": sha, "source_count": 44,
                                  "selection_sha256": "teacher-seal"}))
    source = ModuleType("graph_attention.data.ms_sr_training")
    adapter = ModuleType("graph_attention.data.ms_sr_adapter")
    adapter.MANIFEST_SHA256 = sha
    source.select_role_records = lambda *a, **kw: (
        _valid_teacher_records_with_variable_lengths(), "teacher-seal")
    for key in ("graph_attention", "graph_attention.data"):
        if key not in sys.modules:
            monkeypatch.setitem(sys.modules, key, ModuleType(key))
    monkeypatch.setitem(sys.modules, "graph_attention.data.ms_sr_training", source)
    monkeypatch.setitem(sys.modules, "graph_attention.data.ms_sr_adapter", adapter)
    grids, receipt = bench.authentic_teacher_grids(tmp_path, frozen)
    assert {side: g["teacher_snapshots"] for side, g in grids.items()} == {48: 22, 80: 22}
    assert receipt["teacher_snapshots"] == 44
    assert receipt["teacher_field_payloads_read"] is False
    assert receipt["selection_sha256"] == "teacher-seal"


def test_metadata_only_cli_needs_no_cuda_or_output_directory(monkeypatch, capsys):
    monkeypatch.setattr(bench, "authentic_teacher_grids", lambda *_args: (
        {48: {"fine_side": 48, "periodic_length": 0.001}},
        {"teacher_snapshots": 44, "teacher_field_payloads_read": False}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    bench.main(["--mode", "static", "--metadata-only"])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "PASS"
    assert receipt["grids"]["48"]["periodic_length"] == 0.001


def test_real_m48_packer_equivalence_when_repo_available():
    pytest.importorskip("graph_attention.geometry.two_scale_targeted")
    from graph_attention.geometry.two_scale_targeted import build_targeted_two_scale_patch
    from scripts.m48_ms_sr_train import _owned_cube
    from scripts.m48_4c_patch_batch import pack_patches

    owned = _owned_cube(8, 2, (7, 1, 7))
    _, geom = build_targeted_two_scale_patch(
        coarse_side=4, refinement=2, owned_fine=owned,
        fine_pre_layers=1, fine_post_layers=1, coarse_static_layers=1,
        coarse_dynamic_layers=1,
    )
    count = 3
    packed, fine_idx, coarse_idx = bench.pack_repeated_geometry(geom, count)
    samples = [{"geometry": geom, "state": torch.zeros((len(geom.fine_global), 5)),
                "baseline": torch.zeros((len(geom.fine_global), 5)),
                "coarse_state": torch.zeros((len(geom.coarse_global), 5)),
                "target": torch.zeros((len(owned), 5)), "fm_time": 0.2}
               for _ in range(count)]
    reference = pack_patches(samples, torch.device("cpu"))
    for field in fields(geom):
        a, b = getattr(packed, field.name), getattr(reference["geometry"], field.name)
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        else:
            assert a == b
    torch.testing.assert_close(fine_idx, reference["fine_batch_index"])
    torch.testing.assert_close(coarse_idx, reference["coarse_batch_index"])


def test_sweep_persists_trials_and_stops_after_oom(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_args: "test_gpu")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_args:
                        SimpleNamespace(total_memory=1000))
    monkeypatch.setattr(bench, "authentic_teacher_grids", lambda *_args: (
        {48: {"fine_side": 48, "coarse_side": 24, "periodic_length": 1.0}},
        {"teacher_snapshots": 44, "teacher_cases": 22}))
    geometry = _fixture_geometry()
    monkeypatch.setattr(bench, "build_geometry", lambda *_args: (
        geometry, {"single_graph_tensor_bytes": bench.graph_tensor_bytes(geometry)}))
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *_args, **_kwargs:
                        "test-commit\n")
    calls = []

    def fake_run(command, **_kwargs):
        args = list(command)
        micro = int(args[args.index("--microbatch") + 1])
        output = Path(args[args.index("--worker-output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        calls.append(micro)
        if micro == 1:
            receipt = dict(status="ok", owned_nodes_per_second=125.,
                           network_width=8, network_depth=1,
                           peak_allocated_bytes=250, peak_reserved_bytes=300,
                           gpu_total_memory_bytes=1000)
        else:
            receipt = dict(status="cuda_oom", error="synthetic capacity limit")
        output.write_text(json.dumps(receipt))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    args = SimpleNamespace(mode="static", depth=1, width=8, heads=2,
        warmup=1, measure=2, seed=484, max_gpu_fraction=0.85,
        max_host_pack_gib=4, host_overhead_multiplier=3,
        worker_timeout_seconds=5,
        fine_sides="48", core_sides="12", microbatches="1,2,4",
        output_dir=tmp_path / "capacity", ms_sr_root=tmp_path,
        normalization_file=tmp_path / "frozen.json")
    summary = bench.run_sweep(args)
    assert calls == [1, 2]
    assert summary["status"] == "FINISHED"
    assert summary["no_training_performed"]
    assert summary["best_safe_per_grid_core"][0]["gpu_microbatch"] == 1
    assert [row["status"] for row in summary["trials"]] == ["ok", "cuda_oom"]
    assert (args.output_dir / "trials.csv").exists()
    assert (args.output_dir / "geometry" / "static_n48_p12.pt").exists()


def test_sweep_rejects_missing_gpu_before_creating_output(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA"):
        bench.run_sweep(SimpleNamespace(output_dir=tmp_path / "never"))
    assert not (tmp_path / "never").exists()
