"""CPU-only regression tests for M48.4e A1 geometry-depth audit."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from scripts import m48_4e_a1_halo_geometry as a1


def _receipt(mode="static", side=48, core=12, *, fine=9232, coarse=2506):
    return dict(mode=mode, fine_side=side, owned_core_side=core,
                owned_nodes_per_patch=core**3, fine_read_nodes=fine,
                coarse_read_nodes=coarse, fine_local_edges=52440,
                fine_dilated_edges=151200, coarse_local_edges=13770,
                coarse_dilated_edges=38802, cross_scale_edges=31158,
                single_graph_tensor_bytes=5_469_664,
                geometry_build_seconds=0.1, status="ok")


def test_origin_profiles_are_distinct_and_periodic():
    assert a1.origin_for(48, "seam") == (45, 17, 46)
    assert a1.origin_for(80, "seam") == (77, 27, 78)
    assert a1.origin_for(48, "interior") == (12, 12, 12)
    assert a1.origin_for(80, "interior") == (20, 20, 20)
    with pytest.raises(ValueError, match="unknown origin"):
        a1.origin_for(48, "corner")


def test_static_baseline_metrics_from_511502_receipt():
    m = a1.geometry_metrics(_receipt(), depth=2)
    assert m["fine_halo_nodes"] == 9232 - 1728
    assert m["fine_read_per_owned"] == pytest.approx(9232 / 1728)
    assert m["fine_read_fraction"] == pytest.approx(9232 / 48**3)
    assert m["coarse_read_fraction"] == pytest.approx(2506 / 24**3)
    assert m["idealized_full_coverage_coarse_read_amplification"] == pytest.approx(
        2506 * 48**3 / (1728 * 24**3))
    assert not m["fine_read_covers_entire_domain"]
    assert m["node_work_proxy_per_patch"] == 2 * (2 * 9232 + 2506)
    assert m["edge_work_proxy_per_patch"] == 2 * (52440 + 151200) + 13770 + 38802 + 31158


def test_dynamic_baseline_and_full_coverage_edge_proxy():
    d = _receipt("dynamic", fine=34751, coarse=5208)
    d.update(fine_local_edges=201114, fine_dilated_edges=588948,
             coarse_local_edges=29064, coarse_dilated_edges=83298,
             cross_scale_edges=115552)
    m = a1.geometry_metrics(d, depth=2)
    assert m["fine_read_per_owned"] == pytest.approx(34751 / 1728)
    assert m["node_work_proxy_per_patch"] == 2 * (2 * 34751 + 2 * 5208)
    assert m["edge_work_proxy_per_patch"] == (
        2 * (201114 + 588948) + 2 * (29064 + 83298) + 2 * 115552)
    assert m["full_grid_edge_work_proxy"] > 0
    assert m["idealized_full_coverage_edge_work_ratio"] > 0


def test_reject_inconsistent_read_sets_and_edges():
    x = _receipt(fine=12, coarse=100)
    with pytest.raises(ValueError, match="invalid exact read-set"):
        a1.geometry_metrics(x, 2)
    x = _receipt()
    x["fine_dilated_edges"] = 18 * 9232 + 1
    with pytest.raises(ValueError, match="degree"):
        a1.geometry_metrics(x, 2)


def test_summary_ranges_both_origins_and_strict_missing_failure():
    r = _receipt()
    r.update(a1.geometry_metrics(r, 2), depth=2, origin_kind="seam")
    s = {**r, "origin_kind": "interior", "fine_read_nodes": 9000,
         "fine_read_per_owned": 9000 / 1728}
    a = a1.summarize([r, s], 2)
    assert a["status"] == "PASS"
    assert a["aggregates"][0]["fine_read_nodes_min"] == 9000
    assert a["aggregates"][0]["fine_read_nodes_max"] == 9232
    assert a1.summarize([r], 2)["status"] == "ERROR"
    assert a1.summarize([r, {"status": "worker_timeout"}], 2)["status"] == "ERROR"


def test_run_is_metadata_only_and_writes_trial_receipts(monkeypatch, tmp_path):
    monkeypatch.setattr(a1, "authentic_teacher_grids", lambda *_: (
        {48: {"periodic_length": 1.0}, 80: {"periodic_length": 2.0}},
        {"teacher_snapshots": 44, "teacher_field_payloads_read": False}))
    monkeypatch.setattr(a1.subprocess, "check_output", lambda *a, **k: "test-commit\n")
    calls = []

    def fake_run(cmd, **kwargs):
        assert kwargs["check"] is False
        assert "--mode" in cmd and "--worker" in cmd
        origin = cmd[cmd.index("--origin-kind") + 1]
        period = float(cmd[cmd.index("--periodic-length") + 1])
        assert period in (1.0, 2.0)
        path = Path(cmd[cmd.index("--worker-output") + 1])
        receipt = _receipt()
        receipt.update(a1.geometry_metrics(receipt, 2))
        receipt["origin_kind"] = origin
        path.write_text(json.dumps(receipt))
        calls.append((origin, period))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(a1.subprocess, "run", fake_run)
    args = SimpleNamespace(mode="static", depths="2", core_sides="12",
                           fine_sides="48,80", origins="seam,interior",
                           worker_timeout_seconds=5, max_worker_virtual_gib=32,
                           output_dir=tmp_path / "audit", ms_sr_root=tmp_path,
                           normalization_file=tmp_path / "frozen.json")
    summary = a1.run(args)
    assert summary["status"] == "PASS"
    assert summary["successful_trials"] == 4
    assert len(summary["aggregates"]) == 2
    assert calls == [("seam", 1.0), ("interior", 1.0),
                     ("seam", 2.0), ("interior", 2.0)]
    assert (args.output_dir / "trials.csv").is_file()
    assert (args.output_dir / "summary.json").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        a1.run(args)


def test_integration_exact_seam_matches_calypso_phase_a_when_repo_available():
    pytest.importorskip("graph_attention.geometry.two_scale_targeted")
    from scripts.m48_4e_gpu_capacity import build_geometry
    _, static = build_geometry(48, 12, 2, "static", 1.0)
    _, dynamic = build_geometry(48, 12, 2, "dynamic", 1.0)
    assert (static["fine_read_nodes"], static["coarse_read_nodes"]) == (9232, 2506)
    assert (dynamic["fine_read_nodes"], dynamic["coarse_read_nodes"]) == (34751, 5208)
    _, interior = build_geometry(48, 12, 2, "static", 1.0, origin=(12, 12, 12))
    assert interior["owned_nodes_per_patch"] == 12**3
    assert interior["origin"] == [12, 12, 12]
