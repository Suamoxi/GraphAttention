"""A2 full-domain vs complete-tile parity and benchmark-control regression tests."""
from __future__ import annotations

from types import SimpleNamespace
from copy import deepcopy
import json

import pytest
import torch

from scripts import m48_4e_a2_full_vs_tiled as a2


def _fake_geometry(owned):
    return SimpleNamespace(
        fine_global=torch.tensor(owned, dtype=torch.long),
        owned_fine_local=torch.arange(len(owned), dtype=torch.long),
    )


def test_exact_tiling_geometry_and_cap():
    assert len(a2.tile_origins(48, 24, 8)) == 8
    assert len(a2.tile_origins(80, 20, 64)) == 64
    assert (0, 0, 0) in a2.tile_origins(48, 24, 8)
    assert (24, 24, 24) in a2.tile_origins(48, 24, 8)
    with pytest.raises(ValueError, match="positive divisor"):
        a2.tile_origins(48, 20, 100)
    with pytest.raises(ValueError, match="needs 64 tiles"):
        a2.tile_origins(80, 20, 8)


def test_loss_ownership_requires_each_node_exactly_once():
    complete = [_fake_geometry(range(0, 256)), _fake_geometry(range(256, 512))]
    a2.verify_ownership(8, complete)
    with pytest.raises(RuntimeError, match="exactly once"):
        a2.verify_ownership(8, [_fake_geometry(range(0, 256))])
    with pytest.raises(RuntimeError, match="exactly once"):
        a2.verify_ownership(8, [complete[0], complete[0], complete[1]])
    with pytest.raises(RuntimeError, match="invalid owned"):
        a2.verify_ownership(8, [_fake_geometry([1, 1])])


def test_array_parity_detects_swapped_or_corrupt_values():
    a = torch.arange(30, dtype=torch.float32).reshape(6, 5)
    assert a2.compare_arrays(a, a.clone(), atol=0, rtol=0)["passed"]
    b = a.clone()
    b[0, 0] += 0.01
    assert not a2.compare_arrays(a, b, atol=1e-5, rtol=1e-5)["passed"]
    assert not a2.compare_arrays(a, b[:3], atol=1, rtol=1)["passed"]


def test_saved_initial_output_and_gradient_parity(tmp_path):
    a = {"seed": 484, "parameters": [("p", 3)],
         "prediction": torch.arange(5, dtype=torch.float32),
         "gradient": torch.tensor([0.2, 0.3, 0.4])}
    b = deepcopy(a)
    x, y = tmp_path / "full.pt", tmp_path / "tiled.pt"
    torch.save(a, x)
    torch.save(b, y)
    assert a2.compare_verification(x, y)["passed"]
    b["gradient"] = torch.zeros_like(b["gradient"])
    torch.save(b, y)
    assert not a2.compare_verification(x, y)["passed"]
    b = deepcopy(a)
    b["seed"] = 485
    torch.save(b, y)
    with pytest.raises(RuntimeError, match="different model"):
        a2.compare_verification(x, y)


def _receipt(strategy="full", duration=1.0):
    return {"mode": "static", "fine_side": 48, "depth": 2,
            "core_side": 24, "strategy": strategy, "status": "ok",
            "supervised_nodes_per_update": 48**3,
            "mean_wall_update_seconds": duration,
            "peak_reserved_bytes": 1_000_000,
            "geometry_build_seconds": 0.5,
            "geometry_transfer_seconds": 0.1}


def test_summary_fails_closed_and_uses_same_coverage():
    f, t = _receipt(), _receipt("tiled", duration=8.0)
    parity = {"passed": True, "prediction": {}, "gradient": {}}
    row = a2.summarize_pair(f, t, parity)
    assert row["status"] == "PASS"
    assert row["full_over_tiled_speedup"] == pytest.approx(8.0)
    assert a2.summarize_pair(f, t, None)["status"] == "PARITY_FAILED"
    t["supervised_nodes_per_update"] -= 1
    assert a2.summarize_pair(f, t, parity)["status"] == "COVERAGE_MISMATCH"
    t["status"] = "cuda_or_host_oom"
    assert a2.summarize_pair(f, t, parity)["status"] == "UNQUALIFIED"
    t["status"] = "worker_crash"
    assert a2.summarize_pair(f, t, parity)["status"] == "ERROR"


def test_metadata_run_records_both_strategies_and_initial_verification(monkeypatch, tmp_path):
    monkeypatch.setattr(a2, "authentic_teacher_grids", lambda *_: (
        {48: {"fine_side": 48, "periodic_length": 1.0}},
        {"teacher_snapshots": 44, "teacher_field_payloads_read": False}))
    monkeypatch.setattr(a2.subprocess, "check_output", lambda *a, **k: "test-commit")
    strategies = []

    def fake_worker(args, grid, depth, core, strategy, folder):
        strategies.append(strategy)
        id_ = f"static_n48_d{depth}_p{core}_{strategy}"
        ver = folder / "verification" / f"{id_}.pt"
        ver.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"seed": 484, "parameters": [("x", 3)],
                    "prediction": torch.zeros(48**3, 5),
                    "gradient": torch.zeros(3)}, ver)
        return {**_receipt(strategy, duration=8.0 if strategy == "tiled" else 1.0),
                "trial_id": id_}

    monkeypatch.setattr(a2, "_run_worker", fake_worker)
    args = SimpleNamespace(mode="static", fine_sides="48", depths="2", core_sides="24",
                           width=64, heads=4, warmup=1, measure=2, seed=484,
                           max_tiles=8, max_gpu_fraction=0.85,
                           max_host_graph_gib=12, worker_timeout_seconds=300,
                           output_dir=tmp_path / "a2", ms_sr_root=tmp_path,
                           normalization_file=tmp_path / "scale.json")
    report = a2.run(args)
    assert strategies == ["full", "tiled"]
    assert report["status"] == "PASS" and report["qualified_pairs"] == 1
    assert report["comparisons"][0]["full_over_tiled_speedup"] == pytest.approx(8)
    assert (args.output_dir / "comparisons.csv").is_file()
    assert (args.output_dir / "trials.csv").is_file()
    assert json.loads((args.output_dir / "summary.json").read_text())["status"] == "PASS"
    with pytest.raises(FileExistsError, match="overwrite"):
        a2.run(args)


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_exact_small_periodic_tiling_matches_full_predictions_and_gradients(mode):
    pytest.importorskip("graph_attention.geometry.two_scale_targeted")
    from graph_attention.geometry.two_scale_targeted import build_targeted_two_scale_patch
    from graph_attention.models.two_scale_dinat import TwoScaleDiNAT
    from scripts.m48_ms_sr_train import _owned_cube

    torch.manual_seed(401)
    n, p, d = 8, 4, 1
    full = a2.full_geometry(n, 1.0)
    tiles = []
    for origin in a2.tile_origins(n, p, 8):
        _, geom = build_targeted_two_scale_patch(
            coarse_side=n//2, refinement=2,
            owned_fine=_owned_cube(n, p, origin),
            fine_pre_layers=d, fine_post_layers=d,
            coarse_static_layers=d,
            coarse_dynamic_layers=d if mode == "dynamic" else 0,
        )
        tiles.append(geom)
    a2.verify_ownership(n, tiles)
    model = TwoScaleDiNAT(mode=mode, fine_width=8, coarse_width=8,
                          num_heads=2, fine_pre_layers=d,
                          fine_post_layers=d, coarse_static_layers=d,
                          coarse_dynamic_layers=d)
    other = deepcopy(model)
    global_input = a2.make_global_inputs(n, 484, torch.device("cpu"))
    opt_full = torch.optim.AdamW(model.parameters(), lr=0.0)
    opt_tile = torch.optim.AdamW(other.parameters(), lr=0.0)
    _, a = a2.gpu_step(model, opt_full, global_input, [full], "full", capture=True)
    _, b = a2.gpu_step(other, opt_tile, global_input, tiles, "tiled", capture=True)
    torch.testing.assert_close(a["prediction"], b["prediction"], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(a["gradient"], b["gradient"], rtol=1e-2, atol=1e-4)
