"""A2 full-domain vs complete-tile parity and benchmark-control regression tests."""
from __future__ import annotations

from types import SimpleNamespace, ModuleType
import sys
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


def _active_receipt(mode="static", seed=484):
    names = [(f"{group.split('.')[0]}.blocks.0.attention.a.weight", 1)
             if ".attention" in group else (group + ".weight", 1)
             for group in a2.required_gradient_groups(mode)]
    n = len(names)
    prediction = torch.tensor([[0.05, -0.13, 0.20, 0.07, 0.14],
                               [0.11, 0.03, -0.09, -0.06, 0.01]], dtype=torch.float32)
    step = {"prediction": prediction, "gradient": torch.arange(1, n+1, dtype=torch.float32) * .001,
            "parameters": names, "prediction_rms": 0.2,
            "prediction_spatial_std": 0.2,
            "gradient_group_norms": {x: .1 for x in a2.required_gradient_groups(mode)}}
    return {"seed": seed, "mode": mode, "initialization": a2.ACTIVE_INIT,
            "steps": [deepcopy(step), deepcopy(step)]}


def test_two_step_active_parity_and_fail_closed(tmp_path):
    a = _active_receipt()
    b = deepcopy(a)
    x, y = tmp_path / "full.pt", tmp_path / "tiled.pt"
    torch.save(a, x)
    torch.save(b, y)
    check = a2.compare_verification(x, y)
    assert check["passed"] and len(check["steps"]) == 2
    assert all(s["active_attention_gradients"] for s in check["steps"])
    b["steps"][1]["prediction"][0,0] += .1
    torch.save(b, y)
    assert not a2.compare_verification(x, y)["passed"]
    b = deepcopy(a)
    b["steps"][0]["gradient"] = torch.zeros_like(b["steps"][0]["gradient"])
    torch.save(b, y)
    assert not a2.compare_verification(x, y)["passed"]
    b = deepcopy(a)
    b["steps"][0]["prediction"].zero_()
    b["steps"][0]["prediction_rms"] = 1.0  # Forged metadata cannot bypass tensor check.
    torch.save(b, y)
    assert not a2.compare_verification(x, y)["passed"]
    b = deepcopy(a)
    b["initialization"] = "zero_init"
    torch.save(b, y)
    assert not a2.compare_verification(x, y)["passed"]
    b = deepcopy(a)
    b["seed"] = 485
    torch.save(b, y)
    with pytest.raises(RuntimeError, match="different synthetic"):
        a2.compare_verification(x, y)


def test_dynamic_active_gradient_path_groups(tmp_path):
    a = _active_receipt("dynamic")
    x, y = tmp_path / "f.pt", tmp_path / "t.pt"
    torch.save(a, x)
    torch.save(a, y)
    assert a2.compare_verification(x, y)["passed"]
    for group in ("coarse_dynamic.attention", "down"):
        corrupt = deepcopy(a)
        offset = a2.required_gradient_groups("dynamic").index(group)
        corrupt["steps"][0]["gradient"][offset] = 0
        torch.save(corrupt, y)
        assert not a2.compare_verification(x, y)["passed"]


def test_parity_gate_80_requires_matching_active_48_summary(tmp_path):
    d = {"schema": a2.SCHEMA, "status": "PASS", "source_commit": "newcommit",
         "active_initialization": a2.ACTIVE_INIT,
         "required_verification_steps": a2.VERIFICATION_STEPS,
         "requested": {"mode": "static", "width": 64, "heads": 4},
         "provenance": {"manifest_sha256": "pinned"},
         "comparisons": [{"fine_side": 48, "depth": i, "status": "PASS",
                          "parity": {"passed": True, "steps": [
                              {"passed": True, "active_attention_gradients": True},
                              {"passed": True, "active_attention_gradients": True}]}}
                         for i in (2,4,8)]}
    path = tmp_path / "parity.json"
    path.write_text(json.dumps(d))
    kw = dict(mode="static", depths=[2, 4], width=64, heads=4,
              provenance={"manifest_sha256": "pinned"}, current_commit="newcommit")
    assert a2.validate_48_parity_gate(path, **kw)["status"] == "PASS"
    with pytest.raises(ValueError, match="requires"):
        a2.validate_48_parity_gate(None, **kw)
    for changed in (dict(mode="dynamic"), dict(depths=[6]),
                    dict(width=128), dict(current_commit="changed"),
                    dict(provenance={"manifest_sha256":"wrong"})):
        with pytest.raises(ValueError, match="missing or mismatched"):
            a2.validate_48_parity_gate(path, **{**kw, **changed})
    d["comparisons"][0]["parity"]["steps"][1]["active_attention_gradients"] = False
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="missing or mismatched"):
        a2.validate_48_parity_gate(path, **kw)


def test_tiled_streamed_geometry_is_moved_one_at_a_time(monkeypatch):
    module = ModuleType("scripts.m48_ms_sr_train")
    moved = []
    module._geometry_device = lambda geom, device: (moved.append(geom), geom)[1]
    module._forward_args = lambda geom: {}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    class LinearModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = torch.nn.Parameter(torch.ones(()))
        def forward(self, state, *, baseline, coarse_state, time):
            return state * self.alpha
    model = LinearModel()
    geometry = [SimpleNamespace(fine_global=torch.arange(i, i+4),
                coarse_global=torch.tensor([0]), owned_fine_local=torch.arange(4))
                for i in (0, 4)]
    inp = {"state": torch.randn(8,5), "baseline": torch.zeros(8,5),
           "coarse_state": torch.zeros(1,5), "time": torch.tensor([.5]),
           "target": torch.zeros(8,5)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0)
    loss, _ = a2.gpu_step(model, optimizer, inp, geometry, "tiled",
                          stream_geometry=True)
    assert torch.isfinite(loss) and moved == geometry


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
        torch.save(_active_receipt(), ver)
        return {**_receipt(strategy, duration=8.0 if strategy == "tiled" else 1.0),
                "trial_id": id_}

    monkeypatch.setattr(a2, "_run_worker", fake_worker)
    args = SimpleNamespace(mode="static", fine_sides="48", depths="2", core_sides="24",
                           width=64, heads=4, warmup=2, measure=2, seed=484,
                           max_tiles=8, tiled_geometry_residency="auto",
                           parity_gate_summary=None, max_gpu_fraction=0.85,
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
    a2.activate_attention_paths(model)
    other = deepcopy(model)
    global_input = a2.make_global_inputs(n, 484, torch.device("cpu"))
    opt_full = torch.optim.AdamW(model.parameters(), lr=0.0)
    opt_tile = torch.optim.AdamW(other.parameters(), lr=0.0)
    _, a = a2.gpu_step(model, opt_full, global_input, [full], "full", capture=True)
    _, b = a2.gpu_step(other, opt_tile, global_input, tiles, "tiled", capture=True)
    assert float(a["prediction"].square().mean().sqrt()) > 1e-5
    assert all(v > 1e-12 for v in a["gradient_group_norms"].values())
    torch.testing.assert_close(a["prediction"], b["prediction"], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(a["gradient"], b["gradient"], rtol=1e-2, atol=1e-4)


def test_active_parity_initialization_unfreezes_gates_and_decoder_head():
    class SmallStage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            block = torch.nn.Module()
            block.adaLN_modulation = torch.nn.Sequential(
                torch.nn.SiLU(), torch.nn.Linear(4, 24))
            self.blocks = torch.nn.ModuleList([block])
            self.final_layer = torch.nn.Module()
            self.final_layer.projection = torch.nn.Linear(4, 5)
    model = torch.nn.Module()
    model.mode = "dynamic"
    model.fine_encoder = SmallStage()
    model.coarse_encoder = SmallStage()
    model.fine_decoder = SmallStage()
    model.coarse_dynamic = SmallStage()
    torch.manual_seed(484)
    a2.activate_attention_paths(model)
    for stage in (model.fine_encoder, model.coarse_encoder,
                  model.fine_decoder, model.coarse_dynamic):
        gate = stage.blocks[0].adaLN_modulation[-1]
        assert gate.weight.abs().sum().item() > 0
        assert gate.bias.abs().sum().item() > 0
    assert model.fine_decoder.final_layer.projection.weight.abs().sum().item() > 0
    assert model.fine_decoder.final_layer.projection.bias.abs().sum().item() == 0


def test_80_requires_two_step_active_parity_and_warmup_before_running(monkeypatch, tmp_path):
    monkeypatch.setattr(a2, "authentic_teacher_grids", lambda *_: (
        {80: {"fine_side":80,"periodic_length":1.0}}, {"teacher_snapshots":44}))
    monkeypatch.setattr(a2.subprocess,"check_output", lambda *a,**k: "test-commit")
    args = SimpleNamespace(mode="static", fine_sides="80", depths="2", core_sides="20",
                           width=64, heads=4, warmup=2, measure=2, seed=484,
                           max_tiles=64, tiled_geometry_residency="auto",
                           parity_gate_summary=None, max_gpu_fraction=.85,
                           max_host_graph_gib=12, worker_timeout_seconds=300,
                           output_dir=tmp_path/"a2n80", ms_sr_root=tmp_path,
                           normalization_file=tmp_path/"frozen.json")
    with pytest.raises(ValueError, match="requires --parity-gate-summary"):
        a2.run(args)
    assert not args.output_dir.exists()
    args.fine_sides="48"
    args.core_sides="24"
    args.warmup=1
    with pytest.raises(ValueError, match="invalid"):
        a2.run(args)
