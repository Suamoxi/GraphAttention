"""M45.6 reproducibility, canonical Gaussian, and depth-aggregation tests."""
from __future__ import annotations

import csv
import json

import pytest
import torch

from graph_attention.data.two_scale_synthetic import (
    SyntheticQB, fit_synthetic_scales, make_synthetic_pair,
)
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from scripts.m45_checkpoint_diagnostics import BlockGaussianOracle, _generate_oracle
from scripts.m45_6_oracle_transport import (
    CanonicalGaussianTransport, _source, _analytic_mode_second_moment,
)
from scripts.m45_6_compare_depth import aggregate
from scripts import m45_6_reproducibility as repro


@pytest.mark.parametrize("case", ["local", "shared_mode"])
def test_canonical_gaussian_transport_matches_analytic_mode_moment(case):
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    train = [make_synthetic_pair(qb, case=case, sample_id=i, seed=104+i)
             for i in range(5)]
    scales = fit_synthetic_scales(train)
    oracle = BlockGaussianOracle(qb, train[0].coarse, scales, case, torch.device("cpu"))
    direct = CanonicalGaussianTransport(oracle)
    assert direct.sqrt_local.shape == (qb.nc**3, 5)
    template = train[0].mode[:, 0]
    draws = []
    for seed in range(256):
        x = direct.apply(_source(qb, seed)) * scales.rms_residual
        assert float(qb.restrict(x).abs().max()) < 5e-7
        draws.append(float(x[:, 1] @ template / template.square().sum()))
    measured = sum(v*v for v in draws) / len(draws)
    expected = _analytic_mode_second_moment(qb, train[0].coarse, case)
    assert abs(measured - expected) / expected < 0.30


def test_direct_local_transport_matches_exact_heun_endpoint():
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    train = [make_synthetic_pair(qb, case="local", sample_id=i, seed=130+i)
             for i in range(5)]
    scales = fit_synthetic_scales(train)
    oracle = BlockGaussianOracle(qb, train[0].coarse, scales, "local", torch.device("cpu"))
    direct = CanonicalGaussianTransport(oracle)
    source = _source(qb, 231)
    expected = (direct.apply(source).cpu() * scales.rms_residual).float()
    actual = _generate_oracle(oracle, qb, scales, device=torch.device("cpu"),
                              seed=231, steps=128)
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-6)


def _fake_job(root, job, seed, depth, fm, m2):
    folder = root / job
    folder.mkdir()
    settings = {"training_scope": "full_all", "seed": seed, "train_samples": 24,
                "val_seed": 7002026, "fine_pre_layers": depth,
                "fine_post_layers": depth, "coarse_static_layers": depth,
                "coarse_dynamic_layers": depth, "width": 16, "steps": 2000,
                "refinement": 2}
    (folder / "summary.json").write_text(json.dumps({"settings": settings, "results": [
        {"case": "shared_mode", "mode": "static", "heldout_fm_loss": fm,
         "parameters": 1000},
    ]}))
    (folder / "diagnostics").mkdir()
    with (folder / "diagnostics" / "generation_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "mode", "variant", "heun_steps",
                                              "mode_second_moment", "analytic_mode_second_moment"])
        writer.writeheader()
        writer.writerow({"case": "shared_mode", "mode": "static", "variant": "normal",
                         "heun_steps": 25, "mode_second_moment": m2,
                         "analytic_mode_second_moment": 0.0015})


def test_depth_comparator_matches_equal_seed_and_rejects_duplicate(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    _fake_job(root, "job_1", 2026, 2, 0.85, 1e-5)
    _fake_job(root, "job_2", 2026, 4, 0.70, 2e-5)
    report = aggregate(root, tmp_path / "report")
    assert report["n_matched"] == 1
    with (tmp_path / "report" / "matched_depth.csv").open() as f:
        row = list(csv.DictReader(f))[0]
    assert float(row["fm_improvement_pct"]) == pytest.approx(100*(0.85-0.70)/0.85)
    _fake_job(root, "job_3", 2026, 2, 0.8, 1e-5)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(root, tmp_path / "report_2")


class _PointwiseVelocity(torch.nn.Module):
    mode = "static"
    fine_pre_layers = 1
    fine_post_layers = 1
    coarse_static_layers = 1
    coarse_dynamic_layers = 0

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, state, *, baseline, **kwargs):
        return self.linear(torch.cat((state, baseline), -1))


@pytest.mark.parametrize("task", ["trace", "parity"])
def test_repro_local_cpu_and_matched_paths(tmp_path, monkeypatch, task):
    monkeypatch.setattr(repro, "_model", lambda *args: _PointwiseVelocity())
    params = dict(task=task, cases=("local",), modes=("static",), depth=2,
                  width=16, steps=3, train_samples=4, seed=7, coarse_side=3,
                  refinement=2, core_side=3, device=torch.device("cpu"),
                  deterministic="off")
    torch.manual_seed(442)
    first = repro.execute(output_dir=tmp_path / "first", **params)
    if task == "trace":
        torch.manual_seed(442)
        repro.execute(output_dir=tmp_path / "second", **params)
        comparison = repro.compare(tmp_path / "first", tmp_path / "second",
                                   tmp_path / "compare")
        assert comparison["comparisons"][0]["bitwise_identical"]
    else:
        final = first["runs"][0]["final"]
        assert final["max_owned_prediction_diff"] < 1e-6
        assert final["max_grad_diff"] < 1e-6
        assert final["max_weight_diff"] < 1e-6
