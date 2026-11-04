"""M45.3 tests for fixed-coarse sampling and the exact Gaussian FM oracle."""

import torch
import pytest

from graph_attention.data.two_scale_synthetic import (
    SyntheticQB, fit_synthetic_scales, make_synthetic_pair,
    sample_conditional_residual,
)
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from scripts.m45_checkpoint_diagnostics import (
    BlockGaussianOracle, _as_blocks, _from_blocks, _analytic_mode_second_moment,
    _ensemble_metrics, _correlations, _aggregate_time,
)


def test_fixed_coarse_residual_draws_preserve_restriction_and_are_reproducible():
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    for case in ("local", "shared_mode"):
        for op in ("block_average", "anchor_trilinear"):
            qb = SyntheticQB(mesh, op)
            coarse = make_synthetic_pair(qb, case="local", sample_id=1, seed=123).coarse
            a, template = sample_conditional_residual(
                qb, coarse, case=case,
                generator=torch.Generator(device="cpu").manual_seed(3456),
            )
            b, _ = sample_conditional_residual(
                qb, coarse, case=case,
                generator=torch.Generator(device="cpu").manual_seed(3456),
            )
            c, _ = sample_conditional_residual(
                qb, coarse, case=case,
                generator=torch.Generator(device="cpu").manual_seed(3457),
            )
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            assert not torch.equal(a, c)
            torch.testing.assert_close(qb.restrict(a), torch.zeros_like(coarse), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(qb.restrict(template), torch.zeros_like(coarse[:, :1]), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("case", ["local", "shared_mode"])
def test_exact_gaussian_velocity_endpoints_and_q_nullspace(case):
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    pairs = [make_synthetic_pair(qb, case=case, sample_id=i, seed=10+i) for i in range(4)]
    scales = fit_synthetic_scales(pairs)
    oracle = BlockGaussianOracle(qb, pairs[0].coarse, scales, case, torch.device("cpu"))
    source = qb.nullspace(torch.randn((mesh.fine_side**3, 5), generator=torch.Generator().manual_seed(999)))
    for t in (0.0, 0.1, 0.5, 0.9, 1.0):
        v = oracle.velocity(source, t)
        assert torch.isfinite(v).all()
        torch.testing.assert_close(qb.restrict(v), torch.zeros_like(pairs[0].coarse), atol=1e-5, rtol=1e-5)
        if t == 0.0:
            torch.testing.assert_close(v, -source, atol=1e-5, rtol=1e-5)
        if t == 1.0:
            torch.testing.assert_close(v, source, atol=1e-5, rtol=1e-5)


def test_exact_oracle_rejects_unsupported_operator():
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    qb = SyntheticQB(mesh, "anchor_trilinear")
    pair = make_synthetic_pair(qb, case="local", sample_id=1, seed=5)
    scales = fit_synthetic_scales([pair, make_synthetic_pair(qb, case="local", sample_id=2, seed=6)])
    with pytest.raises(ValueError, match="block_average"):
        BlockGaussianOracle(qb, pair.coarse, scales, "local", torch.device("cpu"))


def test_block_tensor_order_roundtrip():
    x = torch.arange(6**3 * 5, dtype=torch.float32).reshape(6**3, 5)
    torch.testing.assert_close(_from_blocks(_as_blocks(x, 3, 2), 3, 2), x, atol=0, rtol=0)


def test_analytic_shared_mode_variance_and_conditional_ensemble():
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    pair = make_synthetic_pair(qb, case="shared_mode", sample_id=0, seed=145)
    expected_shared = _analytic_mode_second_moment(qb, pair.coarse, "shared_mode")
    expected_local = _analytic_mode_second_moment(qb, pair.coarse, "local")
    torch.testing.assert_close(torch.tensor(expected_shared - expected_local),
                               torch.tensor((1.7 * 0.023)**2), rtol=1e-6, atol=1e-8)
    generator_seed = 7000
    draws = torch.stack([sample_conditional_residual(qb, pair.coarse, case="shared_mode",
                generator=torch.Generator(device="cpu").manual_seed(generator_seed + k))[0]
                for k in range(128)])
    metrics = _ensemble_metrics(draws, pair.mode[:, 0], qb)
    assert abs(metrics["mode_second_moment"] - expected_shared) / expected_shared < 0.35
    assert metrics["max_Q_residual"] < 1e-6
    assert len(_correlations(draws, qb.nf)) == 3 * (qb.nf // 2 + 1)


def test_time_aggregation_groups_by_validation_coarse_field():
    records = []
    for sample, base in ((101, 1.0), (202, 3.0)):
        for draw in range(2):
            rec = {"case": "local", "mode": "static", "sample_id": sample,
                   "t": 0.5, "noise_draw": draw}
            for metric in ("fm_mse_raw", "fm_mse_projected", "oracle_fm_mse",
                           "excess_mse_vs_oracle", "mode_velocity_mse", "oracle_mode_velocity_mse"):
                rec[metric] = base
            records.append(rec)
    rows = _aggregate_time(records)
    assert len(rows) == 1
    assert rows[0]["validation_coarse_fields"] == 2
    assert rows[0]["fm_mse_raw_mean"] == 2.0
    assert rows[0]["fm_mse_raw_sem"] == 1.0
