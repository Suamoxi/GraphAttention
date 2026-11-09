"""M45.7.1 tests: matched paths and exact Gaussian supervision."""

import pytest
import torch

from graph_attention.data.two_scale_synthetic import (
    SyntheticQB, fit_synthetic_scales, make_synthetic_pair,
)
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from scripts.m45_7_oracle_supervision import _make_model, _supervision_target
from scripts.m45_checkpoint_diagnostics import BlockGaussianOracle
from scripts.m45_synthetic_pilot import _path, _state


def _fixture(case="shared_mode"):
    mesh = build_hit_two_scale_geometry(coarse_side=3, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    pairs = [
        make_synthetic_pair(qb, case=case, sample_id=i, seed=2026 + 3001 * i)
        for i in range(4)
    ]
    return qb, pairs, fit_synthetic_scales(pairs)


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_m457_shared_initialization_is_repeatable(mode):
    a = _make_model(mode=mode, seed=2026, width=16, heads=4, depth=2)
    b = _make_model(mode=mode, seed=2026, width=16, heads=4, depth=2)
    assert a.state_dict().keys() == b.state_dict().keys()
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)


def test_m457_static_dynamic_share_existing_parameters():
    static = _make_model(mode="static", seed=2026, width=16, heads=4, depth=2)
    dynamic = _make_model(mode="dynamic", seed=2026, width=16, heads=4, depth=2)
    for key, value in static.state_dict().items():
        if key in dynamic.state_dict():
            torch.testing.assert_close(value, dynamic.state_dict()[key], atol=0, rtol=0)


@pytest.mark.parametrize("case", ["local", "shared_mode"])
def test_m457_oracle_targets_are_nullspace_and_endpoint_consistent(case):
    qb, pairs, scales = _fixture(case)
    pair = pairs[0]
    _, _, x1 = _state(pair, scales)
    oracle = BlockGaussianOracle(qb, pair.coarse, scales, case, torch.device("cpu"))
    for step in (0, 1, 2):
        xt, fm_target, t = _path(x1, 2026 + step * 997, qb)
        target = _supervision_target("oracle", xt, fm_target, t, oracle)
        torch.testing.assert_close(
            qb.restrict(target), torch.zeros_like(pair.coarse),
            atol=1e-5, rtol=1e-5,
        )
        assert torch.isfinite(target).all()
        assert torch.equal(
            _supervision_target("fm", xt, fm_target, t, oracle), fm_target,
        )
    torch.testing.assert_close(
        oracle.velocity(x1, 1.0), x1, atol=1e-5, rtol=1e-5,
    )


def test_m457_supervision_rejects_unavailable_oracle():
    target = torch.zeros((2, 5))
    with pytest.raises(ValueError, match="requires"):
        _supervision_target("oracle", target, target, torch.tensor([0.5]), None)


def test_m457_gaussian_oracle_is_better_than_noisy_velocity_for_expected_target():
    # Conditional expectation minimizes the population MSE, not the paired
    # stochastic target error for every individual realization.
    qb, pairs, scales = _fixture()
    x1 = _state(pairs[0], scales)[2]
    oracle = BlockGaussianOracle(
        qb, pairs[0].coarse, scales, "shared_mode", torch.device("cpu"),
    )
    losses_oracle, losses_zero = [], []
    for seed in range(32):
        xt, fm_target, t = _path(x1, seed + 500, qb)
        best = oracle.velocity(xt, float(t.item()))
        losses_oracle.append(float((best - fm_target).square().mean()))
        losses_zero.append(float(fm_target.square().mean()))
    assert sum(losses_oracle) / 32 < sum(losses_zero) / 32
