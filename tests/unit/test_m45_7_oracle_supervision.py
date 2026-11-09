"""M45.7.1 tests: matched paths and exact Gaussian supervision."""

import pytest
import torch

from graph_attention.data.two_scale_synthetic import (
    SyntheticQB, fit_synthetic_scales, make_synthetic_pair,
    sample_conditional_residual,
)
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from scripts.m45_7_oracle_supervision import (
    _make_model, _supervision_target, _training_residual,
)
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



def test_m457_oracle_targets_match_known_endpoint_velocities():
    qb, pairs, scales = _fixture()
    pair = pairs[0]
    _, _, x1 = _state(pair, scales)
    oracle = BlockGaussianOracle(
        qb, pair.coarse, scales, "shared_mode", torch.device("cpu"),
    )
    noise = qb.nullspace(torch.randn_like(x1))
    zeros = torch.zeros_like(x1)
    torch.testing.assert_close(
        _supervision_target("oracle", noise, zeros, torch.tensor([0.0]), oracle),
        -noise, atol=1e-5, rtol=1e-5,
    )
    torch.testing.assert_close(
        _supervision_target("oracle", x1, zeros, torch.tensor([1.0]), oracle),
        x1, atol=1e-5, rtol=1e-5,
    )


@pytest.mark.parametrize("case", ["local", "shared_mode"])
def test_m457_fixed_residual_is_identical_to_original_training_pair(case):
    qb, pairs, scales = _fixture(case)
    pair = pairs[0]
    actual = _training_residual(
        pair, qb, scales, policy="fixed", case=case, seed=2028, step=7,
    )
    expected = _state(pair, scales)[2]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("case", ["local", "shared_mode"])
def test_m457_resampled_residual_is_paired_nullspace_and_reproducible(case):
    qb, pairs, scales = _fixture(case)
    pair = pairs[1]
    kw = dict(policy="resample", case=case, seed=2028)
    a = _training_residual(pair, qb, scales, step=7, **kw)
    b = _training_residual(pair, qb, scales, step=7, **kw)
    another = _training_residual(pair, qb, scales, step=8, **kw)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not torch.equal(a, another)
    assert not torch.equal(a, _state(pair, scales)[2])

    physical = a * scales.rms_residual
    torch.testing.assert_close(
        qb.restrict(physical), torch.zeros_like(pair.coarse),
        atol=1e-6, rtol=1e-6,
    )
    torch.testing.assert_close(
        qb.restrict(pair.baseline + physical), pair.coarse,
        atol=1e-6, rtol=1e-6,
    )
    # The resampler must use precisely the existing conditional generator.
    rng = torch.Generator(device="cpu").manual_seed(
        2028 + 81_000_000 + (7 + 1) * 1_000_003,
    )
    expected, _ = sample_conditional_residual(
        qb, pair.coarse, case=case, generator=rng,
    )
    torch.testing.assert_close(
        a, expected / scales.rms_residual, atol=0, rtol=0,
    )


def test_m457_resampling_is_independent_of_fm_time_and_noise_stream():
    qb, pairs, scales = _fixture()
    pair = pairs[1]
    seed, step = 2028, 12
    residual = _training_residual(
        pair, qb, scales, policy="resample",
        case="shared_mode", seed=seed, step=step,
    )
    path_seed = seed + 500_000 + step * 997
    a = _path(residual, path_seed, qb)
    # Recreate the residual using the same isolated RNG, then the FM path.
    residual_again = _training_residual(
        pair, qb, scales, policy="resample",
        case="shared_mode", seed=seed, step=step,
    )
    b = _path(residual_again, path_seed, qb)
    for first, second in zip(a, b):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_m457_invalid_residual_policy_fails():
    qb, pairs, scales = _fixture()
    with pytest.raises(ValueError, match="unknown residual policy"):
        _training_residual(
            pairs[0], qb, scales, policy="other",
            case="shared_mode", seed=2028, step=0,
        )
