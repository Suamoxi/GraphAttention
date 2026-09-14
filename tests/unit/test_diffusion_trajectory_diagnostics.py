import math

import pytest
import torch

from scripts.diagnose_diffusion_trajectory import (
    _expected_q_moments,
    _pooled_channel_stats,
    _reverse_schedule_scalars,
    _transition_rows,
)


def test_expected_q_moments_match_forward_marginal_formula() -> None:
    reference_mean = torch.tensor([2.0, -1.0], dtype=torch.float64)
    reference_std = torch.tensor([3.0, 0.5], dtype=torch.float64)
    alpha = 0.25

    mean, std = _expected_q_moments(reference_mean, reference_std, alpha)

    torch.testing.assert_close(
        mean,
        torch.tensor([1.0, -0.5], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    expected_variance = alpha * reference_std.square() + (1.0 - alpha)
    torch.testing.assert_close(std.square(), expected_variance, rtol=0.0, atol=1.0e-12)


def test_reverse_schedule_scalars_match_generalized_ddim_formula() -> None:
    alpha_t = 0.4
    alpha_previous = 0.5
    eta = 1.0

    sigma, direction = _reverse_schedule_scalars(
        alpha_t,
        alpha_previous,
        eta=eta,
        previous_is_clean=False,
    )

    variance_factor = (
        (1.0 - alpha_previous)
        / (1.0 - alpha_t)
        * (1.0 - alpha_t / alpha_previous)
    )
    assert sigma == pytest.approx(math.sqrt(variance_factor))
    assert direction == pytest.approx(
        math.sqrt(max(1.0 - alpha_previous - variance_factor, 0.0))
    )

    final_sigma, final_direction = _reverse_schedule_scalars(
        0.9,
        1.0,
        eta=1.0,
        previous_is_clean=True,
    )
    assert final_sigma == 0.0
    assert final_direction == 0.0


def test_transition_rows_measure_actual_state_update() -> None:
    before = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
    after = torch.tensor([[1.0, 1.0], [4.0, 2.0]], dtype=torch.float64)

    rows = _transition_rows(
        before,
        after,
        from_timestep=10,
        to_timestep=9,
        total_timesteps=10,
        channel_names=("a", "b"),
    )

    by_channel = {row["channel"]: row for row in rows}
    assert by_channel["a"]["update_rms"] == pytest.approx(math.sqrt((1.0 + 4.0) / 2.0))
    assert by_channel["b"]["update_rms"] == pytest.approx(math.sqrt(0.5))
    assert by_channel["__mean__"]["from_timestep"] == 10
    assert by_channel["__mean__"]["to_timestep"] == 9


def test_pooled_channel_stats_use_population_standard_deviation() -> None:
    values = torch.tensor([[1.0, -1.0], [3.0, 1.0]], dtype=torch.float64)

    mean, std, rms, maximum = _pooled_channel_stats(values)

    torch.testing.assert_close(mean, torch.tensor([2.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(std, torch.tensor([1.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(rms, torch.tensor([math.sqrt(5.0), 1.0], dtype=torch.float64))
    torch.testing.assert_close(maximum, torch.tensor([3.0, 1.0], dtype=torch.float64))
