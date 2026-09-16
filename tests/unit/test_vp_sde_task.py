from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks import VPSDEDenoisingTask


class _ZeroEpsilon(torch.nn.Module):
    def forward(
        self,
        inputs: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        batch_index: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        del edge_index, coords, batch_index, conditioning
        return torch.zeros_like(inputs)


def _batch():
    dataset = SyntheticMeshDataset(num_samples=3, spatial_dim=2, seed=7)
    task = VPSDEDenoisingTask(
        state_fields=("rho", "momentum"),
        beta_min=0.1,
        beta_max=20.0,
        training_eps=1.0e-5,
        validation_seed=91,
    )
    batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    return task, dataset, batch


def test_default_vp_schedule_matches_m21_continuous_beta_endpoints() -> None:
    task, _, _ = _batch()
    times = torch.tensor([0.0, 1.0], dtype=torch.float64)
    continuous_beta = task.beta(times)

    torch.testing.assert_close(continuous_beta, torch.tensor([0.1, 20.0], dtype=torch.float64))
    torch.testing.assert_close(
        continuous_beta / 1000.0,
        torch.tensor([1.0e-4, 2.0e-2], dtype=torch.float64),
    )


def test_vp_marginal_coefficients_are_variance_preserving() -> None:
    task, _, _ = _batch()
    times = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64)
    alpha, sigma = task.marginal_coefficients(times)

    torch.testing.assert_close(alpha.square() + sigma.square(), torch.ones_like(times))
    assert alpha[0].item() == pytest.approx(1.0)
    assert sigma[0].item() == pytest.approx(0.0)
    assert alpha[-1].item() == pytest.approx(torch.exp(torch.tensor(-5.025)).item())


def test_vp_training_problem_matches_analytic_marginal() -> None:
    task, _, batch = _batch()
    generator = torch.Generator().manual_seed(123)
    problem = task.make_training_problem(batch, generator=generator)

    times = problem.conditioning[:, -1]
    alpha, sigma = task.marginal_coefficients(times)
    node_alpha = alpha[batch.batch_index].unsqueeze(1)
    node_sigma = sigma[batch.batch_index].unsqueeze(1)
    expected = node_alpha * batch.inputs + node_sigma * problem.targets

    torch.testing.assert_close(problem.inputs, expected)
    assert problem.conditioning_names[-1] == "vp_sde_time"
    assert torch.all(times >= task.training_eps)
    assert torch.all(times <= 1.0)
    assert problem.target_channels == tuple(f"epsilon:{name}" for name in batch.input_channels)


def test_vp_validation_is_sample_id_deterministic_across_batch_order() -> None:
    task, dataset, _ = _batch()
    forward = task.make_validation_problem(
        task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    )
    reverse = task.make_validation_problem(
        task.pack_and_prepare([dataset[1], dataset[0]], dataset.field_catalog)
    )

    forward_first = slice(int(forward.ptr[0]), int(forward.ptr[1]))
    reverse_second = slice(int(reverse.ptr[1]), int(reverse.ptr[2]))
    torch.testing.assert_close(forward.inputs[forward_first], reverse.inputs[reverse_second])
    torch.testing.assert_close(forward.targets[forward_first], reverse.targets[reverse_second])
    torch.testing.assert_close(forward.conditioning[0], reverse.conditioning[1])


def test_vp_training_loss_is_equal_sample_epsilon_mse() -> None:
    task, _, batch = _batch()
    problem = task.make_training_problem(batch, generator=torch.Generator().manual_seed(12))

    exact = task.training_loss(problem.targets.clone(), problem)
    assert exact.mean.item() == pytest.approx(0.0)
    assert exact.sample_count == problem.num_graphs

    zeros = task.training_loss(torch.zeros_like(problem.targets), problem)
    assert zeros.mean.item() > 0.0


def test_vp_score_conversion_matches_epsilon_parameterization() -> None:
    task, _, batch = _batch()
    times = torch.tensor([0.2, 0.7], dtype=batch.inputs.dtype)
    epsilon_hat = torch.ones_like(batch.inputs)
    score = task.score_from_epsilon(epsilon_hat, batch, times)

    _, sigma = task.marginal_coefficients(times)
    expected = -torch.ones_like(batch.inputs) / sigma[batch.batch_index].unsqueeze(1)
    torch.testing.assert_close(score, expected)


def test_probability_flow_and_reverse_drifts_reduce_to_forward_drift_for_zero_score() -> None:
    task, _, batch = _batch()
    model = _ZeroEpsilon()
    state = torch.randn_like(batch.inputs)
    times = torch.tensor([0.2, 0.7], dtype=batch.inputs.dtype)
    beta = task.beta(times)[batch.batch_index].unsqueeze(1)
    expected = -0.5 * beta * state

    ode_drift = task.probability_flow_drift(model, batch, state, times)
    sde_drift = task.reverse_sde_drift(model, batch, state, times)

    torch.testing.assert_close(ode_drift, expected)
    torch.testing.assert_close(sde_drift, expected)


def test_probability_flow_sampling_is_independent_of_clean_reference_values() -> None:
    task, _, batch = _batch()
    model = _ZeroEpsilon()

    generated = task.sample_standardized(
        model,
        batch,
        steps=4,
        method="probability_flow_ode",
        solver="heun",
        sampling_eps=1.0e-3,
        final_denoise=True,
        sampling_seed=44,
        sampling_keys=("gen_0", "gen_1"),
    )

    altered = replace(batch, inputs=batch.inputs + 100.0, targets=batch.targets + 100.0)
    altered_generated = task.sample_standardized(
        model,
        altered,
        steps=4,
        method="probability_flow_ode",
        solver="heun",
        sampling_eps=1.0e-3,
        final_denoise=True,
        sampling_seed=44,
        sampling_keys=("gen_0", "gen_1"),
    )

    torch.testing.assert_close(altered_generated, generated)


def test_reverse_sde_sampling_is_seed_deterministic() -> None:
    task, _, batch = _batch()
    model = _ZeroEpsilon()
    kwargs = {
        "steps": 4,
        "method": "reverse_sde",
        "solver": "euler_maruyama",
        "sampling_eps": 1.0e-3,
        "final_denoise": True,
        "sampling_keys": ("gen_0", "gen_1"),
    }

    first = task.sample_standardized(model, batch, sampling_seed=44, **kwargs)
    second = task.sample_standardized(model, batch, sampling_seed=44, **kwargs)
    different = task.sample_standardized(model, batch, sampling_seed=45, **kwargs)

    torch.testing.assert_close(first, second)
    assert not torch.equal(first, different)


def test_vp_sampler_rejects_incompatible_method_solver_pairs() -> None:
    task, _, batch = _batch()
    model = _ZeroEpsilon()

    with pytest.raises(ValueError, match="probability-flow ODE solver"):
        task.sample_standardized(
            model,
            batch,
            steps=2,
            method="probability_flow_ode",
            solver="euler_maruyama",
        )
    with pytest.raises(ValueError, match="reverse-SDE solver"):
        task.sample_standardized(
            model,
            batch,
            steps=2,
            method="reverse_sde",
            solver="heun",
        )
