import pytest
import torch

from graph_attention.evaluation.diffusion_diagnostics import (
    epsilon_to_x0_mse_factor,
    graph_channel_moments,
    graph_channel_mse,
    reconstruct_x0_from_epsilon,
)


def test_oracle_epsilon_reconstructs_clean_state() -> None:
    clean = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    noise = torch.tensor([[0.2, 0.4], [-0.3, 0.1]], dtype=torch.float64)
    alpha = 0.25
    noisy = alpha**0.5 * clean + (1.0 - alpha) ** 0.5 * noise

    reconstructed = reconstruct_x0_from_epsilon(noisy, noise, alpha)

    torch.testing.assert_close(reconstructed, clean, rtol=0.0, atol=1.0e-12)


def test_x0_mse_matches_epsilon_error_amplification_identity() -> None:
    clean = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    noise = torch.tensor([[0.2], [0.4], [-0.3], [0.1]], dtype=torch.float64)
    epsilon_error = torch.tensor([[0.1], [-0.2], [0.3], [-0.4]], dtype=torch.float64)
    predicted_noise = noise + epsilon_error
    alpha = 0.2
    noisy = alpha**0.5 * clean + (1.0 - alpha) ** 0.5 * noise
    ptr = torch.tensor([0, 1, 4], dtype=torch.long)

    reconstructed = reconstruct_x0_from_epsilon(noisy, predicted_noise, alpha)
    epsilon_mse = graph_channel_mse(predicted_noise, noise, ptr).mean(dim=0)
    x0_mse = graph_channel_mse(reconstructed, clean, ptr).mean(dim=0)

    expected = epsilon_to_x0_mse_factor(alpha) * epsilon_mse
    torch.testing.assert_close(x0_mse, expected, rtol=1.0e-12, atol=1.0e-12)
    assert epsilon_to_x0_mse_factor(alpha) == pytest.approx(4.0)


def test_graph_channel_statistics_keep_graphs_separate() -> None:
    values = torch.tensor([[1.0], [0.0], [2.0], [4.0]], dtype=torch.float64)
    ptr = torch.tensor([0, 1, 4], dtype=torch.long)

    means, stds, maxima = graph_channel_moments(values, ptr)

    assert means.shape == (2, 1)
    assert means[0, 0] == pytest.approx(1.0)
    assert means[1, 0] == pytest.approx(2.0)
    assert stds[0, 0] == pytest.approx(0.0)
    assert stds[1, 0] == pytest.approx((8.0 / 3.0) ** 0.5)
    assert maxima[:, 0].tolist() == pytest.approx([1.0, 4.0])
