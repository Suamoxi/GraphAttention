import torch
from scripts.train_slice_flow_matching import _marginal_generation_metrics


def test_marginal_generation_metrics_reports_exact_empirical_wasserstein() -> None:
    generated = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    target = torch.tensor([[1.0, 1.0], [3.0, 5.0]])

    metrics = _marginal_generation_metrics(generated, target, ("a", "b"))

    assert metrics["a"]["marginal_wasserstein_1"] == 1.0
    assert metrics["b"]["marginal_wasserstein_1"] == 1.0
    assert metrics["a"]["generated_mean"] == 1.0
    assert metrics["a"]["target_mean"] == 2.0
