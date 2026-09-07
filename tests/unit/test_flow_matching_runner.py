from pathlib import Path

import torch
from scripts.train_slice_flow_matching import _marginal_generation_metrics
from torch.utils.tensorboard import SummaryWriter


def test_marginal_generation_metrics_reports_exact_empirical_wasserstein() -> None:
    generated = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    target = torch.tensor([[1.0, 1.0], [3.0, 5.0]])

    metrics = _marginal_generation_metrics(generated, target, ("a", "b"))

    assert metrics["a"]["marginal_wasserstein_1"] == 1.0
    assert metrics["b"]["marginal_wasserstein_1"] == 1.0
    assert metrics["a"]["generated_mean"] == 1.0
    assert metrics["a"]["target_mean"] == 2.0


def test_tensorboard_writer_persists_scalar_event(tmp_path: Path) -> None:
    with SummaryWriter(log_dir=str(tmp_path)) as writer:
        writer.add_scalar("flow_velocity_mse/train_epoch", 1.25, 0)

    assert any(path.name.startswith("events.out.tfevents") for path in tmp_path.iterdir())
