from __future__ import annotations

import torch

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks import DiffusionDenoisingTask, EDMDenoisingTask
from graph_attention.training import train_equal_sample_optimizer_step
from scripts.train_generative import _DDPMAdapter, _EDMAdapter, _forward_model


class _ScalarModel(torch.nn.Module):
    def __init__(self, value: float = 0.25) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        batch_index: torch.Tensor,
        conditioning: torch.Tensor,
        attention_edge_indices: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del edge_index, coords, batch_index, conditioning, attention_edge_indices
        return self.weight * inputs


def _prepared_batch(task: DiffusionDenoisingTask | EDMDenoisingTask):
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=19)
    return task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)


def test_ddpm_adapter_matches_existing_optimizer_helper_exactly() -> None:
    task = DiffusionDenoisingTask(
        state_fields=("rho", "momentum"),
        timesteps=10,
        validation_seed=91,
    )
    batch = _prepared_batch(task)
    problem = task.make_training_problem(
        batch,
        generator=torch.Generator().manual_seed(123),
    )

    adapter_model = _ScalarModel()
    reference_model = _ScalarModel()
    adapter_optimizer = torch.optim.SGD(adapter_model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.01)

    step_loss, step_loss_sum, step_samples = _DDPMAdapter(task).optimizer_step(
        adapter_model,
        adapter_optimizer,
        problem,
    )
    reference = train_equal_sample_optimizer_step(
        reference_model,
        reference_optimizer,
        [problem],
        local_sample_count=problem.num_graphs,
    )

    assert step_samples == reference.local_sample_count
    assert step_loss == float(reference.objective.cpu())
    assert step_loss_sum == step_loss * step_samples
    torch.testing.assert_close(adapter_model.weight, reference_model.weight, rtol=0.0, atol=0.0)


def test_edm_adapter_matches_previous_manual_optimizer_sequence() -> None:
    task = EDMDenoisingTask(
        state_fields=("rho", "momentum"),
        sigma_data=1.0,
        validation_seed=91,
    )
    batch = _prepared_batch(task)
    problem = task.make_training_problem(
        batch,
        generator=torch.Generator().manual_seed(123),
    )

    adapter_model = _ScalarModel()
    reference_model = _ScalarModel()
    adapter_optimizer = torch.optim.SGD(adapter_model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.01)

    step_loss, step_loss_sum, step_samples = _EDMAdapter(task).optimizer_step(
        adapter_model,
        adapter_optimizer,
        problem,
    )

    reference_optimizer.zero_grad(set_to_none=True)
    predictions = _forward_model(reference_model, problem.model_batch)
    losses = task.edm_loss(predictions, problem)
    losses.mean.backward()
    reference_optimizer.step()

    assert step_samples == losses.sample_count
    assert step_loss == float(losses.mean.detach().cpu())
    assert step_loss_sum == float(losses.loss_sum.detach().cpu())
    torch.testing.assert_close(adapter_model.weight, reference_model.weight, rtol=0.0, atol=0.0)
