from __future__ import annotations

import torch

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks import FlowMatchingTask
from scripts.train_generative import _GenericTaskAdapter, _forward_model, _task_adapter


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


def test_flow_matching_uses_common_generative_adapter_without_numerical_rewrite() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=19)
    task = FlowMatchingTask(
        state_fields=("rho", "momentum"),
        validation_seed=91,
    )
    batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    problem = task.make_training_problem(
        batch,
        generator=torch.Generator().manual_seed(123),
    )

    adapter = _task_adapter(task)
    assert isinstance(adapter, _GenericTaskAdapter)
    assert adapter.metric_name == "flow_velocity_mse"
    assert adapter.noise_seed_name == "path_seed"
    assert task.make_model_probe(problem) is problem

    adapter_model = _ScalarModel()
    reference_model = _ScalarModel()
    adapter_optimizer = torch.optim.SGD(adapter_model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.01)

    step_loss, step_loss_sum, step_samples = adapter.optimizer_step(
        adapter_model,
        adapter_optimizer,
        problem,
    )

    reference_optimizer.zero_grad(set_to_none=True)
    predictions = _forward_model(reference_model, problem)
    losses = task.training_loss(predictions, problem)
    losses.mean.backward()
    reference_optimizer.step()

    assert step_samples == losses.sample_count
    assert step_loss == float(losses.mean.detach().cpu())
    assert step_loss_sum == float(losses.loss_sum.detach().cpu())
    torch.testing.assert_close(adapter_model.weight, reference_model.weight, rtol=0.0, atol=0.0)

    metadata = task.training_summary_metadata()
    assert metadata["task"] == "linear_gaussian_flow_matching"
    assert metadata["prediction_type"] == "velocity"
    assert metadata["path"] == "x_t=(1-t)*x_source+t*x_data"
