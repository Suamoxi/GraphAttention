from __future__ import annotations

import pytest
import torch

from graph_attention.data import SyntheticMeshDataset
from graph_attention.objectives import sample_reduced_mse
from graph_attention.tasks import (
    DiffusionDenoisingTask,
    EDMDenoisingTask,
    NodeRegressionTask,
    VPSDEDenoisingTask,
)
from graph_attention.training import train_equal_sample_optimizer_step
from scripts.train_generative import (
    _DDPMAdapter,
    _accumulated_optimizer_step,
    _EDMAdapter,
    _GenericTaskAdapter,
    _forward_model,
    _task_adapter,
)


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


def _prepared_batch(
    task: DiffusionDenoisingTask | EDMDenoisingTask | VPSDEDenoisingTask,
):
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


def test_vp_sde_uses_generic_task_adapter_without_runner_changes() -> None:
    task = VPSDEDenoisingTask(
        state_fields=("rho", "momentum"),
        beta_min=0.1,
        beta_max=20.0,
        validation_seed=91,
    )
    batch = _prepared_batch(task)
    problem = task.make_training_problem(
        batch,
        generator=torch.Generator().manual_seed(123),
    )

    adapter = _task_adapter(task)
    assert isinstance(adapter, _GenericTaskAdapter)
    assert adapter.metric_name == "epsilon_mse"
    assert adapter.noise_seed_name == "vp_sde_noise_seed"

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



class _IdentityTrainingTask(NodeRegressionTask):
    def make_training_problem(
        self,
        batch,
        *,
        generator: torch.Generator | None = None,
    ):
        del generator
        return batch


class _IdentityStandardizers:
    def transform(self, batch):
        return batch


class _MSEAdapter:
    def loss(self, model: torch.nn.Module, problem):
        predictions = _forward_model(model, problem)
        return sample_reduced_mse(
            predictions,
            problem.targets,
            problem.ptr,
            node_weights=problem.node_weights,
        )


def test_gradient_accumulation_matches_one_equal_sample_large_batch_step() -> None:
    dataset = SyntheticMeshDataset(num_samples=4, spatial_dim=2, seed=37)
    task = _IdentityTrainingTask(input_fields=("rho",), target_fields=("rho",))

    full_batch = task.pack_and_prepare(
        [dataset[index] for index in range(4)],
        dataset.field_catalog,
    )
    microbatches = [
        task.pack_and_prepare([dataset[0]], dataset.field_catalog),
        task.pack_and_prepare([dataset[1], dataset[2]], dataset.field_catalog),
        task.pack_and_prepare([dataset[3]], dataset.field_catalog),
    ]

    reference = _ScalarModel()
    accumulated = _ScalarModel()
    accumulated.load_state_dict(reference.state_dict())
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)

    reference_optimizer.zero_grad(set_to_none=True)
    reference_loss = _MSEAdapter().loss(reference, full_batch)
    reference_loss.mean.backward()
    reference_optimizer.step()

    step_loss, loss_sum, sample_count = _accumulated_optimizer_step(
        _MSEAdapter(),
        accumulated,
        accumulated_optimizer,
        microbatches,
        task=task,
        standardizers=_IdentityStandardizers(),
        device=torch.device("cpu"),
        training_generator=torch.Generator().manual_seed(1),
    )

    torch.testing.assert_close(accumulated.weight, reference.weight)
    assert sample_count == 4
    assert step_loss == pytest.approx(float(reference_loss.mean.detach()))
    assert loss_sum == pytest.approx(float(reference_loss.loss_sum.detach()))
