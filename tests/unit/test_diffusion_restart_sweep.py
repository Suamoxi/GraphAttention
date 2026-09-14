import torch
from scripts.diagnose_diffusion_restart_sweep import (
    _equal_sample_stats,
    _forward_marginal_state,
    _keyed_noise,
    _reverse_ancestral_from_state,
)

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks import DiffusionDenoisingTask


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


def _task_and_batch(timesteps: int = 10):
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=7)
    task = DiffusionDenoisingTask(
        state_fields=("rho", "momentum"),
        timesteps=timesteps,
        validation_seed=91,
    )
    batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    return task, batch


def test_forward_marginal_state_matches_ddpm_equation() -> None:
    task, batch = _task_and_batch()
    noise = torch.randn(
        batch.inputs.shape,
        generator=torch.Generator().manual_seed(123),
    )
    timestep = 7

    state = _forward_marginal_state(
        batch.inputs,
        noise,
        task=task,
        timestep=timestep,
    )

    alpha = task._alpha_bar_for(batch.inputs)[timestep]
    expected = torch.sqrt(alpha) * batch.inputs + torch.sqrt(1.0 - alpha) * noise
    torch.testing.assert_close(state, expected)


def test_keyed_restart_noise_is_deterministic_by_purpose() -> None:
    _, batch = _task_and_batch()
    keys = tuple(batch.source.sample_ids)

    first = _keyed_noise(batch, seed=44, purpose="restart_t7", sample_keys=keys)
    second = _keyed_noise(batch, seed=44, purpose="restart_t7", sample_keys=keys)
    different = _keyed_noise(batch, seed=44, purpose="restart_t6", sample_keys=keys)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert not torch.equal(first, different)


def test_reverse_restart_at_timestep_one_matches_final_ddpm_update() -> None:
    task, batch = _task_and_batch()
    model = _ZeroEpsilon()
    initial = torch.randn(
        batch.inputs.shape,
        generator=torch.Generator().manual_seed(5),
    )

    final = _reverse_ancestral_from_state(
        model,
        task=task,
        batch=batch,
        initial_state=initial,
        start_timestep=1,
        reverse_seed=77,
        sample_keys=tuple(batch.source.sample_ids),
    )

    alpha = task._alpha_bar_for(initial)[1]
    torch.testing.assert_close(final, initial / torch.sqrt(alpha))


def test_equal_sample_stats_do_not_weight_larger_graph_more() -> None:
    values = torch.tensor([[1.0], [3.0], [10.0], [10.0], [10.0], [10.0]])
    ptr = torch.tensor([0, 2, 6], dtype=torch.long)

    stats = _equal_sample_stats(values, ptr)

    expected_mean = torch.tensor([(2.0 + 10.0) / 2.0])
    torch.testing.assert_close(stats["mean"], expected_mean)
