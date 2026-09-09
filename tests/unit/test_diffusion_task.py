from dataclasses import replace

import torch

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


def _batch(timesteps: int = 10):
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=7)
    task = DiffusionDenoisingTask(
        state_fields=("rho", "momentum"),
        timesteps=timesteps,
        validation_seed=91,
    )
    batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    return task, dataset, batch


def test_diffusion_training_problem_matches_forward_process_and_graph_time() -> None:
    task, _, batch = _batch()
    generator = torch.Generator().manual_seed(123)
    problem = task.make_training_problem(batch, generator=generator)

    timesteps = torch.round(problem.conditioning[:, -1] * task.timesteps).to(torch.long)
    alpha_bar = task._alpha_bar_for(batch.inputs)
    node_alpha = alpha_bar[timesteps[batch.batch_index]].unsqueeze(1)
    expected = (
        torch.sqrt(node_alpha) * batch.inputs
        + torch.sqrt(1.0 - node_alpha) * problem.targets
    )

    torch.testing.assert_close(problem.inputs, expected)
    assert problem.conditioning_names[-1] == "diffusion_time"
    assert torch.all(problem.conditioning[:, -1] > 0.0)
    assert torch.all(problem.conditioning[:, -1] <= 1.0)
    assert problem.target_channels == tuple(
        f"epsilon:{name}" for name in batch.input_channels
    )


def test_diffusion_validation_is_sample_id_deterministic_across_batch_order() -> None:
    task, dataset, _ = _batch()
    forward = task.make_validation_problem(
        task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    )
    reverse = task.make_validation_problem(
        task.pack_and_prepare([dataset[1], dataset[0]], dataset.field_catalog)
    )

    forward_first = slice(int(forward.ptr[0]), int(forward.ptr[1]))
    reverse_first = slice(int(reverse.ptr[1]), int(reverse.ptr[2]))
    torch.testing.assert_close(forward.inputs[forward_first], reverse.inputs[reverse_first])
    torch.testing.assert_close(forward.targets[forward_first], reverse.targets[reverse_first])
    torch.testing.assert_close(forward.conditioning[0], reverse.conditioning[1])


def test_diffusion_sampling_does_not_use_clean_reference_values() -> None:
    task, _, batch = _batch()
    model = _ZeroEpsilon()
    generated = task.sample_standardized(
        model,
        batch,
        steps=5,
        eta=0.0,
        sampling_seed=44,
        sampling_keys=("gen_000000", "gen_000001"),
    )

    altered = replace(batch, inputs=batch.inputs + 100.0, targets=batch.targets + 100.0)
    altered_generated = task.sample_standardized(
        model,
        altered,
        steps=5,
        eta=0.0,
        sampling_seed=44,
        sampling_keys=("gen_000000", "gen_000001"),
    )
    torch.testing.assert_close(altered_generated, generated)


def test_eta_one_with_all_steps_is_labeled_ancestral_ddpm() -> None:
    task, _, _ = _batch(timesteps=10)

    assert task.sampler_name(steps=10, eta=1.0) == "ddpm_ancestral"
    assert task.sampler_name(steps=5, eta=1.0) == "ddim"
    assert task.sampler_name(steps=10, eta=0.0) == "ddim"
