from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training import ChannelStandardizer, TaskStandardizers
from scripts.generate_generative import (
    _generate_test_population,
    _model_evaluations,
    _sampling_settings,
)


class _ZeroVelocity(torch.nn.Module):
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
        return torch.zeros_like(inputs)


def _batch():
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=7)
    task = FlowMatchingTask(state_fields=("rho", "momentum"), validation_seed=91)
    batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    return task, batch


def _identity_standardizers(channel_names: tuple[str, ...]) -> TaskStandardizers:
    mean = torch.zeros(len(channel_names), dtype=torch.float32)
    scale = torch.ones(len(channel_names), dtype=torch.float32)
    return TaskStandardizers(
        inputs=ChannelStandardizer(channel_names=channel_names, mean=mean, scale=scale),
        targets=ChannelStandardizer(
            channel_names=channel_names,
            mean=mean.clone(),
            scale=scale.clone(),
        ),
        train_sample_ids=("synthetic_train",),
        physical_nondimensionalization=False,
        weighting="sample_balanced",
    )


def test_flow_ode_sampling_settings_and_nfe() -> None:
    settings = _sampling_settings(
        OmegaConf.create(
            {
                "steps": 50,
                "method": "flow_ode",
                "solver": "heun",
                "sampling_eps": 0.0,
                "final_denoise": False,
                "seed": 5678,
            }
        )
    )
    assert settings["sampling_eps"] == 0.0
    assert settings["final_denoise"] is False
    assert _model_evaluations(
        steps=50,
        method="flow_ode",
        solver="heun",
        final_denoise=False,
    ) == 100

    with pytest.raises(ValueError, match="sampling_eps must be 0 for flow_ode"):
        _sampling_settings(
            OmegaConf.create(
                {
                    "steps": 50,
                    "method": "flow_ode",
                    "solver": "heun",
                    "sampling_eps": 1.0e-3,
                    "final_denoise": False,
                    "seed": 5678,
                }
            )
        )


def test_flow_task_exposes_generic_sampler_contract() -> None:
    task, batch = _batch()
    assert (
        task.sampler_name(
            steps=50,
            method="flow_ode",
            solver="heun",
            final_denoise=False,
        )
        == "flow_ode_heun_steps50"
    )

    first = task.sample_standardized(
        _ZeroVelocity(),
        batch,
        steps=2,
        method="flow_ode",
        solver="heun",
        sampling_eps=0.0,
        final_denoise=False,
        sampling_seed=44,
        sampling_keys=("gen_000000", "gen_000001"),
    )
    repeated = task.sample_standardized(
        _ZeroVelocity(),
        batch,
        steps=2,
        method="flow_ode",
        solver="heun",
        sampling_eps=0.0,
        final_denoise=False,
        sampling_seed=44,
        sampling_keys=("gen_000000", "gen_000001"),
    )
    torch.testing.assert_close(first, repeated, rtol=0.0, atol=0.0)

    changed_keys = task.sample_standardized(
        _ZeroVelocity(),
        batch,
        steps=2,
        method="flow_ode",
        solver="heun",
        sampling_eps=0.0,
        final_denoise=False,
        sampling_seed=44,
        sampling_keys=("gen_100000", "gen_100001"),
    )
    assert not torch.equal(first, changed_keys)


def test_generic_population_generation_supports_flow_matching() -> None:
    task, batch = _batch()
    standardizers = _identity_standardizers(batch.input_channels)
    sampling = {
        "steps": 2,
        "method": "flow_ode",
        "solver": "heun",
        "sampling_eps": 0.0,
        "final_denoise": False,
        "seed": 44,
    }
    sampler = task.sampler_name(
        steps=2,
        method="flow_ode",
        solver="heun",
        final_denoise=False,
    )

    artifact, generation = _generate_test_population(
        _ZeroVelocity(),
        [batch],
        task=task,
        standardizers=standardizers,
        device=torch.device("cpu"),
        sampling=sampling,
        sampler=sampler,
    )

    assert artifact["generated_ids"] == ("gen_000000", "gen_000001")
    assert artifact["reference_ids"] == batch.source.sample_ids
    assert artifact["generated_reference_pairing"] is False
    assert artifact["generated_nondimensional"].shape == batch.inputs.shape
    assert artifact["target_nondimensional"].shape == batch.inputs.shape
    assert artifact["model_evaluations"] == 4
    assert generation["model_evaluations"] == 4
    assert generation["initial_state_distribution"] == "standard_normal_at_t0"
    assert generation["reference_pairing"] is False
