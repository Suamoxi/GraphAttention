from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks import VPSDEDenoisingTask
from graph_attention.training import ChannelStandardizer, TaskStandardizers
from scripts.generate_generative import (
    _generate_test_population,
    _generation_name,
    _model_evaluations,
    _sampling_settings,
)


class _ZeroEpsilon(torch.nn.Module):
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


def _identity_standardizers(channel_names: tuple[str, ...]) -> TaskStandardizers:
    mean = torch.zeros(len(channel_names), dtype=torch.float32)
    scale = torch.ones(len(channel_names), dtype=torch.float32)
    return TaskStandardizers(
        inputs=ChannelStandardizer(channel_names=channel_names, mean=mean, scale=scale),
        targets=ChannelStandardizer(channel_names=channel_names, mean=mean.clone(), scale=scale.clone()),
        train_sample_ids=("synthetic_train",),
        physical_nondimensionalization=False,
        weighting="sample_balanced",
    )


def test_vp_generation_model_evaluation_counts_include_final_denoise() -> None:
    assert _model_evaluations(
        steps=1000,
        method="probability_flow_ode",
        solver="heun",
        final_denoise=True,
    ) == 2001
    assert _model_evaluations(
        steps=1000,
        method="reverse_sde",
        solver="euler_maruyama",
        final_denoise=True,
    ) == 1001
    assert _model_evaluations(
        steps=1000,
        method="discrete_ancestral",
        solver="ddpm",
        final_denoise=False,
    ) == 1000


def test_discrete_ancestral_sampling_settings_use_t0_without_final_denoise() -> None:
    settings = _sampling_settings(
        OmegaConf.create(
            {
                "steps": 1000,
                "method": "discrete_ancestral",
                "solver": "ddpm",
                "sampling_eps": 0.0,
                "final_denoise": False,
                "seed": 5678,
            }
        )
    )
    assert settings["sampling_eps"] == 0.0
    assert settings["final_denoise"] is False

    with pytest.raises(ValueError, match="must be 0 for discrete_ancestral"):
        _sampling_settings(
            OmegaConf.create(
                {
                    "steps": 1000,
                    "method": "discrete_ancestral",
                    "solver": "ddpm",
                    "sampling_eps": 1.0e-3,
                    "final_denoise": False,
                    "seed": 5678,
                }
            )
        )


def test_generation_name_is_stable_and_filesystem_safe() -> None:
    name = _generation_name(
        sampler="vp_probability_flow_heun_steps1000_denoise",
        sampling_eps=1.0e-3,
        seed=5678,
        override=None,
    )
    assert name == "vp_probability_flow_heun_steps1000_denoise_eps0p001_seed5678"


def test_generic_population_generation_preserves_unpaired_artifact_contract() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=7)
    task = VPSDEDenoisingTask(
        state_fields=("rho", "momentum"),
        beta_min=0.1,
        beta_max=20.0,
        training_eps=1.0e-5,
        validation_seed=91,
    )
    batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    standardizers = _identity_standardizers(batch.input_channels)
    sampling = {
        "steps": 2,
        "method": "probability_flow_ode",
        "solver": "heun",
        "sampling_eps": 1.0e-3,
        "final_denoise": True,
        "seed": 44,
    }
    sampler = task.sampler_name(
        steps=2,
        method="probability_flow_ode",
        solver="heun",
        final_denoise=True,
    )

    artifact, generation = _generate_test_population(
        _ZeroEpsilon(),
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
    assert generation["reference_pairing"] is False
    assert generation["model_evaluations"] == 5
