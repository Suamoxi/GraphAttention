from pathlib import Path

import torch

from graph_attention.training import ChannelStandardizer, TaskStandardizers
from scripts.generate_slice_diffusion import _generation_name, _load_standardizers


def test_diffusion_generation_name_distinguishes_sampler_settings() -> None:
    assert (
        _generation_name(
            sampler="ddim",
            steps=100,
            eta=0.0,
            seed=5678,
            override=None,
        )
        == "ddim_steps100_eta0_seed5678"
    )
    assert (
        _generation_name(
            sampler="ddpm_ancestral",
            steps=1000,
            eta=1.0,
            seed=5678,
            override=None,
        )
        == "ddpm_ancestral_steps1000_eta1_seed5678"
    )


def test_saved_standardizers_reload_without_refitting(tmp_path: Path) -> None:
    standardizers = TaskStandardizers(
        inputs=ChannelStandardizer(
            channel_names=("rho.value", "rhou.x"),
            mean=torch.tensor([1.0, 2.0]),
            scale=torch.tensor([0.5, 3.0]),
        ),
        targets=ChannelStandardizer(
            channel_names=("rho.value", "rhou.x"),
            mean=torch.tensor([1.0, 2.0]),
            scale=torch.tensor([0.5, 3.0]),
        ),
        train_sample_ids=("train_0", "train_1"),
        physical_nondimensionalization=True,
    )
    path = tmp_path / "standardizers.pt"
    torch.save(
        {
            "weighting": standardizers.weighting,
            "physical_nondimensionalization": standardizers.physical_nondimensionalization,
            "train_sample_ids": standardizers.train_sample_ids,
            "inputs": {
                "channel_names": standardizers.inputs.channel_names,
                "mean": standardizers.inputs.mean,
                "scale": standardizers.inputs.scale,
            },
            "targets": {
                "channel_names": standardizers.targets.channel_names,
                "mean": standardizers.targets.mean,
                "scale": standardizers.targets.scale,
            },
        },
        path,
    )

    loaded = _load_standardizers(path)

    assert loaded.train_sample_ids == standardizers.train_sample_ids
    assert loaded.physical_nondimensionalization is True
    torch.testing.assert_close(loaded.inputs.mean, standardizers.inputs.mean)
    torch.testing.assert_close(loaded.inputs.scale, standardizers.inputs.scale)
