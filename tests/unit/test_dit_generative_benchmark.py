import torch
from omegaconf import OmegaConf

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks import EDMDenoisingTask, FlowMatchingTask
from graph_attention.tasks.linear_diffusion import LinearBetaDiffusionDenoisingTask
from graph_attention.training.model_factory import instantiate_controlled_model


def _full_dit_config():
    return OmegaConf.create(
        {
            "_target_": "graph_attention.models.full_dit.FullDiTGraphTransformer",
            "hidden_dim": 16,
            "num_heads": 4,
            "num_layers": 2,
            "spatial_dim": 2,
            "mlp_ratio": 2,
            "condition_embed_dim": 16,
            "use_coord_mlp": True,
            "coordinate_normalization": "centered_bbox",
            "coordinate_normalization_eps": 1.0e-8,
            "use_sdpa": False,
        }
    )


def test_full_dit_initialization_is_identical_across_ddpm_edm_and_flow() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=5)
    tasks = (
        LinearBetaDiffusionDenoisingTask(
            state_fields=("rho", "momentum"),
            physical_nondimensionalization=False,
            timesteps=1000,
            beta_start=1.0e-4,
            beta_end=2.0e-2,
            validation_seed=91,
        ),
        EDMDenoisingTask(
            state_fields=("rho", "momentum"),
            physical_nondimensionalization=False,
            validation_seed=91,
        ),
        FlowMatchingTask(
            state_fields=("rho", "momentum"),
            physical_nondimensionalization=False,
            validation_seed=91,
        ),
    )

    states = []
    for task in tasks:
        batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
        problem = task.make_validation_problem(batch)
        probe = task.make_model_probe(problem)
        assert probe.conditioning.shape[1] == 1

        model, metadata = instantiate_controlled_model(
            _full_dit_config(),
            probe,
            seed=42,
        )
        assert metadata["policy"] == "matched_full_local_dit_initialization"
        states.append(
            {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        )

    assert list(states[0]) == list(states[1]) == list(states[2])
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0.0, atol=0.0)
        torch.testing.assert_close(states[0][name], states[2][name], rtol=0.0, atol=0.0)
