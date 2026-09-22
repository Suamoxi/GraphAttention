import torch
from omegaconf import OmegaConf

from graph_attention.data import SyntheticMeshDataset
from graph_attention.models.full_dit import (
    FullDiTGraphTransformer,
    FullDiTMultiheadAttention,
)
from graph_attention.models.local_dit import (
    LocalDiTGraphTransformer,
    LocalDiTMultiheadAttention,
)
from graph_attention.tasks import EDMDenoisingTask
from graph_attention.training.model_factory import instantiate_controlled_model


def _complete_edge_index(num_nodes: int) -> torch.Tensor:
    source = torch.arange(num_nodes, dtype=torch.long).repeat(num_nodes)
    target = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(num_nodes)
    return torch.stack((source, target))


def _model_kwargs() -> dict[str, object]:
    return {
        "in_channels": 5,
        "out_channels": 5,
        "hidden_dim": 16,
        "num_heads": 4,
        "num_layers": 2,
        "spatial_dim": 2,
        "mlp_ratio": 2,
        "conditioning_channels": 1,
        "condition_embed_dim": 16,
        "use_coord_mlp": True,
        "coordinate_normalization": "centered_bbox",
        "use_sdpa": False,
    }


def test_full_and_local_dit_are_separate_model_modules() -> None:
    assert FullDiTGraphTransformer.__module__ == "graph_attention.models.full_dit"
    assert LocalDiTGraphTransformer.__module__ == "graph_attention.models.local_dit"


def test_full_and_local_attention_match_on_complete_graph() -> None:
    torch.manual_seed(7)
    full = FullDiTMultiheadAttention(16, 4, use_sdpa=False)
    local = LocalDiTMultiheadAttention(16, 4)
    local.load_state_dict(full.state_dict(), strict=True)

    inputs = torch.randn(6, 16)
    batch_index = torch.zeros(6, dtype=torch.long)
    complete = _complete_edge_index(6)

    full_output = full(
        inputs,
        edge_index=complete,
        batch_index=batch_index,
    )
    local_output = local(
        inputs,
        edge_index=complete,
        batch_index=batch_index,
    )

    torch.testing.assert_close(full_output, local_output, rtol=1.0e-5, atol=1.0e-6)


def test_full_attention_never_mixes_different_graphs() -> None:
    torch.manual_seed(11)
    attention = FullDiTMultiheadAttention(8, 2, use_sdpa=False)
    inputs = torch.randn(6, 8)
    batch_index = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    empty_edges = torch.empty((2, 0), dtype=torch.long)

    reference = attention(
        inputs,
        edge_index=empty_edges,
        batch_index=batch_index,
    )
    changed = inputs.clone()
    changed[3:] += 100.0
    perturbed = attention(
        changed,
        edge_index=empty_edges,
        batch_index=batch_index,
    )

    torch.testing.assert_close(reference[:3], perturbed[:3], rtol=0.0, atol=0.0)


def test_full_and_local_dit_have_identical_parameter_contracts() -> None:
    torch.manual_seed(13)
    full = FullDiTGraphTransformer(**_model_kwargs())
    torch.manual_seed(99)
    local = LocalDiTGraphTransformer(**_model_kwargs())

    incompatible = local.load_state_dict(full.state_dict(), strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert list(full.state_dict()) == list(local.state_dict())
    assert sum(parameter.numel() for parameter in full.parameters()) == sum(
        parameter.numel() for parameter in local.parameters()
    )


def test_dit_factory_matches_full_and_local_initialization_for_edm() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=5)
    task = EDMDenoisingTask(
        state_fields=("rho", "momentum"),
        physical_nondimensionalization=False,
        validation_seed=91,
    )
    batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    problem = task.make_validation_problem(batch)
    probe = task.make_model_probe(problem)

    common = {
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
    full_cfg = OmegaConf.create(
        {
            "_target_": "graph_attention.models.full_dit.FullDiTGraphTransformer",
            **common,
        }
    )
    local_cfg = OmegaConf.create(
        {
            "_target_": "graph_attention.models.local_dit.LocalDiTGraphTransformer",
            **common,
        }
    )

    full, full_metadata = instantiate_controlled_model(full_cfg, probe, seed=42)
    local, local_metadata = instantiate_controlled_model(local_cfg, probe, seed=42)

    assert full_metadata["policy"] == "matched_full_local_dit_initialization"
    assert local_metadata["policy"] == "matched_full_local_dit_initialization"
    for name, full_value in full.state_dict().items():
        torch.testing.assert_close(
            full_value,
            local.state_dict()[name],
            rtol=0.0,
            atol=0.0,
        )

    full_output = full(
        problem.model_batch.inputs,
        edge_index=problem.model_batch.edge_index,
        coords=problem.model_batch.coords,
        batch_index=problem.model_batch.batch_index,
        conditioning=problem.model_batch.conditioning,
    )
    local_output = local(
        problem.model_batch.inputs,
        edge_index=problem.model_batch.edge_index,
        coords=problem.model_batch.coords,
        batch_index=problem.model_batch.batch_index,
        conditioning=problem.model_batch.conditioning,
    )

    assert full_output.shape == problem.model_batch.inputs.shape
    assert local_output.shape == problem.model_batch.inputs.shape
    torch.testing.assert_close(
        full_output,
        torch.zeros_like(full_output),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        local_output,
        torch.zeros_like(local_output),
        rtol=0.0,
        atol=0.0,
    )
