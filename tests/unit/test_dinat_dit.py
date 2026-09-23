import pytest
import torch
from omegaconf import OmegaConf

from graph_attention.data import SyntheticMeshDataset
from graph_attention.geometry import edge_relative_displacement
from graph_attention.models.dinat_dit import (
    AlternatingDilatedGeometricDiT,
    DiNATDiTMultiheadAttention,
)
from graph_attention.models.geometric_transformer import GeometricSparseMultiheadAttention
from graph_attention.tasks import EDMDenoisingTask
from graph_attention.training.model_factory import instantiate_controlled_model


class _RecordingGeometricAttention(torch.nn.Module):
    def __init__(self, calls: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        super().__init__()
        self.calls = calls

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor | None,
        edge_displacement: torch.Tensor,
    ) -> torch.Tensor:
        del batch_index
        self.calls.append((edge_index.detach().clone(), edge_displacement.detach().clone()))
        return torch.zeros_like(inputs)


def _tiny_model(num_layers: int = 4) -> AlternatingDilatedGeometricDiT:
    return AlternatingDilatedGeometricDiT(
        in_channels=3,
        out_channels=3,
        hidden_dim=8,
        num_heads=2,
        num_layers=num_layers,
        spatial_dim=2,
        mlp_ratio=2,
        conditioning_channels=1,
        condition_embed_dim=8,
        use_coord_mlp=True,
        coordinate_normalization="centered_bbox",
        use_sdpa=False,
    )


def test_dinat_dit_attention_reuses_m12_geometric_attention_math() -> None:
    torch.manual_seed(9)
    reference = GeometricSparseMultiheadAttention(8, 2, 2)
    candidate = DiNATDiTMultiheadAttention(8, 2, 2)
    candidate.load_state_dict(reference.state_dict(), strict=True)

    inputs = torch.randn(5, 8)
    coords = torch.randn(5, 2)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 0, 1]],
        dtype=torch.long,
    )
    displacement = edge_relative_displacement(coords, edge_index)
    batch_index = torch.zeros(5, dtype=torch.long)

    expected = reference(inputs, edge_index, displacement)
    actual = candidate(
        inputs,
        edge_index=edge_index,
        batch_index=batch_index,
        edge_displacement=displacement,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_dinat_dit_alternates_local_and_exact_two_hop_topologies() -> None:
    model = _tiny_model(num_layers=4)
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    for block in model.blocks:
        block.attention = _RecordingGeometricAttention(calls)

    inputs = torch.randn(4, 3)
    coords = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    )
    batch_index = torch.zeros(4, dtype=torch.long)
    conditioning = torch.tensor([[0.25]])
    local = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    dilated = torch.tensor(
        [[0, 2, 1, 3], [2, 0, 3, 1]],
        dtype=torch.long,
    )

    output = model(
        inputs,
        edge_index=local,
        coords=coords,
        batch_index=batch_index,
        conditioning=conditioning,
        attention_edge_indices={"dilated": dilated},
    )

    assert output.shape == inputs.shape
    assert len(calls) == 4
    assert torch.equal(calls[0][0], local)
    assert torch.equal(calls[1][0], dilated)
    assert torch.equal(calls[2][0], local)
    assert torch.equal(calls[3][0], dilated)
    torch.testing.assert_close(calls[0][1], edge_relative_displacement(coords, local))
    torch.testing.assert_close(calls[1][1], edge_relative_displacement(coords, dilated))


def test_dinat_dit_factory_matches_full_dit_shared_parameters() -> None:
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
        "num_layers": 4,
        "spatial_dim": 2,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "condition_embed_dim": 16,
        "use_coord_mlp": True,
        "coordinate_normalization": "centered_bbox",
        "coordinate_normalization_eps": 1.0e-8,
        "use_sdpa": False,
        "qkv_bias": True,
        "out_proj_bias": False,
    }
    full_cfg = OmegaConf.create(
        {
            "_target_": "graph_attention.models.full_dit.FullDiTGraphTransformer",
            **common,
        }
    )
    dinat_cfg = OmegaConf.create(
        {
            "_target_": "graph_attention.models.dinat_dit.AlternatingDilatedGeometricDiT",
            **common,
        }
    )

    full, _ = instantiate_controlled_model(full_cfg, probe, seed=42)
    dinat, metadata = instantiate_controlled_model(dinat_cfg, probe, seed=42)

    assert metadata["policy"] == "matched_full_dit_shared_parameters_plus_dinat_geometry"
    assert metadata["layer_topology_schedule"] == "local_exact2hop_alternating_local_first"

    dinat_state = dinat.state_dict()
    geometry_names = set(metadata["geometry_parameter_names"])
    assert geometry_names
    for name, value in full.state_dict().items():
        assert name not in geometry_names
        torch.testing.assert_close(value, dinat_state[name], rtol=0.0, atol=0.0)


def test_dinat_dit_uses_adaln_zero_output_initialization() -> None:
    model = _tiny_model(num_layers=2)
    inputs = torch.randn(4, 3)
    coords = torch.randn(4, 2)
    batch_index = torch.zeros(4, dtype=torch.long)
    conditioning = torch.tensor([[0.4]])
    local = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    dilated = torch.tensor([[0, 2, 1, 3], [2, 0, 3, 1]], dtype=torch.long)

    output = model(
        inputs,
        edge_index=local,
        coords=coords,
        batch_index=batch_index,
        conditioning=conditioning,
        attention_edge_indices={"dilated": dilated},
    )

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0.0, atol=0.0)


def test_dinat_dit_rejects_unknown_sparse_attention_backend() -> None:
    with pytest.raises(ValueError, match="sparse_attention_backend"):
        DiNATDiTMultiheadAttention(
            8,
            2,
            2,
            sparse_attention_backend="unknown",
        )


def test_dinat_dit_dgl_backend_matches_scatter_when_available() -> None:
    try:
        import dgl.sparse  # noqa: F401
    except Exception as exc:
        pytest.skip(f"DGL sparse backend unavailable: {exc}")

    torch.manual_seed(17)
    scatter = DiNATDiTMultiheadAttention(
        8,
        2,
        2,
        sparse_attention_backend="scatter",
    )
    dgl = DiNATDiTMultiheadAttention(
        8,
        2,
        2,
        sparse_attention_backend="dgl",
    )
    dgl.load_state_dict(scatter.state_dict(), strict=True)

    coords = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 2, 1, 3],
            [1, 0, 3, 2, 2, 0, 3, 1],
        ],
        dtype=torch.long,
    )
    displacement = edge_relative_displacement(coords, edge_index)
    batch_index = torch.zeros(4, dtype=torch.long)

    scatter_inputs = torch.randn(4, 8, requires_grad=True)
    dgl_inputs = scatter_inputs.detach().clone().requires_grad_(True)

    scatter_output = scatter(
        scatter_inputs,
        edge_index=edge_index,
        batch_index=batch_index,
        edge_displacement=displacement,
    )
    dgl_output = dgl(
        dgl_inputs,
        edge_index=edge_index,
        batch_index=batch_index,
        edge_displacement=displacement,
    )

    torch.testing.assert_close(dgl_output, scatter_output, rtol=1.0e-5, atol=1.0e-6)

    scatter_output.square().sum().backward()
    dgl_output.square().sum().backward()
    torch.testing.assert_close(
        dgl_inputs.grad,
        scatter_inputs.grad,
        rtol=1.0e-4,
        atol=1.0e-5,
    )

    for (scatter_name, scatter_param), (dgl_name, dgl_param) in zip(
        scatter.named_parameters(),
        dgl.named_parameters(),
        strict=True,
    ):
        assert scatter_name == dgl_name
        assert scatter_param.grad is not None
        assert dgl_param.grad is not None
        torch.testing.assert_close(
            dgl_param.grad,
            scatter_param.grad,
            rtol=1.0e-4,
            atol=1.0e-5,
        )
