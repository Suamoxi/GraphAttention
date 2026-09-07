import math

import pytest
import torch

from graph_attention.data import SplitManifest, SyntheticMeshDataset
from graph_attention.geometry import edge_relative_displacement
from graph_attention.models import (
    GeometricSparseGraphTransformer,
    GeometricSparseMultiheadAttention,
)
from graph_attention.tasks import NodeRegressionTask
from graph_attention.training import fit_train_standardizers, train_equal_sample_optimizer_step


def _directed_chain(num_nodes: int) -> torch.Tensor:
    forward = torch.arange(num_nodes - 1, dtype=torch.long)
    reverse = forward + 1
    return torch.stack(
        (
            torch.cat((forward, reverse)),
            torch.cat((reverse, forward)),
        )
    )


def _explicit_reference(
    attention: GeometricSparseMultiheadAttention,
    inputs: torch.Tensor,
    edge_index: torch.Tensor,
    edge_displacement: torch.Tensor,
) -> torch.Tensor:
    num_nodes = inputs.shape[0]
    qkv = attention.qkv(inputs).reshape(
        num_nodes,
        3,
        attention.num_heads,
        attention.head_dim,
    )
    query, key, value = qkv.unbind(dim=1)
    geometry_bias = attention.geometry_mlp(edge_displacement)
    aggregated = torch.zeros_like(query)

    source = edge_index[0]
    target = edge_index[1]
    for node in range(num_nodes):
        mask = target == node
        if not bool(mask.any()):
            continue
        neighbors = source[mask]
        scores = (query[node].unsqueeze(0) * key[neighbors]).sum(dim=-1)
        scores = scores / math.sqrt(attention.head_dim)
        scores = scores + geometry_bias[mask]
        weights = torch.softmax(scores, dim=0)
        aggregated[node] = (weights.unsqueeze(-1) * value[neighbors]).sum(dim=0)

    return attention.out_proj(aggregated.reshape(num_nodes, attention.hidden_dim))


def test_geometric_attention_matches_explicit_reference() -> None:
    torch.manual_seed(3)
    attention = GeometricSparseMultiheadAttention(hidden_dim=12, num_heads=3, spatial_dim=2)
    inputs = torch.randn(6, 12)
    coords = torch.randn(6, 2)
    edge_index = torch.tensor(
        [[0, 2, 1, 2, 4, 3, 5], [1, 1, 2, 3, 3, 4, 4]],
        dtype=torch.long,
    )
    displacement = edge_relative_displacement(coords, edge_index)

    sparse = attention(inputs, edge_index, displacement)
    reference = _explicit_reference(attention, inputs, edge_index, displacement)

    torch.testing.assert_close(sparse, reference, rtol=1e-5, atol=1e-6)


def test_geometric_transformer_is_translation_invariant() -> None:
    torch.manual_seed(5)
    model = GeometricSparseGraphTransformer(
        in_channels=2,
        out_channels=1,
        hidden_dim=16,
        num_heads=4,
        num_layers=2,
        spatial_dim=2,
        mlp_ratio=2,
    )
    inputs = torch.randn(8, 2)
    coords = torch.randn(8, 2)
    edge_index = _directed_chain(8)

    reference = model(inputs, edge_index=edge_index, coords=coords)
    translated = model(
        inputs,
        edge_index=edge_index,
        coords=coords + torch.tensor([2.0, -1.5]),
    )

    torch.testing.assert_close(translated, reference, rtol=1e-5, atol=1e-6)


def test_geometric_transformer_changes_when_relative_geometry_changes() -> None:
    torch.manual_seed(7)
    model = GeometricSparseGraphTransformer(
        in_channels=2,
        out_channels=1,
        hidden_dim=16,
        num_heads=4,
        num_layers=2,
        spatial_dim=2,
        mlp_ratio=2,
    )
    inputs = torch.randn(7, 2)
    coords = torch.randn(7, 2)
    edge_index = _directed_chain(7)
    changed_coords = coords.clone()
    changed_coords[3, 0] += 0.75

    reference = model(inputs, edge_index=edge_index, coords=coords)
    changed = model(inputs, edge_index=edge_index, coords=changed_coords)

    assert not torch.allclose(changed, reference)


def test_geometric_transformer_matches_packed_and_independent_execution() -> None:
    dataset = SyntheticMeshDataset(num_samples=3, spatial_dim=2, seed=31)
    task = NodeRegressionTask(input_fields=("momentum",), target_fields=("rho",))
    samples = [dataset[index] for index in range(3)]
    packed = task.pack_and_prepare(samples, dataset.field_catalog)

    torch.manual_seed(11)
    model = GeometricSparseGraphTransformer(
        in_channels=2,
        out_channels=1,
        hidden_dim=16,
        num_heads=4,
        num_layers=2,
        spatial_dim=2,
        mlp_ratio=2,
    )
    packed_output = model(
        packed.inputs,
        edge_index=packed.edge_index,
        coords=packed.coords,
        batch_index=packed.batch_index,
        conditioning=packed.conditioning,
    )

    independent = []
    for sample in samples:
        single = task.pack_and_prepare([sample], dataset.field_catalog)
        independent.append(
            model(
                single.inputs,
                edge_index=single.edge_index,
                coords=single.coords,
                batch_index=single.batch_index,
                conditioning=single.conditioning,
            )
        )

    torch.testing.assert_close(
        packed_output,
        torch.cat(independent, dim=0),
        rtol=1e-5,
        atol=1e-6,
    )


def test_geometric_transformer_is_node_renumbering_equivariant() -> None:
    torch.manual_seed(13)
    model = GeometricSparseGraphTransformer(
        in_channels=2,
        out_channels=3,
        hidden_dim=24,
        num_heads=4,
        num_layers=2,
        spatial_dim=2,
        mlp_ratio=2,
        conditioning_channels=1,
    )
    inputs = torch.randn(7, 2)
    coords = torch.randn(7, 2)
    edge_index = _directed_chain(7)
    batch_index = torch.zeros(7, dtype=torch.long)
    conditioning = torch.tensor([[0.25]])
    permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())

    original = model(
        inputs,
        edge_index=edge_index,
        coords=coords,
        batch_index=batch_index,
        conditioning=conditioning,
    )
    permuted = model(
        inputs[permutation],
        edge_index=inverse[edge_index],
        coords=coords[permutation],
        batch_index=batch_index[permutation],
        conditioning=conditioning,
    )

    torch.testing.assert_close(permuted, original[permutation], rtol=1e-5, atol=1e-6)


def test_geometric_transformer_supports_m6_training_step() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=37)
    samples = [dataset[0], dataset[1]]
    task = NodeRegressionTask(input_fields=("momentum",), target_fields=("rho",))
    split = SplitManifest(train_ids=tuple(sample.sample_id for sample in samples))
    standardizers = fit_train_standardizers(task, samples, dataset.field_catalog, split)
    batch = task.pack_and_prepare(samples, dataset.field_catalog)

    torch.manual_seed(17)
    model = GeometricSparseGraphTransformer(
        in_channels=2,
        out_channels=1,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        spatial_dim=2,
        mlp_ratio=2,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    result = train_equal_sample_optimizer_step(
        model,
        optimizer,
        [batch],
        local_sample_count=2,
        standardizers=standardizers,
    )

    assert torch.isfinite(result.objective)
    assert result.local_sample_count == 2


def test_geometric_transformer_rejects_incompatible_coordinates() -> None:
    model = GeometricSparseGraphTransformer(
        in_channels=2,
        out_channels=1,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        spatial_dim=2,
    )
    inputs = torch.randn(4, 2)
    edge_index = _directed_chain(4)

    with pytest.raises(ValueError, match="coords must have shape"):
        model(inputs, edge_index=edge_index, coords=torch.randn(4, 3))
    with pytest.raises(TypeError, match="share one dtype"):
        model(inputs, edge_index=edge_index, coords=torch.randn(4, 2, dtype=torch.float64))
    coords = torch.randn(4, 2)
    coords[0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        model(inputs, edge_index=edge_index, coords=coords)


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"spatial_dim": 0}, ValueError),
        ({"spatial_dim": True}, TypeError),
        ({"hidden_dim": 10, "num_heads": 4}, ValueError),
    ],
)
def test_geometric_transformer_rejects_invalid_configuration(
    kwargs: dict[str, object],
    error_type: type[Exception],
) -> None:
    configuration = {
        "in_channels": 2,
        "out_channels": 1,
        "hidden_dim": 16,
        "num_heads": 4,
        "num_layers": 1,
        "spatial_dim": 2,
        "mlp_ratio": 2,
        "conditioning_channels": 0,
    }
    configuration.update(kwargs)

    with pytest.raises(error_type):
        GeometricSparseGraphTransformer(**configuration)  # type: ignore[arg-type]
