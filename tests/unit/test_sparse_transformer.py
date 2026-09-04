import math

import pytest
import torch

from graph_attention.data import SplitManifest, SyntheticMeshDataset
from graph_attention.models import SparseGraphTransformer, SparseMultiheadAttention
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


def _dense_reference(
    attention: SparseMultiheadAttention,
    inputs: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    num_nodes = inputs.shape[0]
    qkv = attention.qkv(inputs).reshape(
        num_nodes,
        3,
        attention.num_heads,
        attention.head_dim,
    )
    query, key, value = qkv.unbind(dim=1)
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
        weights = torch.softmax(scores, dim=0)
        aggregated[node] = (weights.unsqueeze(-1) * value[neighbors]).sum(dim=0)

    return attention.out_proj(aggregated.reshape(num_nodes, attention.hidden_dim))


def test_sparse_attention_matches_explicit_neighbor_reference() -> None:
    torch.manual_seed(3)
    attention = SparseMultiheadAttention(hidden_dim=12, num_heads=3)
    inputs = torch.randn(6, 12)
    edge_index = torch.tensor(
        [
            [0, 2, 1, 2, 4, 3, 5],
            [1, 1, 2, 3, 3, 4, 4],
        ],
        dtype=torch.long,
    )

    sparse = attention(inputs, edge_index)
    reference = _dense_reference(attention, inputs, edge_index)

    torch.testing.assert_close(sparse, reference, rtol=1e-5, atol=1e-6)


def test_sparse_attention_is_invariant_to_edge_list_order() -> None:
    torch.manual_seed(5)
    attention = SparseMultiheadAttention(hidden_dim=16, num_heads=4)
    inputs = torch.randn(8, 16)
    edge_index = _directed_chain(8)
    permutation = torch.randperm(edge_index.shape[1])

    reference = attention(inputs, edge_index)
    reordered = attention(inputs, edge_index[:, permutation])

    torch.testing.assert_close(reordered, reference, rtol=1e-5, atol=1e-6)


def test_sparse_attention_supports_cpu_bfloat16_autocast() -> None:
    torch.manual_seed(6)
    attention = SparseMultiheadAttention(hidden_dim=16, num_heads=4)
    inputs = torch.randn(8, 16)
    edge_index = _directed_chain(8)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = attention(inputs, edge_index)

    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()


def test_sparse_transformer_matches_packed_and_independent_execution() -> None:
    dataset = SyntheticMeshDataset(num_samples=3, spatial_dim=2, seed=31)
    task = NodeRegressionTask(input_fields=("momentum",), target_fields=("rho",))
    samples = [dataset[index] for index in range(3)]
    packed = task.pack_and_prepare(samples, dataset.field_catalog)

    torch.manual_seed(7)
    model = SparseGraphTransformer(
        in_channels=2,
        out_channels=1,
        hidden_dim=16,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2,
    )
    packed_output = model(
        packed.inputs,
        edge_index=packed.edge_index,
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


def test_sparse_transformer_is_node_renumbering_equivariant() -> None:
    torch.manual_seed(11)
    model = SparseGraphTransformer(
        in_channels=2,
        out_channels=3,
        hidden_dim=24,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2,
        conditioning_channels=1,
    )
    inputs = torch.randn(7, 2)
    edge_index = _directed_chain(7)
    batch_index = torch.zeros(7, dtype=torch.long)
    conditioning = torch.tensor([[0.25]])
    permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())

    original = model(
        inputs,
        edge_index=edge_index,
        batch_index=batch_index,
        conditioning=conditioning,
    )
    permuted = model(
        inputs[permutation],
        edge_index=inverse[edge_index],
        batch_index=batch_index[permutation],
        conditioning=conditioning,
    )

    torch.testing.assert_close(permuted, original[permutation], rtol=1e-5, atol=1e-6)


def test_sparse_transformer_handles_empty_edges_without_nan() -> None:
    torch.manual_seed(13)
    model = SparseGraphTransformer(
        in_channels=2,
        out_channels=1,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
    )
    output = model(
        torch.randn(5, 2),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )

    assert output.shape == (5, 1)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_sparse_transformer_supports_m6_training_step() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=37)
    samples = [dataset[0], dataset[1]]
    task = NodeRegressionTask(input_fields=("momentum",), target_fields=("rho",))
    split = SplitManifest(train_ids=tuple(sample.sample_id for sample in samples))
    standardizers = fit_train_standardizers(task, samples, dataset.field_catalog, split)
    batch = task.pack_and_prepare(samples, dataset.field_catalog)

    torch.manual_seed(17)
    model = SparseGraphTransformer(
        in_channels=2,
        out_channels=1,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
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


def test_sparse_transformer_rejects_invalid_edges() -> None:
    model = SparseGraphTransformer(
        in_channels=2,
        out_channels=1,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
    )
    inputs = torch.randn(4, 2)

    with pytest.raises(TypeError, match="torch.long"):
        model(inputs, edge_index=torch.zeros((2, 2), dtype=torch.int32))
    with pytest.raises(ValueError, match="out-of-range"):
        model(
            inputs,
            edge_index=torch.tensor([[0, 4], [1, 2]], dtype=torch.long),
        )


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"hidden_dim": 0}, ValueError),
        ({"num_heads": 0}, ValueError),
        ({"num_layers": 0}, ValueError),
        ({"mlp_ratio": 0}, ValueError),
        ({"conditioning_channels": -1}, ValueError),
        ({"hidden_dim": 10, "num_heads": 4}, ValueError),
        ({"num_heads": True}, TypeError),
    ],
)
def test_sparse_transformer_rejects_invalid_configuration(
    kwargs: dict[str, object],
    error_type: type[Exception],
) -> None:
    configuration = {
        "in_channels": 2,
        "out_channels": 1,
        "hidden_dim": 16,
        "num_heads": 4,
        "num_layers": 1,
        "mlp_ratio": 2,
        "conditioning_channels": 0,
    }
    configuration.update(kwargs)

    with pytest.raises(error_type):
        SparseGraphTransformer(**configuration)  # type: ignore[arg-type]
