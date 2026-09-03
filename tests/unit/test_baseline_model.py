import pytest
import torch

from graph_attention.data import SyntheticMeshDataset
from graph_attention.models import NodeLinearBaseline
from graph_attention.tasks import NodeRegressionTask


def test_node_linear_baseline_maps_explicit_channel_counts() -> None:
    model = NodeLinearBaseline(in_channels=2, out_channels=3)
    inputs = torch.randn(5, 2)

    outputs = model(inputs)

    assert outputs.shape == (5, 3)


def test_node_linear_baseline_matches_packed_and_independent_execution() -> None:
    dataset = SyntheticMeshDataset(num_samples=3, spatial_dim=2, seed=19)
    task = NodeRegressionTask(input_fields=("momentum",), target_fields=("rho",))
    samples = [dataset[index] for index in range(3)]
    packed = task.pack_and_prepare(samples, dataset.field_catalog)

    torch.manual_seed(7)
    model = NodeLinearBaseline(in_channels=2, out_channels=1)
    packed_output = model(
        packed.inputs,
        batch_index=packed.batch_index,
        conditioning=packed.conditioning,
    )
    independent_outputs = []
    for sample in samples:
        single = task.pack_and_prepare([sample], dataset.field_catalog)
        independent_outputs.append(
            model(
                single.inputs,
                batch_index=single.batch_index,
                conditioning=single.conditioning,
            )
        )

    torch.testing.assert_close(packed_output, torch.cat(independent_outputs, dim=0))


def test_node_linear_baseline_is_node_renumbering_equivariant() -> None:
    torch.manual_seed(5)
    model = NodeLinearBaseline(
        in_channels=2,
        out_channels=3,
        conditioning_channels=1,
    )
    inputs = torch.randn(6, 2)
    batch_index = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.long)
    conditioning = torch.tensor([[0.1], [0.2]])
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])

    original = model(
        inputs,
        batch_index=batch_index,
        conditioning=conditioning,
    )
    permuted = model(
        inputs[permutation],
        batch_index=batch_index[permutation],
        conditioning=conditioning,
    )

    torch.testing.assert_close(permuted, original[permutation])


def test_node_linear_baseline_expands_graph_conditioning_per_node() -> None:
    model = NodeLinearBaseline(
        in_channels=1,
        out_channels=1,
        conditioning_channels=1,
        bias=False,
    )
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.0, 10.0]]))

    inputs = torch.tensor([[1.0], [2.0], [3.0]])
    batch_index = torch.tensor([0, 0, 1], dtype=torch.long)
    conditioning = torch.tensor([[0.5], [1.0]])

    output = model(
        inputs,
        batch_index=batch_index,
        conditioning=conditioning,
    )

    torch.testing.assert_close(output[:, 0], torch.tensor([6.0, 7.0, 13.0]))


def test_node_linear_baseline_rejects_incompatible_inputs_or_conditioning() -> None:
    model = NodeLinearBaseline(in_channels=2, out_channels=1)
    with pytest.raises(ValueError, match="expected 2"):
        model(torch.randn(3, 1))
    with pytest.raises(TypeError, match="expected model dtype"):
        model(torch.randn(3, 2, dtype=torch.float64))

    conditioned = NodeLinearBaseline(
        in_channels=2,
        out_channels=1,
        conditioning_channels=1,
    )
    with pytest.raises(ValueError, match="required"):
        conditioned(torch.randn(3, 2))
    with pytest.raises(TypeError, match="torch.long"):
        conditioned(
            torch.randn(3, 2),
            batch_index=torch.zeros(3, dtype=torch.int32),
            conditioning=torch.randn(1, 1),
        )


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"in_channels": 0, "out_channels": 1}, ValueError),
        ({"in_channels": 1, "out_channels": 0}, ValueError),
        (
            {"in_channels": 1, "out_channels": 1, "conditioning_channels": -1},
            ValueError,
        ),
        ({"in_channels": True, "out_channels": 1}, TypeError),
    ],
)
def test_node_linear_baseline_rejects_invalid_configuration(
    kwargs: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        NodeLinearBaseline(**kwargs)  # type: ignore[arg-type]
