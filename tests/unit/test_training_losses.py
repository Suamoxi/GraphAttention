import pytest
import torch

from graph_attention.training import sample_reduced_mse


def test_sample_reduced_mse_gives_equal_weight_to_different_graph_sizes() -> None:
    predictions = torch.tensor([[1.0], [1.0], *([[2.0]] * 8)])
    targets = torch.zeros_like(predictions)
    ptr = torch.tensor([0, 2, 10], dtype=torch.long)

    aggregate = sample_reduced_mse(predictions, targets, ptr)

    torch.testing.assert_close(aggregate.per_sample, torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(aggregate.mean, torch.tensor(2.5))
    assert aggregate.sample_count == 2


def test_sample_reduced_mse_averages_channels_before_spatial_reduction() -> None:
    predictions = torch.tensor([[1.0, 3.0], [3.0, 1.0]])
    targets = torch.zeros_like(predictions)
    ptr = torch.tensor([0, 2], dtype=torch.long)

    aggregate = sample_reduced_mse(predictions, targets, ptr)

    torch.testing.assert_close(aggregate.per_sample, torch.tensor([5.0]))


def test_sample_reduced_mse_uses_supplied_node_weights_within_each_graph() -> None:
    predictions = torch.tensor([[1.0], [3.0], [2.0], [4.0]])
    targets = torch.zeros_like(predictions)
    ptr = torch.tensor([0, 2, 4], dtype=torch.long)
    weights = torch.tensor([1.0, 3.0, 2.0, 2.0])

    aggregate = sample_reduced_mse(
        predictions,
        targets,
        ptr,
        node_weights=weights,
    )

    torch.testing.assert_close(aggregate.per_sample, torch.tensor([7.0, 10.0]))


def test_sample_reduced_mse_rejects_invalid_spatial_weights() -> None:
    predictions = torch.ones((4, 1))
    targets = torch.zeros_like(predictions)
    ptr = torch.tensor([0, 2, 4], dtype=torch.long)

    with pytest.raises(ValueError, match="non-negative"):
        sample_reduced_mse(
            predictions,
            targets,
            ptr,
            node_weights=torch.tensor([1.0, -1.0, 1.0, 1.0]),
        )

    with pytest.raises(ValueError, match="sum to zero"):
        sample_reduced_mse(
            predictions,
            targets,
            ptr,
            node_weights=torch.tensor([0.0, 0.0, 1.0, 1.0]),
        )


def test_sample_reduced_mse_rejects_invalid_shapes_ptr_and_nonfinite_values() -> None:
    predictions = torch.ones((3, 1))
    targets = torch.zeros_like(predictions)

    with pytest.raises(ValueError, match="share one shape"):
        sample_reduced_mse(predictions, torch.zeros((2, 1)), torch.tensor([0, 3]))

    with pytest.raises(ValueError, match="end at the total node count"):
        sample_reduced_mse(predictions, targets, torch.tensor([0, 2], dtype=torch.long))

    bad = predictions.clone()
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        sample_reduced_mse(bad, targets, torch.tensor([0, 3], dtype=torch.long))
