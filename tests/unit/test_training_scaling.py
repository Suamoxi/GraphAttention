import pytest
import torch

from graph_attention.data import (
    FieldCatalog,
    FieldRole,
    FieldSpec,
    FieldSupport,
    Mesh,
    Sample,
    SplitManifest,
)
from graph_attention.tasks import NodeRegressionTask
from graph_attention.training import fit_train_standardizers


_CATALOG = FieldCatalog(
    (
        FieldSpec(
            name="q",
            support=FieldSupport.NODE,
            role=FieldRole.PRIMARY_STATE,
            stored=False,
        ),
    )
)


def _sample(sample_id: str, values: list[float]) -> Sample:
    count = len(values)
    coords = torch.arange(count, dtype=torch.float64).unsqueeze(1)
    return Sample(
        sample_id=sample_id,
        mesh=Mesh(
            coords=coords,
            edge_index=torch.empty((2, 0), dtype=torch.long),
        ),
        fields={"q": torch.tensor(values, dtype=torch.float64)},
    )


def _task() -> NodeRegressionTask:
    return NodeRegressionTask(input_fields=("q",), target_fields=("q",))


def test_train_scaler_is_sample_balanced_not_node_balanced() -> None:
    small = _sample("small", [0.0, 2.0])
    large = _sample("large", [10.0] * 8)
    split = SplitManifest(train_ids=("small", "large"))

    scalers = fit_train_standardizers(_task(), [small, large], _CATALOG, split)

    expected_mean = torch.tensor([5.5], dtype=torch.float64)
    expected_std = torch.tensor([20.75**0.5], dtype=torch.float64)
    torch.testing.assert_close(scalers.inputs.mean, expected_mean)
    torch.testing.assert_close(scalers.inputs.scale, expected_std)
    torch.testing.assert_close(scalers.targets.mean, expected_mean)
    assert scalers.weighting == "sample_balanced"
    assert scalers.train_sample_ids == split.train_ids


def test_train_scaler_is_invariant_to_training_sample_iteration_order() -> None:
    first = _sample("a", [-1.0, 1.0, 3.0])
    second = _sample("b", [8.0, 10.0])
    split = SplitManifest(train_ids=("a", "b"))

    forward = fit_train_standardizers(_task(), [first, second], _CATALOG, split)
    reverse = fit_train_standardizers(_task(), [second, first], _CATALOG, split)

    torch.testing.assert_close(forward.inputs.mean, reverse.inputs.mean)
    torch.testing.assert_close(forward.inputs.scale, reverse.inputs.scale)


def test_train_scaler_rejects_non_training_and_missing_samples() -> None:
    train = _sample("train", [0.0, 2.0])
    validation = _sample("validation", [10.0, 12.0])
    split = SplitManifest(train_ids=("train",), validation_ids=("validation",))

    with pytest.raises(ValueError, match="not declared"):
        fit_train_standardizers(_task(), [train, validation], _CATALOG, split)

    split_missing = SplitManifest(train_ids=("train", "missing"))
    with pytest.raises(ValueError, match="did not contain every"):
        fit_train_standardizers(_task(), [train], _CATALOG, split_missing)


def test_train_scaler_rejects_duplicate_samples_and_near_zero_variance() -> None:
    varying = _sample("varying", [0.0, 2.0])
    split = SplitManifest(train_ids=("varying",))
    with pytest.raises(ValueError, match="more than once"):
        fit_train_standardizers(_task(), [varying, varying], _CATALOG, split)

    constant = _sample("constant", [4.0, 4.0, 4.0])
    constant_split = SplitManifest(train_ids=("constant",))
    with pytest.raises(ValueError, match="zero/near-zero variance"):
        fit_train_standardizers(_task(), [constant], _CATALOG, constant_split)


def test_task_standardizers_transform_and_inverse_targets() -> None:
    first = _sample("a", [0.0, 2.0])
    second = _sample("b", [6.0, 10.0, 14.0])
    split = SplitManifest(train_ids=("a", "b"))
    task = _task()
    scalers = fit_train_standardizers(task, [first, second], _CATALOG, split)
    batch = task.pack_and_prepare([first, second], _CATALOG)

    scaled = scalers.transform(batch)
    restored_targets = scalers.inverse_targets(scaled.targets, scaled.target_channels)

    assert scaled.source is batch.source
    torch.testing.assert_close(restored_targets, batch.targets)
    assert scaled.input_channels == batch.input_channels
    assert scaled.target_channels == batch.target_channels


def test_channel_standardizer_rejects_wrong_channel_semantics() -> None:
    first = _sample("a", [0.0, 2.0])
    second = _sample("b", [4.0, 8.0])
    split = SplitManifest(train_ids=("a", "b"))
    scalers = fit_train_standardizers(_task(), [first, second], _CATALOG, split)

    with pytest.raises(ValueError, match="channel order"):
        scalers.inputs.transform(torch.ones((2, 1), dtype=torch.float64), ("other.value",))
