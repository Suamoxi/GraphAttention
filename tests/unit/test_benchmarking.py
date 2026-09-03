import pytest
import torch

from graph_attention.utils.benchmarking import measure_callable


def test_measure_callable_reports_cpu_timing_without_cuda_memory() -> None:
    tensor = torch.ones(16)

    measurement = measure_callable(
        lambda: tensor.square().sum(),
        device="cpu",
        warmup=1,
        repetitions=3,
    )

    assert measurement.timing.warmup == 1
    assert measurement.timing.repetitions == 3
    assert measurement.timing.median_ms >= 0.0
    assert measurement.timing.min_ms >= 0.0
    assert measurement.timing.max_ms >= measurement.timing.min_ms
    assert measurement.cuda_memory is None


@pytest.mark.parametrize(
    ("warmup", "repetitions", "error"),
    [
        (-1, 1, ValueError),
        (0, 0, ValueError),
        (True, 1, TypeError),
        (0, False, TypeError),
    ],
)
def test_measure_callable_rejects_invalid_counts(
    warmup: int,
    repetitions: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        measure_callable(lambda: None, device="cpu", warmup=warmup, repetitions=repetitions)
