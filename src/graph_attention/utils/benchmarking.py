"""Small reproducible timing helpers for M7 performance measurements."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from operator import index as operator_index
from statistics import mean, median, pstdev
from time import perf_counter
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class CudaMemorySummary:
    """CUDA allocator state associated with one timed measurement."""

    allocated_before_bytes: int
    reserved_before_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    peak_incremental_allocated_bytes: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Repeated synchronized latency statistics in milliseconds."""

    warmup: int
    repetitions: int
    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    """Timing plus optional CUDA allocator measurements."""

    timing: TimingSummary
    cuda_memory: CudaMemorySummary | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "timing": self.timing.as_dict(),
            "cuda_memory": None if self.cuda_memory is None else self.cuda_memory.as_dict(),
        }


def measure_callable(
    function: Callable[[], object],
    *,
    device: torch.device | str,
    warmup: int = 10,
    repetitions: int = 50,
) -> BenchmarkMeasurement:
    """Measure a callable with explicit device synchronization.

    CUDA peak statistics are reset only after warmup. The reported peak therefore
    includes the resident model/input allocation that exists when timed execution
    begins plus any temporary allocation created during the measured calls.
    """

    warmup_count = _nonnegative_count(warmup, "warmup")
    repeat_count = _positive_count(repetitions, "repetitions")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")

    for _ in range(warmup_count):
        function()
        _synchronize(target_device)

    _synchronize(target_device)
    cuda_before: tuple[int, int] | None = None
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
        cuda_before = (
            torch.cuda.memory_allocated(target_device),
            torch.cuda.memory_reserved(target_device),
        )

    durations_ms: list[float] = []
    for _ in range(repeat_count):
        _synchronize(target_device)
        started = perf_counter()
        function()
        _synchronize(target_device)
        durations_ms.append((perf_counter() - started) * 1000.0)

    timing = TimingSummary(
        warmup=warmup_count,
        repetitions=repeat_count,
        median_ms=median(durations_ms),
        mean_ms=mean(durations_ms),
        std_ms=pstdev(durations_ms),
        min_ms=min(durations_ms),
        max_ms=max(durations_ms),
    )

    cuda_memory = None
    if cuda_before is not None:
        allocated_before, reserved_before = cuda_before
        peak_allocated = torch.cuda.max_memory_allocated(target_device)
        peak_reserved = torch.cuda.max_memory_reserved(target_device)
        cuda_memory = CudaMemorySummary(
            allocated_before_bytes=allocated_before,
            reserved_before_bytes=reserved_before,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            peak_incremental_allocated_bytes=max(peak_allocated - allocated_before, 0),
        )

    return BenchmarkMeasurement(timing=timing, cuda_memory=cuda_memory)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nonnegative_count(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        count = operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


def _positive_count(value: int, name: str) -> int:
    count = _nonnegative_count(value, name)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count
