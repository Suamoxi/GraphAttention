"""Optional synchronized diagnostic timings for M41 CPU-streamed inference.

A synchronization before AND after each measured stage attributes queued
CUDA work to the stage that issued it. These timings are diagnostic WALL
times, not pure kernel timings: they include launch overhead, allocations,
transfer staging and synchronize overhead. Do not compare the profiled total
with the uninstrumented forward benchmark for performance decisions.

No I/O, graph, model, seed or tensor transformations are performed here.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Iterator

import torch


@dataclass
class StreamedStageProfiler:
    """In-memory bounded aggregate timers, never per-partition trace events."""

    stage_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    stage_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    layers_seconds: dict[int, dict[str, float]] = field(default_factory=dict)
    layers_calls: dict[int, dict[str, int]] = field(default_factory=dict)
    layer_wall_seconds: dict[int, float] = field(default_factory=dict)
    forward_wall_seconds: float | None = None
    synchronization_calls: int = 0

    def _sync(self, device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            self.synchronization_calls += 1

    @contextmanager
    def measure(
        self, stage: str, *, device: torch.device, layer_index: int | None = None
    ) -> Iterator[None]:
        """Measure disjoint stage wall time, synchronizing only when on CUDA."""
        self._sync(device)
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync(device)
            elapsed = time.perf_counter() - started
            self.stage_seconds[stage] += elapsed
            self.stage_calls[stage] += 1
            if layer_index is not None:
                stage_times = self.layers_seconds.setdefault(layer_index, defaultdict(float))
                stage_counts = self.layers_calls.setdefault(layer_index, defaultdict(int))
                stage_times[stage] += elapsed
                stage_counts[stage] += 1

    def report(self) -> dict:
        if self.forward_wall_seconds is None:
            raise RuntimeError("no profiled forward has completed")
        return {
            "timing_method": "synchronized_wall_seconds_not_cuda_kernel_time",
            "profiled_forward_wall_seconds": self.forward_wall_seconds,
            "stage_seconds": dict(sorted(self.stage_seconds.items())),
            "stage_calls": dict(sorted(self.stage_calls.items())),
            "stage_total_seconds": float(sum(self.stage_seconds.values())),
            "layer_wall_seconds": {
                str(k): v for k, v in sorted(self.layer_wall_seconds.items())
            },
            "layers_stage_seconds": {
                str(k): dict(sorted(v.items()))
                for k, v in sorted(self.layers_seconds.items())
            },
            "layers_stage_calls": {
                str(k): dict(sorted(v.items()))
                for k, v in sorted(self.layers_calls.items())
            },
            "synchronization_calls": self.synchronization_calls,
            "notes": [
                "The profile intentionally inserts synchronization and perturbs execution.",
                "Partition loading includes NPZ reading plus CPU CSR and displacement assembly.",
                "The attention stage includes geometry bias, sparse sampled_addmm, softmax, and aggregation.",
                "GPU stage timings are synchronized elapsed wall time, not isolated CUDA kernel time.",
                "Blocking owned GPU-to-CPU copies enforce per-partition completion; buffer swaps are cheap.",
                "Stage totals do not include every Python or allocation operation in forward.",
                "Compare unprofiled repeats for real forward latency, not the profiled total.",
            ],
        }
