"""Summarize PyTorch CPU/CUDA events from a single, full 129^3 training step.

The reports are intentionally descriptive. Inclusive CPU operator time,
exclusive CUDA kernel time, and synchronized wall time are different quantities.
An event can have more than one associated kernel. A tiny amount of CUDA overlap
may cause additive kernel durations to exceed elapsed wall time.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


PHASE_PREFIX = "m43_profile/"


def phase_of_event(event: Any) -> str:
    """Categorize GPU kernels by nearest parent record_function annotation."""
    seen: set[int] = set()
    cursor = event
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        name = str(getattr(cursor, "name", ""))
        if name.startswith(PHASE_PREFIX):
            return name.removeprefix(PHASE_PREFIX)
        cursor = getattr(cursor, "cpu_parent", None)
    return "unattributed"


def kernel_groups(events: Iterable[Any], *, limit: int = 40) -> dict[str, Any]:
    """Aggregate device kernel durations (microseconds) by phase and kernel."""
    kernels: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    by_phase: dict[str, float] = defaultdict(float)
    for event in events:
        phase = phase_of_event(event)
        for kernel in (getattr(event, "kernels", ()) or ()):
            name = str(getattr(kernel, "name", "<unknown>"))
            us = float(getattr(kernel, "duration", 0.0))
            if us < 0:
                continue
            bucket = kernels[(phase, name)]
            bucket[0] += us
            bucket[1] += 1
            by_phase[phase] += us
    total_us = sum(by_phase.values())
    ranked = sorted(kernels.items(), key=lambda item: item[1][0], reverse=True)
    return {
        "kernel_count": sum(int(bucket[1]) for bucket in kernels.values()),
        "sum_cuda_kernel_seconds": total_us / 1e6,
        "kernel_time_seconds_by_phase": {
            name: value / 1e6 for name, value in sorted(by_phase.items())
        },
        "top_cuda_kernels": [
            {
                "phase": phase,
                "name": name,
                "launches": int(value[1]),
                "total_seconds": value[0] / 1e6,
                "fraction_of_kernel_duration": value[0] / total_us
                if total_us else None,
            }
            for (phase, name), value in ranked[:limit]
        ],
    }


def _self_gpu_microseconds(op: Any) -> float:
    for name in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(op, name, None)
        if value is not None:
            return max(0.0, float(value))
    return 0.0


def operator_groups(operators: Iterable[Any], *, limit: int = 40) -> dict[str, Any]:
    ranked = sorted(operators, key=_self_gpu_microseconds, reverse=True)
    total_gpu = sum(_self_gpu_microseconds(op) for op in ranked)
    return {
        "sum_exclusive_gpu_operator_seconds": total_gpu / 1e6,
        "top_pytorch_operators": [
            {
                "name": str(getattr(op, "key", getattr(op, "name", "<unknown>"))),
                "call_count": int(getattr(op, "count", 1)),
                "self_gpu_seconds": _self_gpu_microseconds(op) / 1e6,
                "fraction_of_self_gpu_operator_time": (
                    _self_gpu_microseconds(op) / total_gpu
                    if total_gpu else None
                ),
                "self_cpu_seconds": float(
                    getattr(op, "self_cpu_time_total", 0)
                ) / 1e6,
            }
            for op in ranked[:limit]
        ],
    }


def format_table(entries: list[dict], *, name_key: str, seconds_key: str) -> str:
    lines = [f"{'Seconds':>12} {'Calls':>10}  Name", "-" * 82]
    for entry in entries:
        count = entry.get("launches", entry.get("call_count", 0))
        scope = str(entry.get("phase", ""))
        name = str(entry.get(name_key, ""))
        if scope:
            name = f"{scope}: {name}"
        lines.append(f"{entry[seconds_key]:12.4f} {count:10}  {name}")
    return "\n".join(lines) + "\n"
