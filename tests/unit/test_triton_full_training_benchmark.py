"""CPU-only contracts for the paired real-HDF5 M37 performance pilot."""
from __future__ import annotations

import pytest

from scripts.benchmark_triton_full_training import measured_steps
from scripts.m43_b1_training_step import model_config


def _reading(step: int) -> dict:
    return {
        "forward_seconds": float(step),
        "backward_seconds": float(step * 2),
        "optimizer_seconds": float(step * 3),
        "step_seconds": float(step * 6),
        "peak_allocated_gib": float(step),
        "peak_reserved_gib": float(step + 1),
        "loss_probe_only": 0.01 * step,
    }


def test_excludes_first_compilation_step_from_measured_medians():
    report = {"status": "ok", "per_step": [_reading(i) for i in range(1, 5)]}
    result = measured_steps(report, warmup=1, expected_steps=4)
    assert result is not None
    assert result["measured_steps"] == 3
    assert result["median_seconds"]["forward_seconds"] == 3.0
    assert result["median_seconds"]["backward_seconds"] == 6.0
    assert result["median_seconds"]["optimizer_seconds"] == 9.0
    assert result["median_seconds"]["step_seconds"] == 18.0
    assert result["peak_allocated_gib"] == 4.0
    assert result["peak_reserved_gib"] == 5.0
    assert result["losses"] == pytest.approx([0.01, 0.02, 0.03, 0.04])


def test_missing_or_failed_steps_not_misreported_as_valid_benchmark():
    assert measured_steps({"status": "failed", "per_step": [_reading(1)]}, warmup=1, expected_steps=1) is None
    assert measured_steps({"status": "ok", "per_step": [_reading(1)]}, warmup=1, expected_steps=4) is None


def test_historical_m43_default_backend_is_unchanged():
    default = model_config()
    optional = model_config(sparse_backend="torch_sparse_triton")
    assert default.sparse_attention_backend == "torch_sparse"
    assert optional.sparse_attention_backend == "torch_sparse_triton"
    assert default.hidden_dim == optional.hidden_dim == 128
    assert default.num_layers == optional.num_layers == 10
    with pytest.raises(ValueError, match="unsupported"):
        model_config(sparse_backend="scatter")
