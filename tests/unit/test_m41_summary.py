"""M41.1 multi-layout report summarization, no GPU or M37 artifacts needed."""

import json

import pytest

from scripts.summarize_m41_partitioned import summarize


def report(blocks, partitions):
    return {
        "blocks": blocks,
        "num_partitions": partitions,
        "num_nodes": 35937,
        "backend": "torch_sparse",
        "precision": "float32; no autocast; TF32 disabled",
        "sample_id": "heldout_sample",
        "artifact_sha256": {"best.pt": "a", "standardizers.pt": "b"},
        "generation": {
            "steps": 50,
            "seed": 5678,
            "original": {"median_ms": 1000.0},
            "partitioned": {"median_ms": 1200.0},
        },
        "partitions": [{"halo": 10} for _ in range(partitions)],
        "nodewise_work_ratio": 1.1,
        "plan_build_seconds": 1.5,
        "plan_persistent_allocated_gib": 0.25,
        "original": {
            "median_ms": 12.0, "peak_allocated_gib": 3.0,
            "incremental_peak_allocated_gib": 0.5,
        },
        "partitioned": {
            "median_ms": 14.0, "peak_allocated_gib": 3.5,
            "incremental_peak_allocated_gib": 0.7,
        },
        "errors": {
            "layers": [
                {"max_abs": 1e-6, "relative_l2": 2e-7},
                {"max_abs": 4e-6, "relative_l2": 1e-7},
            ],
            "output": {"max_abs": 5e-6},
            "heun": {"max_abs": 7e-6},
        },
        "passed": True,
        "evidence": "LOCAL_BENCHMARK",
    }


def test_summary_sorts_layouts_and_computes_overhead(tmp_path):
    first = report([2, 2, 2], 8)
    second = report([1, 1, 1], 1)
    (tmp_path / "blocks_2x2x2.json").write_text(json.dumps(first))
    (tmp_path / "blocks_1x1x1.json").write_text(json.dumps(second))
    rows = summarize(tmp_path)
    assert [r["blocks"] for r in rows] == ["1x1x1", "2x2x2"]
    assert rows[0]["max_layer_abs_error"] == 4e-6
    assert rows[1]["total_halo_copies"] == 80
    assert rows[1]["forward_overhead_pct"] == pytest.approx(100 * (14 / 12 - 1))
    assert rows[1]["heun_overhead_pct"] == pytest.approx(20)
    assert all(row["passed"] for row in rows)


def test_summary_does_not_mix_checkpoints_or_duplicate_layouts(tmp_path):
    first = report([1, 1, 1], 1)
    second = report([2, 1, 1], 2)
    second["artifact_sha256"]["best.pt"] = "different_checkpoint"
    (tmp_path / "blocks_1x1x1.json").write_text(json.dumps(first))
    (tmp_path / "blocks_2x1x1.json").write_text(json.dumps(second))
    with pytest.raises(ValueError, match="different checkpoints"):
        summarize(tmp_path)

    second = report([1, 1, 1], 1)
    (tmp_path / "blocks_2x1x1.json").write_text(json.dumps(second))
    with pytest.raises(ValueError, match="duplicate"):
        summarize(tmp_path)


def test_summary_handles_nonfinite_numerical_failure(tmp_path):
    failing = report([1, 1, 1], 1)
    failing["passed"] = False
    failing["errors"]["layers"] = [
        {"max_abs": None, "relative_l2": None},
    ]
    (tmp_path / "blocks_1x1x1.json").write_text(json.dumps(failing))
    rows = summarize(tmp_path)
    assert rows[0]["passed"] is False
    assert rows[0]["max_layer_abs_error"] != rows[0]["max_layer_abs_error"]  # NaN
