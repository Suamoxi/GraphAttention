"""M41.3 benchmark pairing, provenance, and failure tests (CPU-only)."""

from __future__ import annotations

import json

import pytest

from scripts.summarize_m41_3 import compare


def _report(executor, *, sha="same", success=True, blocks=(2, 2, 2)):
    return {
        "executor": executor,
        "benchmark": (
            "m41_3_host_streamed_halo_v1"
            if executor == "streamed" else "m41_2_owned_query_halo_v1"
        ),
        "sample_id": "solut_hit_00007500",
        "num_nodes": 35937,
        "backend": "torch_sparse",
        "precision": "float32; no autocast; TF32 disabled",
        "artifact_sha256": {"best.pt": sha},
        "blocks": list(blocks),
        "num_partitions": 8,
        "nodewise_work_ratio": 1.375,
        "max_partition_read_nodes": 6000,
        "hidden_storage": "cpu_double_buffer" if executor == "streamed" else "gpu",
        "host_hidden_two_buffer_gib": 0.035,
        "original": {"median_ms": 31},
        "partitioned": {
            "median_ms": 80 if executor == "streamed" else 60,
            "peak_allocated_gib": 0.3 if executor == "streamed" else 0.4,
        },
        "generation": {
            "steps": 50, "seed": 5678,
            "partitioned": {"median_ms": 8000 if executor == "streamed" else 6000},
        },
        "errors": {
            "layers": [{"max_abs": 2.5e-5}],
            "output": {"max_abs": 4e-5},
            "heun": {"max_abs": 1e-6},
        },
        "passed": success,
    }


def _write(tmp_path, label, report, *, suffix="2x2x2"):
    (tmp_path / f"{label}_blocks_{suffix}.json").write_text(json.dumps(report))


def test_summary_paired_checkpoint(tmp_path):
    _write(tmp_path, "owned", _report("owned_query"))
    _write(tmp_path, "streamed", _report("streamed"))
    rows = compare(tmp_path)
    assert len(rows) == 1
    assert rows[0]["passed"]
    assert rows[0]["m41_3_vs_m41_2_speedup"] == pytest.approx(0.75)
    assert rows[0]["m41_3_vs_m37_forward_ratio"] == pytest.approx(80 / 31)
    assert rows[0]["m41_3_host_hidden_two_buffer_gib"] == pytest.approx(0.035)


def test_summary_refuses_missing_or_mismatched_pairs(tmp_path):
    _write(tmp_path, "owned", _report("owned_query"))
    with pytest.raises(ValueError, match="paired"):
        compare(tmp_path)
    _write(tmp_path, "streamed", _report("streamed", sha="different"))
    with pytest.raises(ValueError, match="mismatch"):
        compare(tmp_path)


def test_summary_failed_numerical_gate_stays_failed(tmp_path):
    _write(tmp_path, "owned", _report("owned_query"))
    bad = _report("streamed", success=False)
    bad["errors"]["layers"][0]["max_abs"] = None
    _write(tmp_path, "streamed", bad)
    rows = compare(tmp_path)
    assert not rows[0]["passed"]
    assert rows[0]["m41_3_max_layer_abs"] == float("inf")


def test_summary_rejects_nonstreamed_candidate(tmp_path):
    _write(tmp_path, "owned", _report("owned_query"))
    incorrect = _report("streamed")
    incorrect["hidden_storage"] = "gpu"
    _write(tmp_path, "streamed", incorrect)
    with pytest.raises(ValueError, match="host-streaming"):
        compare(tmp_path)
