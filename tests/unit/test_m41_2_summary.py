"""M41.2 report pairing and provenance checks (CPU-only)."""

import json

import pytest

from scripts.summarize_m41_2 import compare


def _report(executor, blocks=(2, 2, 2), *, sha="shared"):
    return {
        "executor": executor,
        "sample_id": "solut_hit_00007500",
        "num_nodes": 35937,
        "backend": "torch_sparse",
        "precision": "float32; no autocast; TF32 disabled",
        "artifact_sha256": {"best.pt": sha},
        "blocks": list(blocks),
        "num_partitions": 8,
        "nodewise_work_ratio": 1.375,
        "plan_persistent_allocated_gib": 0.05,
        "original": {"median_ms": 30},
        "partitioned": {
            "median_ms": 100 if executor == "layerwise_full" else 50,
            "peak_allocated_gib": 0.3,
        },
        "generation": {
            "steps": 50,
            "seed": 5678,
            "partitioned": {"median_ms": 10000 if executor == "layerwise_full" else 5000},
        },
        "errors": {
            "output": {"max_abs": 1e-5},
            "layers": [{"max_abs": 2e-5}],
            "heun": {"max_abs": 1e-6},
        },
        "passed": True,
    }


def _save(directory, prefix, report):
    (directory / f"{prefix}_blocks_2x2x2.json").write_text(json.dumps(report))


def test_paired_summary(tmp_path):
    _save(tmp_path, "full", _report("layerwise_full"))
    _save(tmp_path, "owned", _report("owned_query"))
    rows = compare(tmp_path)
    assert len(rows) == 1
    assert rows[0]["passed"]
    assert rows[0]["owned_vs_full_forward_speedup"] == pytest.approx(2)
    assert rows[0]["owned_vs_full_heun_speedup"] == pytest.approx(2)
    assert rows[0]["m41_2_vs_m37_forward_ratio"] == pytest.approx(50 / 30)


def test_refuses_different_weights_and_missing_layout(tmp_path):
    _save(tmp_path, "full", _report("layerwise_full"))
    with pytest.raises(ValueError, match="paired"):
        compare(tmp_path)
    _save(tmp_path, "owned", _report("owned_query", sha="other"))
    with pytest.raises(ValueError, match="contract mismatch"):
        compare(tmp_path)


def test_failed_candidate_is_not_marked_passed(tmp_path):
    original = _report("layerwise_full")
    optimized = _report("owned_query")
    optimized["passed"] = False
    optimized["errors"]["layers"][0]["max_abs"] = None
    _save(tmp_path, "full", original)
    _save(tmp_path, "owned", optimized)
    row = compare(tmp_path)[0]
    assert not row["passed"]
    assert row["m41_2_max_layer_abs"] == float("inf")
