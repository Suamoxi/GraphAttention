"""M41.5 paired 32/64 crossover and profiling report preflight tests."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark_m41_5_paired_profile import (
    LAYOUT_PAIR, _prepared_layouts, _profile_brief, paired_order,
)


def test_paired_order_is_balanced_and_interleaved():
    expected = [
        (1, 32), (1, 64),
        (2, 64), (2, 32),
        (3, 32), (3, 64),
        (4, 64), (4, 32),
        (5, 32), (5, 64),
        (6, 64), (6, 32),
    ]
    assert LAYOUT_PAIR == (32, 64)
    assert list(paired_order(6)) == expected
    assert len(paired_order(6)) == 12
    with pytest.raises(ValueError, match="at least two"):
        paired_order(1)


def _test_prepared(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    np.save(base / "coords.npy", np.ones((4, 3), dtype=np.float32))
    (base / "csr").mkdir()
    (base / "partitions").mkdir()
    (base / "partitions" / "partitions.json").write_text("{}")
    basic = {
        "status": "generic_topology_ready",
        "num_nodes": 4,
        "num_onehop_edges": 6,
        "num_partition_local_edges": 6,
        "num_partition_twohop_edges": 8,
        "coordinate_divisor": 1.0,
        "include_self_attention": False,
        "include_local_self_attention": False,
    }
    (base / "prepare.json").write_text(json.dumps({
        **basic, "num_partitions": 64, "blocks": [4, 4, 4],
    }))
    root = tmp_path / "layouts"
    layout = root / "layout_032"
    layout.mkdir(parents=True)
    os.symlink(base / "csr", layout / "csr", target_is_directory=True)
    os.symlink(base / "coords.npy", layout / "coords.npy")
    (layout / "partitions").mkdir()
    (layout / "partitions" / "partitions.json").write_text("{}")
    (layout / "prepare.json").write_text(json.dumps({
        **basic, "num_partitions": 32, "blocks": [4, 4, 2],
    }))
    return base, root, layout


def test_paired_preflight_reuses_same_global_csr_and_coords(tmp_path):
    base, root, layout = _test_prepared(tmp_path)
    report, paths = _prepared_layouts(base, root)
    assert report["num_nodes"] == 4
    assert paths[32] == layout
    assert paths[64] == base
    mutated = json.loads((layout / "prepare.json").read_text())
    mutated["num_partition_twohop_edges"] += 1
    (layout / "prepare.json").write_text(json.dumps(mutated))
    with pytest.raises(ValueError, match="immutable graph field"):
        _prepared_layouts(base, root)


def test_profile_brief_requires_valid_stage_duration_and_layers():
    profile = {
        "timing_method": "synchronized_wall_seconds_not_cuda_kernel_time",
        "profiled_forward_wall_seconds": 2.0,
        "stage_seconds": {
            "partition_metadata_load": 0.6,
            "gpu_attention_context": 0.5,
            "d2h_owned_readback": 0.2,
        },
        "stage_calls": {"partition_metadata_load": 30},
        "layer_wall_seconds": {"0": 1.5},
        "layers_stage_seconds": {"0": {"gpu_attention_context": 0.5}},
        "synchronization_calls": 70,
    }
    result = {
        "forward_seconds": 2.04,
        "stage_profile": profile,
        "host_memory_after_forward": {"rss_peak_gib": 3.5},
        "gpu_peak_allocated_gib": 1.5,
        "gpu_peak_reserved_gib": 2.2,
    }
    summary = _profile_brief(result)
    assert summary["stage_share_of_profiled_forward_pct"]["gpu_attention_context"] == 25.0
    assert summary["synchronization_calls"] == 70
    profile["stage_seconds"]["gpu_attention_context"] = 20.0
    with pytest.raises(ValueError, match="durations"):
        _profile_brief(result)
