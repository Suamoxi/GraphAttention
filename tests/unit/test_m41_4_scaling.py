"""M41.4 reproducibility, reused one-hop graph, and numerical summary tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from graph_attention.geometry.external_csr import DiskCSR, build_disk_csr
from graph_attention.geometry.partition_csr import build_partition_files, spatial_owners
from scripts.m41_4_repartition import repartition, LAYOUTS, MAX_OWNED, layout_path
from scripts.benchmark_m41_4_scalability import _plan, _validate_report
from scripts.summarize_m41_4_scalability import _output_difference, _stats


def _small_base(tmp_path):
    """Build a non-Cartesian test graph; don't assume regular connectivity."""
    base = tmp_path / "base"
    base.mkdir()
    n = 70
    rng = np.random.default_rng(9238)
    xyz = rng.uniform(low=[-0.5, 2.0, 1.0], high=[2.0, 4.0, 3.0], size=(n, 3)).astype(np.float32)
    np.save(base / "coords.npy", xyz)
    # A directed cycle plus reverse links. Independent of coordinates.
    edges = [(i, (i+1) % n) for i in range(n)] + [(i, (i+3) % n) for i in range(n)]
    edges = edges + [(b, a) for a, b in edges]
    source, target = zip(*edges)
    built = build_disk_csr(
        [(np.asarray(source), np.asarray(target))],
        num_nodes=n, output_dir=base/"csr", bucket_nodes=11,
    )
    csr = DiskCSR.open(base/"csr")
    owners = spatial_owners(xyz, blocks=(2, 1, 1))
    original = build_partition_files(csr, owners, output_dir=base/"partitions", max_owned_nodes=n)
    prep = {
        "status": "generic_topology_ready", "num_nodes": n,
        "num_partitions": original["num_partitions"], "blocks": [2, 1, 1],
        "num_onehop_edges": built["directed_edges"],
        "num_partition_local_edges": sum(p["local_edges"] for p in original["parts"]),
        "num_partition_twohop_edges": sum(p["dilated_edges"] for p in original["parts"]),
        "include_self_attention": False, "include_local_self_attention": False,
        "coordinate_divisor": 1.0,
    }
    (base/"prepare.json").write_text(json.dumps(prep))
    return base


def test_repartition_reuses_csr_and_preserves_exact_edge_totals(tmp_path):
    base = _small_base(tmp_path)
    target = tmp_path/"new_layout"
    summary = repartition(
        base, target, blocks=(2, 2, 1), expected_partitions=4,
        max_owned_nodes=70, chunk_targets=2,
    )
    assert summary["num_partitions"] == 4
    assert summary["num_onehop_edges"] == json.loads((base/"prepare.json").read_text())["num_onehop_edges"]
    assert summary["num_partition_twohop_edges"] == json.loads((base/"prepare.json").read_text())["num_partition_twohop_edges"]
    assert (target/"csr").is_symlink() and (target/"coords.npy").is_symlink()
    assert (target/"csr").resolve() == (base/"csr").resolve()
    assert summary["halo_read_amplification"] >= 1.0
    original = json.loads((base/"prepare.json").read_text())
    assert original["blocks"] == [2, 1, 1]
    assert (target/"partitions"/"partitions.json").is_file()
    with pytest.raises(FileExistsError):
        repartition(base, target, blocks=(2, 2, 1), expected_partitions=4,
                    max_owned_nodes=70)


def test_partition_count_contract(tmp_path):
    base = _small_base(tmp_path)
    with pytest.raises(ValueError, match="disagree"):
        repartition(base, tmp_path/"invalid", blocks=(2,2,1),
                    expected_partitions=5, max_owned_nodes=70)
    assert LAYOUTS[16] == (4,2,2)
    assert LAYOUTS[32] == (4,4,2)
    assert LAYOUTS[64] == (4,4,4)
    assert LAYOUTS[128] == (8,4,4)
    assert MAX_OWNED[16] > MAX_OWNED[64]
    assert layout_path(tmp_path, 128).name == "layout_128"


def test_benchmark_balanced_order_and_report_identity():
    assert _plan(3, ("lazy", "preload")) == [
        (1,"lazy"),(1,"preload"),(2,"preload"),(2,"lazy"),
        (3,"lazy"),(3,"preload"),
    ]
    with pytest.raises(ValueError, match="unique"):
        _plan(2, ("lazy", "lazy"))
    valid = {
        "status": "full_highres_forward_passed", "nodes": 70,
        "output_shape": [70,5], "partition_count": 4,
        "preload_partitions": False, "seed": 5678,
        "forward_seconds": 1.2, "gpu_peak_allocated_gib": 0.03,
        "host_memory_after_forward": {"rss_peak_gib": 0.4},
        "checkpoint_sha256": "abc",
    }
    _validate_report(valid, num_nodes=70, partitions=4, preload=False, seed=5678)
    with pytest.raises(ValueError, match="configuration"):
        _validate_report(valid, num_nodes=70, partitions=4, preload=True, seed=5678)


def test_full_output_comparison_reads_in_chunks(tmp_path):
    original = np.arange(301*5, dtype=np.float32).reshape(301, 5)/100.0
    np.save(tmp_path/"a.npy", original)
    same = original + 1e-6
    np.save(tmp_path/"b.npy", same)
    result = _output_difference(tmp_path/"a.npy", tmp_path/"b.npy", atol=2e-4, rtol=2e-4, chunk_rows=7)
    assert result["passed"] and result["violations"] == 0
    modified = original.copy()
    modified[300, 4] += 1.0
    np.save(tmp_path/"c.npy", modified)
    result = _output_difference(tmp_path/"a.npy", tmp_path/"c.npy", atol=2e-4, rtol=2e-4, chunk_rows=7)
    assert not result["passed"] and result["violations"] == 1
    assert result["max_abs"] == pytest.approx(1.0)
    assert _stats([1.0, 3.0])["median"] == 2.0
