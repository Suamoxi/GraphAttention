"""Compile immutable M41.4 NPZ partitions to M41.6 mmap-ready CSR/geometry.

The source one-hop CSR and NPZ partitions are left unchanged. This optional
ONE-TIME, per-layout conversion stores ordinary .npy arrays that can be
mapped on-demand via mmap_mode='c', avoiding repeated source NPZ extraction
and displacement reconstruction for the 10-layer streamed model.

One partition at a time is materialized, with no global two-hop graph.
The complete cache manifest is written atomically ONLY when every
partition file is prepared and validated. No silent overwrites.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from graph_attention.models.disk_streamed_dinat import DiskPartitionCollection
from graph_attention.models.selective_disk_dinat import (
    source_manifest_sha256, validate_mmap_cache,
)


def compile_cache(*, prepared_dir: Path, output_dir: Path) -> dict:
    start = time.perf_counter()
    prepared = Path(prepared_dir).resolve(strict=True)
    part_dir = prepared / "partitions"
    prepare = json.loads((prepared / "prepare.json").read_text())
    parts = json.loads((part_dir / "partitions.json").read_text())
    if (prepare.get("status") != "generic_topology_ready"
        or parts.get("status") != "complete"
        or prepare["num_nodes"] != parts["num_nodes"]
        or prepare["num_partitions"] != parts["num_partitions"]):
        raise ValueError("invalid or inconsistent original partition manifests")
    dest = Path(output_dir).expanduser().resolve()
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite cache: {dest}")
    dest.mkdir(parents=True)
    # Lazy memmap of the original physical coords; the compiled displacement
    # uses M37's original exact source-minus-target convention.
    coordinates = np.load(prepared / "coords.npy", mmap_mode="c")
    if coordinates.shape != (prepare["num_nodes"], 3) or coordinates.dtype != np.float32:
        raise ValueError("prepared coordinates must be [N,3] float32")
    collection = DiskPartitionCollection(
        part_dir, torch.from_numpy(coordinates), prepare["num_partitions"]
    )

    def save(folder: Path, name: str, data: torch.Tensor) -> None:
        filename = folder / f"{name}.npy"
        np.save(filename, data.detach().numpy(), allow_pickle=False)

    written_bytes = 0
    for index in range(len(collection)):
        t0 = time.perf_counter()
        partition = collection[index]
        target = dest / f"part_{index:05d}"
        target.mkdir()
        owned = partition.nodes.owned_global
        read = partition.nodes.read_global
        save(target, "owned", owned)
        save(target, "read", read)
        save(target, "owned_local", partition.nodes.owned_local)
        for kind in ("local", "dilated"):
            attn = getattr(partition.nodes, kind)
            topology = getattr(partition, kind)
            if (topology.edge_order is None
                or not torch.equal(
                    topology.edge_order,
                    torch.arange(topology.col_indices.numel(), dtype=torch.long),
                )):
                raise ValueError("cache compiler requires sorted, unpermuted source metadata")
            save(target, f"{kind}_crow", topology.pattern.crow_indices())
            save(target, f"{kind}_row", topology.row_indices)
            save(target, f"{kind}_col", topology.col_indices)
            save(target, f"{kind}_displacement", attn.displacement)
        written_bytes += sum(file.stat().st_size for file in target.glob("*.npy"))
        print(
            f"[M41.6 compile] {index+1}/{len(collection)} parts, "
            f"owned={len(owned):,}, read={len(read):,}; "
            f"part_seconds={time.perf_counter()-t0:.2f}", flush=True,
        )
        del partition
    manifest = {
        "status": "complete",
        "format_version": 1,
        "num_nodes": prepare["num_nodes"],
        "num_partitions": prepare["num_partitions"],
        "source_partition_directory": str(part_dir.resolve()),
        "source_partition_sha256": source_manifest_sha256(part_dir),
        "construction": "one_partition_at_a_time_no_global_2hop",
        "edge_order": "identity_sorted_by_target_source",
        "displacement": "unmodified_physical_source_minus_target_float32",
        "cache_bytes": written_bytes,
        "cache_gib": written_bytes / 1024**3,
        "compile_seconds": time.perf_counter() - start,
    }
    temporary = dest / "cache.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, dest / "cache.json")
    validate_mmap_cache(
        dest, partition_dir=part_dir, expected_nodes=prepare["num_nodes"],
        expected_partitions=prepare["num_partitions"],
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    compile_cache(prepared_dir=args.prepared_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
