"""M41.4 partition-count sweep using an existing disk-backed one-hop CSR.

No HDF5 mesh reload, global edge rebuild or Cartesian shortcut is used.
The base 64-partition preparation remains immutable. Variants hold only new
ownership labels and owned-query one/two-hop topology; coords/CSR are symlinked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from graph_attention.geometry.external_csr import DiskCSR
from graph_attention.geometry.partition_csr import build_partition_files, spatial_owners


LAYOUTS: dict[int, tuple[int, int, int]] = {
    16: (4, 2, 2),
    32: (4, 4, 2),
    64: (4, 4, 4),
    128: (8, 4, 4),
}
MAX_OWNED: dict[int, int] = {
    16: 1_200_000,
    32: 650_000,
    64: 400_000,
    128: 180_000,
}


def layout_path(layout_root: Path, number: int) -> Path:
    if number not in LAYOUTS:
        raise ValueError(f"unsupported partition layout {number}")
    return Path(layout_root) / f"layout_{number:03d}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repartition(
    base_dir: Path,
    output_dir: Path,
    *,
    blocks: tuple[int, int, int],
    expected_partitions: int,
    max_owned_nodes: int,
    chunk_targets: int = 8192,
) -> dict:
    """Create an independent prepared layout, retaining only one-hop CSR once.

    A destination must not exist, including a partially failed destination.
    This makes accidental benchmark contamination and silent overwrites fail.
    """
    start = time.perf_counter()
    base = Path(base_dir).resolve(strict=True)
    dest = Path(output_dir).expanduser().resolve()
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite prepared layout: {dest}")
    if len(blocks) != 3 or any(int(v) != v or v < 1 for v in blocks):
        raise ValueError("blocks must be three positive integers")
    if expected_partitions != int(np.prod(blocks)):
        raise ValueError("expected_partitions and requested block count disagree")
    base_report = json.loads((base / "prepare.json").read_text())
    if base_report.get("status") != "generic_topology_ready":
        raise ValueError("base graph was not fully prepared")
    n = int(base_report["num_nodes"])
    raw_coords = np.load(base / "coords.npy", mmap_mode="r")
    if raw_coords.shape != (n, 3) or raw_coords.dtype != np.float32:
        raise ValueError("base model-facing coordinates must be float32 [N,3]")
    graph = DiskCSR.open(base / "csr")
    if graph.num_nodes != n or graph.directed_edges != base_report["num_onehop_edges"]:
        raise ValueError("base CSR and manifest disagree")
    metadata_hash = _sha256(base / "csr" / "csr.json")
    dest.mkdir(parents=True)
    # Keep the two large immutable files shared; do not copy the mesh or CSR.
    os.symlink(base / "coords.npy", dest / "coords.npy")
    os.symlink(base / "csr", dest / "csr", target_is_directory=True)
    t_owners = time.perf_counter()
    owners = spatial_owners(raw_coords, blocks=blocks)
    occupied = int(owners.max()) + 1
    if occupied != expected_partitions:
        raise ValueError(
            f"requested {expected_partitions} blocks but only {occupied} are occupied"
        )
    np.save(dest / "owners.npy", owners)
    owner_seconds = time.perf_counter() - t_owners
    print(f"[repartition] {occupied} blocks; owner mapping {owner_seconds:.2f}s", flush=True)
    t_halo = time.perf_counter()
    part = build_partition_files(
        graph,
        owners,
        output_dir=dest / "partitions",
        chunk_targets=chunk_targets,
        max_owned_nodes=max_owned_nodes,
        include_self_attention=bool(base_report["include_self_attention"]),
        include_local_self_attention=bool(base_report["include_local_self_attention"]),
    )
    halo_seconds = time.perf_counter() - t_halo
    local = sum(p["local_edges"] for p in part["parts"])
    twohop = sum(p["dilated_edges"] for p in part["parts"])
    if local != base_report["num_partition_local_edges"]:
        raise ValueError("local-edge conservation failed against base graph")
    if twohop != base_report["num_partition_twohop_edges"]:
        raise ValueError("two-hop edge conservation failed against base graph")
    read_total = sum(p["read_nodes"] for p in part["parts"])
    halo_total = sum(p["halo_nodes"] for p in part["parts"])
    report = {
        **base_report,
        "status": "generic_topology_ready",
        "blocks": list(blocks),
        "num_partitions": occupied,
        "max_partition_owned_nodes": max(p["owned_nodes"] for p in part["parts"]),
        "max_partition_read_nodes": max(p["read_nodes"] for p in part["parts"]),
        "num_partition_local_edges": local,
        "num_partition_twohop_edges": twohop,
        "total_partition_read_nodes": read_total,
        "total_halo_nodes_across_partitions": halo_total,
        "halo_read_amplification": read_total / n,
        "reused_base_prepared_dir": str(base),
        "reused_csr_metadata_sha256": metadata_hash,
        "coordinate_ownership_seconds": owner_seconds,
        "partition_topology_seconds": halo_seconds,
        "repartition_seconds": time.perf_counter() - start,
        "max_owned_nodes_guard": max_owned_nodes,
        "chunk_targets": chunk_targets,
        "preparation_method": "reuse_onehop_csr_recompute_owners_and_partition_halos",
    }
    # Atomic, last: infer only considers directories with a complete manifest.
    tmp = dest / "prepare.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, dest / "prepare.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--layout-root", type=Path, required=True)
    p.add_argument("--layout", type=int, choices=sorted(LAYOUTS), required=True)
    p.add_argument("--chunk-targets", type=int, default=8192)
    args = p.parse_args()
    if args.layout == 64:
        raise ValueError("reuse the already prepared 64-partition base; do not regenerate it")
    repartition(
        args.base_dir,
        layout_path(args.layout_root, args.layout),
        blocks=LAYOUTS[args.layout],
        expected_partitions=args.layout,
        max_owned_nodes=MAX_OWNED[args.layout],
        chunk_targets=args.chunk_targets,
    )


if __name__ == "__main__":
    main()
