"""M41.6: selective NPZ and mmap cache preserve M37/M41.4 exact output."""
from __future__ import annotations

import json
from itertools import product

import h5py
import numpy as np
import pytest
import torch

from graph_attention.geometry.external_csr import (
    DiskCSR, build_hdf5_disk_csr,
)
from graph_attention.geometry.partition_csr import (
    build_partition_files, spatial_owners,
)
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT
from graph_attention.models.disk_streamed_dinat import build_disk_streamed_dinat_plan
from graph_attention.models.streamed_dinat import forward_streamed_dinat
from graph_attention.models.streamed_stage_profile import StreamedStageProfiler
from graph_attention.models.selective_disk_dinat import (
    SelectiveDiskPartitionCollection, MappedDiskPartitionCollection,
    validate_mmap_cache,
)
from scripts.m41_6_compile_metadata import compile_cache


def _irregular_shuffled_hex(tmp_path, *, include_self=False):
    side = 3
    n = side**3
    rng = np.random.default_rng(416)
    xyz = np.asarray(list(product(range(side), repeat=3)), dtype=np.float32)
    xyz += rng.normal(scale=0.08, size=xyz.shape).astype(np.float32)
    permutation = rng.permutation(n)
    inverse = np.argsort(permutation)
    cells = []
    for x, y, z in product(range(side-1), repeat=3):
        corners = ((x,y,z),(x+1,y,z),(x+1,y+1,z),(x,y+1,z),
                   (x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1))
        node_ids = [((a*side)+b)*side+c for a,b,c in corners]
        cells.append(inverse[node_ids]+1)
    mesh = tmp_path/"mesh.h5"
    with h5py.File(mesh, "w") as f:
        f.create_group("Connectivity").create_dataset(
            "hex->node", data=np.asarray(cells, dtype=np.uint64).reshape(-1)
        )
    base = tmp_path/"original"
    base.mkdir()
    coords = xyz[permutation].copy()
    np.save(base/"coords.npy", coords)
    csr_meta = build_hdf5_disk_csr(
        mesh, num_nodes=n, output_dir=base/"csr",
        bucket_nodes=7, chunk_cells=3,
    )
    owner = spatial_owners(coords, blocks=(2,2,2), chunk_nodes=4)
    graph = DiskCSR.open(base/"csr")
    part_meta = build_partition_files(
        graph, owner, output_dir=base/"partitions",
        chunk_targets=2, max_owned_nodes=30,
        include_self_attention=include_self,
    )
    report = {
        "status": "generic_topology_ready", "num_nodes": n,
        "num_partitions": part_meta["num_partitions"],
        "blocks": [2, 2, 2],
        "num_onehop_edges": csr_meta["directed_edges"],
        "num_partition_local_edges": sum(p["local_edges"] for p in part_meta["parts"]),
        "num_partition_twohop_edges": sum(p["dilated_edges"] for p in part_meta["parts"]),
        "coordinate_divisor": 1.0,
        "include_self_attention": include_self,
        "include_local_self_attention": False,
    }
    (base/"prepare.json").write_text(json.dumps(report))
    return base, torch.tensor(coords, dtype=torch.float32)


def _model(include_self=False):
    torch.manual_seed(741)
    model = AlternatingDilatedGeometricDiT(
        in_channels=5, out_channels=5,
        hidden_dim=32, num_heads=4, num_layers=4,
        spatial_dim=3, conditioning_channels=1,
        sparse_attention_backend="torch_sparse",
        include_self_attention=include_self,
    ).eval()
    with torch.no_grad():
        for layer in model.blocks:
            torch.nn.init.normal_(layer.adaLN_modulation[-1].weight, std=0.02)
            torch.nn.init.normal_(layer.adaLN_modulation[-1].bias, std=0.05)
        torch.nn.init.normal_(model.final_layer.projection.weight, std=0.08)
    return model


@pytest.mark.parametrize("include_self", [False, True])
def test_selective_and_mmap_exact_forward_parity(tmp_path, include_self):
    base, coords = _irregular_shuffled_hex(tmp_path, include_self=include_self)
    model = _model(include_self=include_self)
    saved_model = {k: v.clone() for k, v in model.state_dict().items()}
    directory = base/"partitions"
    cache = tmp_path/"compiled"
    built = compile_cache(prepared_dir=base, output_dir=cache)
    assert built["status"] == "complete"
    assert built["num_partitions"] > 1
    assert built["cache_bytes"] > 0
    assert validate_mmap_cache(
        cache, partition_dir=directory,
        expected_nodes=len(coords), expected_partitions=built["num_partitions"],
    )["source_partition_sha256"]
    torch.manual_seed(746)
    x = torch.randn(len(coords), 5)
    condition = torch.tensor([[400.0]], dtype=torch.float32)
    predictions = {}
    for mode in ("baseline", "selective", "mmap"):
        plan = build_disk_streamed_dinat_plan(
            model, coords=coords, partition_dir=directory,
            metadata_mode=mode,
            metadata_cache_dir=cache if mode == "mmap" else None,
        )
        if mode != "baseline":
            first_owner = plan.partitions.load_for_pass(0, "owned")
            assert first_owner.nodes.read_global is None
            assert first_owner.nodes.local is None
            assert first_owner.local is None
            first_local = plan.partitions.load_for_pass(0, "local")
            assert first_local.nodes.local is not None
            assert first_local.nodes.dilated is None
            assert first_local.dilated is None
            assert first_local.local.edge_order is None
            first_dilated = plan.partitions.load_for_pass(0, "dilated")
            assert first_dilated.nodes.dilated is not None
            assert first_dilated.nodes.local is None
            assert first_dilated.local is None
        profiler = StreamedStageProfiler() if mode == "selective" else None
        with torch.inference_mode():
            if mode == "mmap":
                part = plan.partitions.load_for_pass(0, "dilated")
                assert not part.dilated.pattern.is_inference()
                assert not part.dilated.pattern.crow_indices().is_inference()
                assert not part.dilated.pattern.col_indices().is_inference()
            result = forward_streamed_dinat(
                model, x, plan=plan, conditioning=condition,
                stage_profiler=profiler,
            )
        predictions[mode] = result.clone()
        if profiler is not None:
            report = profiler.report()
            assert report["metadata_detail_calls"]["metadata_npz_decode"] > 0
            assert report["metadata_detail_calls"]["metadata_csr_assembly"] > 0
            assert report["metadata_detail_calls"]["metadata_relative_displacement"] > 0
            assert report["stage_calls"]["gpu_attention_context"] == (
                len(model.blocks)*len(plan.partitions)
            )
    for mode in ("selective", "mmap"):
        torch.testing.assert_close(
            predictions[mode], predictions["baseline"],
            atol=2e-4, rtol=2e-4,
        )
    for name, orig in saved_model.items():
        assert torch.equal(orig, model.state_dict()[name])


def test_compiled_cache_rejects_stale_graph_and_existing_dir(tmp_path):
    base, coords = _irregular_shuffled_hex(tmp_path)
    cache = tmp_path/"compiled"
    compile_cache(prepared_dir=base, output_dir=cache)
    with pytest.raises(FileExistsError, match="overwrite"):
        compile_cache(prepared_dir=base, output_dir=cache)
    original = json.loads((base/"partitions"/"partitions.json").read_text())
    modified = {**original, "status": "modified"}
    (base/"partitions"/"partitions.json").write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="stale"):
        validate_mmap_cache(
            cache, partition_dir=base/"partitions",
            expected_nodes=len(coords), expected_partitions=original["num_partitions"],
        )


def test_metadata_mode_validation(tmp_path):
    base, coords = _irregular_shuffled_hex(tmp_path)
    model = _model()
    with pytest.raises(ValueError, match="metadata_mode"):
        build_disk_streamed_dinat_plan(
            model, coords=coords, partition_dir=base/"partitions",
            metadata_mode="invalid",
        )
    with pytest.raises(ValueError, match="preload"):
        build_disk_streamed_dinat_plan(
            model, coords=coords, partition_dir=base/"partitions",
            metadata_mode="selective", preload=True,
        )
    with pytest.raises(ValueError, match="compiled"):
        build_disk_streamed_dinat_plan(
            model, coords=coords, partition_dir=base/"partitions",
            metadata_mode="mmap",
        )
