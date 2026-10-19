"""Small, real-layout HDF5 B1 preflight and training-step contracts (CPU)."""
from __future__ import annotations

from itertools import product
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from graph_attention.data import (
    AVBP_FIELD_CATALOG, AVBPHDF5Dataset, AVBPSampleSpec,
)
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m43_b1_memory_gate import (
    choose_snapshot, inspect_case, parse_case, run_memory_gate,
)
from scripts.m43_b1_training_step import (
    _model_forward, model_config, normalized_probe_state,
)


def _make_case(root: Path, *, case: str = "Re54/Ma0.3", side: int = 3) -> Path:
    folder = root/case
    mesh = folder/"MESH"/"mesh.mesh.h5"
    mesh.parent.mkdir(parents=True)
    coords = np.array(
        list(product(range(side), repeat=3)), dtype=np.float64,
    )
    cells = []
    for i, j, k in product(range(side - 1), repeat=3):
        corners = (
            (i,j,k), (i+1,j,k), (i+1,j+1,k), (i,j+1,k),
            (i,j,k+1), (i+1,j,k+1), (i+1,j+1,k+1), (i,j+1,k+1),
        )
        cells.append([((x*side+y)*side+z)+1 for x,y,z in corners])
    with h5py.File(mesh, "w") as file:
        axes = file.create_group("Coordinates")
        for index, axis in enumerate("xyz"):
            axes.create_dataset(axis, data=coords[:, index])
        file.create_group("Connectivity").create_dataset(
            "hex->node", data=np.array(cells, dtype=np.uint64).reshape(-1)
        )
    solut = folder/"SOLUT"
    solut.mkdir()
    for idx in range(3):
        with h5py.File(solut / f"solut_hit_{idx:08d}.h5", "w") as file:
            grp = file.create_group("GaseousPhase")
            t = np.arange(side**3, dtype=np.float64)
            for scale, name in enumerate(("rho","rhou","rhov","rhow","rhoE"), 1):
                grp.create_dataset(name, data=(scale * 0.3 + t * scale * 0.1 + idx))
    return folder


def test_parse_case_and_incorrect_resolution():
    assert parse_case("Re54/Ma0.3:65") == ("Re54/Ma0.3", 65)
    assert parse_case("Re070/Ma0.15:193") == ("Re070/Ma0.15", 193)
    with pytest.raises(ValueError, match="expected"):
        parse_case("../../tmp:65")
    with pytest.raises(ValueError, match="mesh side"):
        parse_case("Re54/Ma0.3:2")


def test_b1_hdf5_preflight_and_snapshot_selection(tmp_path):
    directory = _make_case(tmp_path)
    meta = inspect_case(tmp_path, "Re54/Ma0.3:3")
    assert meta["nodes"] == 27
    assert meta["hex_cells"] == 8
    assert meta["snapshot_count"] == 3
    assert meta["snapshot_file"].endswith("solut_hit_00000001.h5")
    assert set(meta["fields"]) == {"rho","rhou","rhov","rhow","rhoE"}
    with pytest.raises(ValueError, match="expected 64 nodes"):
        inspect_case(tmp_path, "Re54/Ma0.3:4")
    second = directory/"RUN"/"SOLUT"
    second.mkdir(parents=True)
    (second/"duplicate.h5").write_bytes(b"not-read")
    with pytest.raises(ValueError, match="exactly one nonempty SOLUT"):
        choose_snapshot(directory)


def test_m43_b1_preflight_creates_no_training_or_gpu_artifacts(tmp_path):
    _make_case(tmp_path/"data")
    report = run_memory_gate(
        root=tmp_path/"data", output_dir=tmp_path/"results",
        specs=("Re54/Ma0.3:3",), preflight_only=True,
    )
    assert report["status"] == "preflight_complete"
    assert not (tmp_path/"results"/"runs").exists()
    assert not (tmp_path/"results"/".active.lock").exists()
    meta = json.loads((tmp_path/"results"/"preflight.json").read_text())
    assert meta["cases"][0]["resolution"] == 3


def test_small_real_hex_graph_fm_optimizer_step_without_training_changes(tmp_path):
    directory = _make_case(tmp_path)
    snapshot, _ = choose_snapshot(directory)
    dataset = AVBPHDF5Dataset(samples=[
        AVBPSampleSpec("probe", snapshot, "tiny-hex", directory/"MESH"/"mesh.mesh.h5")
    ])
    task = FlowMatchingTask(
        state_fields=("rho","rhou","rhov","rhow","rhoE"),
        physical_nondimensionalization=False, time_embedding_scale=1000.0,
    )
    collator = GraphTaskCollator(
        task, AVBP_FIELD_CATALOG,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    packed = collator([dataset[0]])
    prepared = normalized_probe_state(
        task_batch_to_device(
            packed, device=torch.device("cpu"), dtype=torch.float32,
        )
    )
    assert prepared.inputs.shape == (27,5)
    assert prepared.edge_index.shape[1] > 0
    assert prepared.attention_edge_indices["dilated"].shape[1] > 0
    assert prepared.conditioning.shape[1] == 0
    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(1042)
    problem = task.make_training_problem(prepared, generator=generator)
    model_cfg = model_config(hidden_dim=16, num_layers=2, num_heads=2)
    model, _ = instantiate_controlled_model(model_cfg, problem, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output = _model_forward(model, problem)
    loss = task.training_loss(output, problem).mean
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss.detach())
    assert output.shape == (27,5)
