"""M43 Experiment B six-case scientific contracts, with synthetic HDF5 only."""
from __future__ import annotations

from itertools import product
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from graph_attention.data.multicase_avbp import (
    AVBPMultiCaseHITDataset, DEFAULT_M43_CASES, HITCaseSpec,
    reference_filename,
)
from graph_attention.data.multicase_split import (
    balanced_case_interleave, temporal_case_split,
)
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training import fit_train_standardizers
from graph_attention.training.data_pipeline import GraphTaskCollator
from scripts.train_m43_b_multires import (
    _indices_by_case, _references_signature, _smoke_split,
    evaluate, inspect_inventory, train_epoch,
)


def _fixture_archive(tmp_path: Path):
    root = tmp_path / "archive"
    refdir = tmp_path / "refs"
    refdir.mkdir()
    chosen = tuple(
        HITCaseSpec(spec.case_id, 3 if spec.side == 65 else 4)
        for spec in DEFAULT_M43_CASES
    )
    for ordinal, spec in enumerate(chosen):
        directory = root / spec.case_id
        mesh_path = directory / "MESH" / "mesh.mesh.h5"
        mesh_path.parent.mkdir(parents=True)
        coords = np.array(list(product(range(spec.side), repeat=3)), dtype=np.float64)
        cells = []
        for x, y, z in product(range(spec.side-1), repeat=3):
            corners = [
                (x,y,z), (x+1,y,z), (x+1,y+1,z), (x,y+1,z),
                (x,y,z+1), (x+1,y,z+1), (x+1,y+1,z+1), (x,y+1,z+1),
            ]
            cells.extend([
                ((i*spec.side + j)*spec.side + k)+1 for i,j,k in corners
            ])
        with h5py.File(mesh_path, "w") as mesh:
            cg = mesh.create_group("Coordinates")
            for i,axis in enumerate("xyz"):
                cg.create_dataset(axis, data=coords[:,i])
            mesh.create_group("Connectivity").create_dataset(
                "hex->node", data=np.asarray(cells, dtype=np.int32)
            )
        solut = directory / "SOLUT"
        solut.mkdir()
        for t in range(25):
            with h5py.File(solut / f"solut_hit_{t:08d}.h5", "w") as h5:
                fields = h5.create_group("GaseousPhase")
                values = np.arange(spec.side**3, dtype=np.float64)
                for k, name in enumerate(("rho","rhou","rhov","rhow","rhoE")):
                    fields.create_dataset(
                        name, data=(k+1)*0.1 + values*0.05
                        + t*0.002 + ordinal*0.01,
                    )
        ref_file = refdir / reference_filename(spec.case_id)
        ref_file.write_text(
            f"""case_id: {spec.case_id}
reference_scheme: synthetic_common_hit
references:
  rho_ref:
    value: 1.2
    units: kg/m^3
    definition: prescribed_density_fixture
    provenance: test_explicit_case_fixture
    inference_available: true
  U_ref:
    value: {1.0 + ordinal*0.1}
    units: m/s
    definition: prescribed_rms_velocity_fixture
    provenance: test_explicit_case_fixture
    inference_available: true
  L_ref:
    value: {float(spec.side-1)}
    units: m
    definition: prescribed_box_length_fixture
    provenance: test_explicit_case_fixture
    inference_available: true
regime:
  Re_lambda_target:
    value: {54.0 if spec.side == 3 else 94.0}
    definition: prescribed_Taylor_Re_fixture
    provenance: synthetic_test
    inference_available: true
"""
        )
    return root, refdir, chosen


def _full_split(dataset):
    plan = temporal_case_split({
        c.case_id: tuple(dataset.sample_ids[i] for i in dataset.indices_by_case[c.case_id])
        for c in dataset.cases
    }, temporal_embargo=1)
    return plan


def test_six_cases_resolutions_and_snapshot_manifest_without_reading_fields(tmp_path):
    root, refdir, cases = _fixture_archive(tmp_path)
    dataset = AVBPMultiCaseHITDataset(
        root=root, cases=cases, reference_dir=refdir,
        expected_snapshots_per_case=25,
    )
    assert len(dataset) == 150
    assert len(dataset.cases) == 6
    assert set(dataset.resolution_by_sample_id.values()) == {3,4}
    assert tuple(len(dataset.indices_by_case[c.case_id]) for c in cases) == (25,)*6
    assert len(dataset._mesh_cache) == 0
    assert len(_references_signature(dataset)) == 6
    info = inspect_inventory(dataset, reference_dir=refdir)
    assert info["samples"] == 150
    assert all(c["reference_file_exists"] for c in info["cases"])
    assert dataset[0].reference_scales["rho_ref"].value == 1.2


def test_missing_case_references_fail_closed_but_inspect_still_works(tmp_path):
    root, refdir, cases = _fixture_archive(tmp_path)
    (refdir / reference_filename(cases[1].case_id)).unlink()
    with pytest.raises(FileNotFoundError, match="authoritative case reference"):
        AVBPMultiCaseHITDataset(
            root=root, cases=cases, reference_dir=refdir,
            expected_snapshots_per_case=25,
        )
    inspect = AVBPMultiCaseHITDataset(
        root=root, cases=cases, reference_dir=refdir,
        expected_snapshots_per_case=25,
        require_references=False,
    )
    assert len(inspect) == 150
    assert inspect.inventory[1]["reference_file_exists"] is False


def test_temporal_embargo_disjoint_and_resolution_balanced_order(tmp_path):
    root, refdir, cases = _fixture_archive(tmp_path)
    dataset = AVBPMultiCaseHITDataset(
        root=root, cases=cases, reference_dir=refdir,
        expected_snapshots_per_case=25,
    )
    plan = _full_split(dataset)
    assert len(plan.manifest.train_ids) == 16*6
    assert len(plan.manifest.validation_ids) == 2*6
    assert len(plan.manifest.test_ids) == 3*6
    assert len(plan.excluded_ids) == 4*6
    assert not set(plan.manifest.all_ids) & set(plan.excluded_ids)
    assert set(plan.manifest.all_ids) | set(plan.excluded_ids) == set(dataset.sample_ids)
    selected = _indices_by_case(plan.manifest.train_ids, dataset)
    order_a = balanced_case_interleave(selected, seed=42, epoch=0)
    order_b = balanced_case_interleave(selected, seed=42, epoch=0)
    order_other = balanced_case_interleave(selected, seed=42, epoch=1)
    assert order_a == order_b
    assert order_a != order_other
    assert len(order_a) == 16*6 and len(set(order_a)) == len(order_a)
    assert [dataset.resolution_by_sample_id[dataset.sample_ids[i]]
            for i in order_a[:6]] == [c.side for c in cases]
    assert len(_smoke_split(plan.manifest, dataset).train_ids) == 6


def test_smoke_forward_backward_shared_weights_and_per_case_validation(tmp_path):
    root, refdir, cases = _fixture_archive(tmp_path)
    dataset = AVBPMultiCaseHITDataset(
        root=root, cases=cases, reference_dir=refdir,
        expected_snapshots_per_case=25,
    )
    complete = _full_split(dataset)
    smoke = _smoke_split(complete.manifest, dataset)
    assert all(id in complete.manifest.train_ids for id in smoke.train_ids)
    task = FlowMatchingTask(
        state_fields=("rho","rhou","rhov","rhow","rhoE"),
        physical_nondimensionalization=True,
        validation_seed=1234, time_embedding_scale=1000.0,
    )
    indices_by_id = {sid: i for i,sid in enumerate(dataset.sample_ids)}
    scaling = fit_train_standardizers(
        task,
        (dataset[indices_by_id[sid]] for sid in smoke.train_ids),
        dataset.field_catalog,
        smoke,
    )
    assert scaling.physical_nondimensionalization is True
    assert scaling.weighting == "sample_balanced"
    assert len(scaling.train_sample_ids) == 6
    assert torch.isfinite(scaling.inputs.mean).all()
    collator = GraphTaskCollator(
        task, dataset.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    reference = collator([dataset[indices_by_id[smoke.train_ids[0]]]])
    assert reference.attention_edge_indices["dilated"].shape[1] > 0
    model = AlternatingDilatedGeometricDiT(
        in_channels=5, out_channels=5,
        hidden_dim=16, num_heads=2, num_layers=2,
        spatial_dim=3, conditioning_channels=1,
        condition_embed_dim=16,
        sparse_attention_backend="scatter",
        activation_checkpointing=True,
        use_coord_mlp=True, dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(1042)
    trainstats = train_epoch(
        dataset, _indices_by_case(smoke.train_ids, dataset),
        epoch=0, seed=42, accumulation=6,
        model=model, optimizer=optimizer, task=task,
        standardizers=scaling.to(device="cpu", dtype=torch.float32),
        collator=collator, device=torch.device("cpu"),
        generator=generator, max_steps=1,
    )
    assert trainstats["samples"] == 6
    assert trainstats["optimizer_steps"] == 1
    assert len(trainstats["by_case"]) == 6
    assert len(trainstats["by_resolution"]) == 2
    validation = evaluate(
        dataset, tuple(indices_by_id[sid] for sid in smoke.validation_ids),
        model=model, task=task,
        standardizers=scaling.to(device="cpu", dtype=torch.float32),
        collator=collator, device=torch.device("cpu"),
    )
    assert validation["samples"] == 6
    assert len(validation["by_case"]) == 6
    assert set(validation["by_resolution"]) == {"3","4"}
    assert all(value["physical_samples"] == 1
               for value in validation["by_case"].values())
