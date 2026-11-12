"""M47 physical regime references, stationary-only training and conditioning gates."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from graph_attention.data.multicase_avbp import HITCaseSpec, AVBPMultiCaseHITDataset
from scripts.m46_stationarity_selection import select_snapshots
from scripts import m47_reference_conditioning as physical
from scripts import m47_qualified_setup as qualified


def _base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cases = (HITCaseSpec("Re54/Ma0.05", 3), HITCaseSpec("Re94/Ma0.2", 4))
    monkeypatch.setattr(physical, "DEFAULT_M43_CASES", cases)
    refs = tmp_path / "refs"
    root = tmp_path / "archive"
    refs.mkdir()
    for spec in cases:
        code = spec.case_id.replace("/", "_")
        (refs / (code + ".yaml")).write_text(
            f"""case_id: {spec.case_id}
reference_scheme: fixed_hit_box
references:
  rho_ref: {{value: 1.2, units: kg/m^3, definition: rho0, provenance: setup, inference_available: true}}
  U_ref: {{value: 10.0, units: m/s, definition: forcing_velocity, provenance: setup, inference_available: true}}
  L_ref: {{value: 2.0, units: m, definition: box_length, provenance: setup, inference_available: true}}
regime:
  Re_lambda_target: {{value: 54.0, definition: nominal_Taylor_Re, provenance: setup, inference_available: true}}
"""
        )
        p = root / spec.case_id / "RUN" / "mixture_database.dat"
        p.parent.mkdir(parents=True)
        p.write_text("synthetic physics source file for test\n")
    return cases, refs, root


def _physical_input(cases):
    return {"version": physical.VERSION, "cases": {
        c.case_id: {
            "a_ref": {
                "value": 50.0, "units": "m/s", "definition": "setup_sound_speed",
                "provenance": "manually audited EOS", "source_path": "RUN/mixture_database.dat",
            },
            "mu_ref": {
                "value": 0.04, "units": "kg/(m*s)",
                "definition": "setup_dynamic_viscosity_at_reference_state",
                "provenance": "manually audited viscosity law", "source_path": "RUN/mixture_database.dat",
            },
        } for c in cases
    }}


def test_missing_data_is_reported_not_inferred_from_case_label(tmp_path, monkeypatch):
    cases, refs, root = _base(tmp_path, monkeypatch)
    source_bytes = {case.case_id: (refs / physical.reference_filename(case.case_id)).read_bytes()
                    for case in cases}
    generated, report = physical.build_reference_cases(root, refs, None)
    assert generated == {}
    assert report["status"] == "missing_physical_inputs"
    assert all(report["missing"][c.case_id] == ["a_ref", "mu_ref or nu_ref"] for c in cases)
    assert all((refs / physical.reference_filename(c.case_id)).read_bytes() == source_bytes[c.case_id]
               for c in cases)


def test_build_explicit_conditions_and_preserve_original_refs(tmp_path, monkeypatch):
    cases, refs, root = _base(tmp_path, monkeypatch)
    physical_input = tmp_path / "physical_inputs.yaml"
    OmegaConf.save(OmegaConf.create(_physical_input(cases)), physical_input)
    before = {c.case_id: (refs / physical.reference_filename(c.case_id)).read_bytes() for c in cases}
    payload, report = physical.build_reference_cases(root, refs, physical_input)
    assert report["status"] == "ready"
    assert len(payload) == 2
    # 1.2 kg/m^3 * 10 m/s * 2 m / 0.04 kg/(m s) = 600, Ma=10/50=0.2
    for c in cases:
        item = payload[c.case_id]
        assert item["regime"]["Re_ref"]["value"] == pytest.approx(600.)
        assert item["regime"]["Ma_ref"]["value"] == pytest.approx(0.2)
        assert item["regime"]["log_Re_condition"]["value"] == pytest.approx(math.log(600))
        assert item["regime"]["Ma_condition"]["value"] == pytest.approx(2.)
        assert item["regime"]["Re_lambda_target"]["value"] == 54.
        assert report["cases"][c.case_id]["evidence"]["mu_ref"]["source_sha256"] == hashlib.sha256(
            b"synthetic physics source file for test\n").hexdigest()
        assert (refs / physical.reference_filename(c.case_id)).read_bytes() == before[c.case_id]
    # Identical reference states have identical Re and Ma despite different case labels.
    assert payload[cases[0].case_id]["regime"]["Re_ref"]["value"] == payload[cases[1].case_id]["regime"]["Re_ref"]["value"]


def test_missing_or_conflicting_physics_fails(tmp_path, monkeypatch):
    cases, refs, root = _base(tmp_path, monkeypatch)
    p = tmp_path / "in.yaml"
    inputs = _physical_input(cases)
    inputs["cases"][cases[0].case_id]["a_ref"]["value"] = None
    OmegaConf.save(OmegaConf.create(inputs), p)
    with pytest.raises(TypeError, match="explicit numerical literal"):
        physical.build_reference_cases(root, refs, p)
    inputs = _physical_input(cases)
    inputs["cases"][cases[0].case_id]["mu_ref"]["units"] = "m^2/s"
    OmegaConf.save(OmegaConf.create(inputs), p)
    with pytest.raises(ValueError, match="expected units"):
        physical.build_reference_cases(root, refs, p)
    inputs = _physical_input(cases)
    inputs["cases"][cases[0].case_id]["a_ref"]["source_path"] = "RUN/missing.dat"
    OmegaConf.save(OmegaConf.create(inputs), p)
    with pytest.raises(FileNotFoundError):
        physical.build_reference_cases(root, refs, p)


def test_kinematic_viscosity_is_equivalent(tmp_path, monkeypatch):
    cases, refs, root = _base(tmp_path, monkeypatch)
    inp = _physical_input(cases)
    for vals in inp["cases"].values():
        vals["nu_ref"] = vals.pop("mu_ref")
        vals["nu_ref"]["value"] = 0.04 / 1.2
        vals["nu_ref"]["units"] = "m^2/s"
        vals["nu_ref"]["definition"] = "reference_kinematic_viscosity"
    p = tmp_path / "in.yaml"
    OmegaConf.save(OmegaConf.create(inp), p)
    payload, report = physical.build_reference_cases(root, refs, p)
    assert report["cases"][cases[0].case_id]["Re_ref"] == pytest.approx(600.)
    assert payload[cases[0].case_id]["regime"]["Ma_ref"]["value"] == pytest.approx(0.2)


def _qualification_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cases = tuple(HITCaseSpec(name, 65 if name.startswith("Re54") else 129)
                  for name in ("Re54/Ma0.05", "Re54/Ma0.2", "Re54/Ma0.3",
                               "Re94/Ma0.05", "Re94/Ma0.2", "Re94/Ma0.3"))
    monkeypatch.setattr(qualified, "DEFAULT_M43_CASES", cases)
    inventory = {
        "cases": [{"case_id": c.case_id, "side": c.side, "snapshots": 100} for c in cases],
        "samples": [{"case_id": c.case_id,
                     "sample_id": f"{c.case_id}:solut_hit_{i:08d}",
                     "snapshot": f"/archive/{c.case_id}/SOLUT/solut_hit_{i:08d}.h5",
                     "counter": i, "source": "legacy_archive"}
                    for c in cases for i in range(1, 101)],
    }
    policy = {"version": "m46_stationary_eligibility_v1", "cases": {
        c.case_id: {"first_counter": 41, "last_counter": 100,
                    "transient_end_counter": 10, "require_full_window": True,
                    "evidence": "M43 reviewed exploratory stationary cutoff for legacy dataset"}
        for c in cases
    }}
    inv_file = tmp_path / "inventory.json"
    rule_file = tmp_path / "policy.json"
    manifest_file = tmp_path / "qualified_manifest.json"
    inv_file.write_text(json.dumps(inventory))
    rule_file.write_text(json.dumps(policy))
    qualified_manifest, _, _ = select_snapshots(inventory, policy)
    qualified_manifest["qualification"]["source_inventory_sha256"] = qualified.file_sha256(inv_file)
    qualified_manifest["qualification"]["policy_sha256"] = qualified.file_sha256(rule_file)
    manifest_file.write_text(json.dumps(qualified_manifest))
    return manifest_file, inv_file, rule_file


def test_eligible_training_selection_excludes_240_transient_uncertain(tmp_path, monkeypatch):
    manifest, inventory, policy = _qualification_fixture(tmp_path, monkeypatch)
    ids, signature = qualified.eligible_sample_ids(manifest, inventory, policy)
    assert len(ids) == 360 and len(set(ids)) == 360
    assert all(int(s.rsplit("_", 1)[1]) >= 41 for s in ids)
    assert signature["stationarity_selection"].startswith("exploratory")
    bad = json.loads(manifest.read_text())
    bad["samples"][0]["counter"] = 1
    manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="differs from reviewed"):
        qualified.eligible_sample_ids(manifest, inventory, policy)


def test_stale_selection_policy_fails_closed(tmp_path, monkeypatch):
    manifest, inventory, policy = _qualification_fixture(tmp_path, monkeypatch)
    raw = json.loads(policy.read_text())
    raw["cases"]["Re54/Ma0.05"]["first_counter"] = 45
    policy.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="stale"):
        qualified.eligible_sample_ids(manifest, inventory, policy)



def _tiny_h5_case(tmp_path: Path):
    """One physically small 3^3 synthetic mesh; six numbered snapshots."""
    import h5py
    import numpy as np
    from itertools import product

    case = HITCaseSpec("Re54/Ma0.05", 3)
    root = tmp_path / "archive"
    refdir = tmp_path / "refs"
    refdir.mkdir()
    d = root / case.case_id
    mesh_path = d / "MESH" / "mesh.mesh.h5"
    mesh_path.parent.mkdir(parents=True)
    coords = np.array(list(product(range(3), repeat=3)), dtype=np.float64)
    cells = []
    for x, y, z in product(range(2), repeat=3):
        vertices = [(x,y,z), (x+1,y,z), (x+1,y+1,z), (x,y+1,z),
                    (x,y,z+1), (x+1,y,z+1), (x+1,y+1,z+1), (x,y+1,z+1)]
        cells.extend([((i*3+j)*3+k)+1 for i,j,k in vertices])
    with h5py.File(mesh_path, "w") as mesh:
        group = mesh.create_group("Coordinates")
        for axis, col in zip("xyz", range(3)):
            group.create_dataset(axis, data=coords[:,col])
        mesh.create_group("Connectivity").create_dataset(
            "hex->node", data=np.asarray(cells, dtype=np.int32))
    solut = d / "SOLUT"
    solut.mkdir()
    for t in range(1, 7):
        with h5py.File(solut / f"solut_hit_{t:08d}.h5", "w") as out:
            group = out.create_group("GaseousPhase")
            for i, name in enumerate(("rho", "rhou", "rhov", "rhow", "rhoE")):
                group.create_dataset(name, data=np.full((27,), 1.2+i*0.1+t*0.01))
    (refdir / physical.reference_filename(case.case_id)).write_text(
        f"""case_id: {case.case_id}
reference_scheme: synthetic_hit
references:
  rho_ref: {{value: 1.2, units: kg/m^3, definition: rho, provenance: setup, inference_available: true}}
  U_ref: {{value: 10., units: m/s, definition: U, provenance: setup, inference_available: true}}
  L_ref: {{value: 2., units: m, definition: L, provenance: setup, inference_available: true}}
regime:
  Ma_condition: {{value: 2., definition: Ma_ref_divided_by_fixed_0.1, provenance: setup, inference_available: true}}
  log_Re_condition: {{value: 6.396929655, definition: natural_log_Re_ref, provenance: setup, inference_available: true}}
""")
    return root, refdir, case


def test_dataset_allowlist_and_conditional_flow_probe(tmp_path):
    """Minimal CPU test of the existing DiT/Flow Matching conditioning path."""
    import torch
    from graph_attention.tasks import FlowMatchingTask
    from graph_attention.training.data_pipeline import GraphTaskCollator

    root, refdir, case = _tiny_h5_case(tmp_path)
    allow = tuple(f"{case.case_id}:solut_hit_{i:08d}" for i in range(4, 7))
    dataset = AVBPMultiCaseHITDataset(
        root=root, cases=(case,), reference_dir=refdir,
        expected_snapshots_per_case=6, selected_sample_ids=allow,
    )
    assert len(dataset) == 3
    assert dataset.inventory[0]["count"] == 3
    assert dataset.inventory[0]["discovered_count"] == 6
    assert {int(s.rsplit("_", 1)[1]) for s in dataset.sample_ids} == {4,5,6}
    task = FlowMatchingTask(
        state_fields=("rho", "rhou", "rhov", "rhow", "rhoE"),
        conditioning_parameters=physical.CONDITION_NAMES["mach_re"],
        physical_nondimensionalization=True,
        time_embedding_scale=1000.0,
    )
    collator = GraphTaskCollator(
        task, dataset.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    state = collator([dataset[0]])
    assert state.conditioning_names == ("Ma_condition", "log_Re_condition")
    torch.testing.assert_close(
        state.conditioning[0],
        torch.tensor([2., 6.396929655], dtype=state.conditioning.dtype),
    )
    path = task.make_validation_problem(state)
    assert path.conditioning.shape == (1, 3)
    assert path.conditioning_names == ("Ma_condition", "log_Re_condition", "flow_time")


def test_dataset_allowlist_rejects_missing_snapshot(tmp_path):
    root, refdir, case = _tiny_h5_case(tmp_path)
    with pytest.raises(ValueError, match="selected snapshot set mismatch"):
        AVBPMultiCaseHITDataset(
            root=root, cases=(case,), reference_dir=refdir,
            expected_snapshots_per_case=6,
            selected_sample_ids=(
                f"{case.case_id}:solut_hit_00000004",
                f"{case.case_id}:solut_hit_00000100",
            ),
        )