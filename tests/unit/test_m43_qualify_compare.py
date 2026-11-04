"""End-to-end synthetic audit-cache regression, without production DNS or GPU."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts import m43_qualify_compare as m43
from graph_attention.evaluation.m43_qualification import QualificationConfig


@pytest.fixture
def synthetic_cache(tmp_path: Path, monkeypatch):
    cases = ("Re54/Ma0.05", "Re94/Ma0.05")
    monkeypatch.setattr(m43, "ALL_CASES", cases)
    root = tmp_path / "audit"
    for ci, case in enumerate(cases):
        directory = root / "cases" / case.replace("/", "_")
        (directory / "spectra").mkdir(parents=True)
        (directory / "lag").mkdir()
        mesh = directory / "mesh.h5"
        with h5py.File(mesh, "w") as h5:
            h5.create_dataset("dummy", data=np.arange(3))
        state_files = []
        history = []
        for index in range(100):
            spinup = min(1.0, 0.4 + 0.6 * index / 20)
            row = {
                "snapshot_index": index,
                "turbulent_k": spinup * (200 + 10 * ci),
                "enstrophy": spinup * (10 + ci),
                "std_rho": spinup * 0.01,
                "mean_rho": 1.2,
                "u_rms_one_component": np.sqrt(spinup),
                "f4_spectral_unresolved_energy_fraction": 0.04,
            }
            history.append(row)
            k = np.arange(1, 41, dtype=float)
            spectrum = np.exp(-k / 12) * spinup
            np.savez_compressed(directory / "spectra" / f"snapshot_{index:03d}.npz",
                                k=k, E=spectrum)
            for factor in (2, 4, 8):
                r = np.arange(1, 9, dtype=float) * (0.1 if ci == 0 else 0.05)
                np.savez_compressed(
                    directory / "lag" / f"snapshot_{index:03d}_f{factor}.npz",
                    r=r, r_ll=np.exp(-r), s2_ll=r ** 0.7, s3_ll=-r, s4_ll=r**1.2,
                )
            # Only snapshots in the reviewed stationary interval need raw files.
            if index >= 35:
                file = directory / f"solut_{index+1:08d}.h5"
                with h5py.File(file, "w") as h5:
                    phase = h5.create_group("GaseousPhase")
                    rho = np.ones(32) * 1.2
                    for field in m43.CONSERVATIVE_FIELDS:
                        raw = {
                            "rho": rho,
                            "rhou": rho * np.linspace(1.9, 2.1, 32),
                            "rhov": rho * np.linspace(0.2, 0.5, 32),
                            "rhow": rho * np.linspace(0.3, 0.8, 32),
                            "rhoE": rho * 20,
                        }[field]
                        phase.create_dataset(field, data=raw)
                st = file.stat()
                state_files.append([str(file), st.st_size, st.st_mtime_ns])
            else:
                state_files.append([str(directory / f"solut_{index+1:08d}.h5"), 0, 0])
        with (directory / "time_history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        st = mesh.stat()
        (directory / "source_signature.json").write_text(json.dumps({
            "mesh": str(mesh), "mesh_size": st.st_size, "mesh_mtime_ns": st.st_mtime_ns,
            "snapshots": state_files,
            "run_params_save_schedule": {
                "save_interval_over_forcing_characteristic_time": 1.00233
            },
        }))
    refs = {case: {
        "field_scales": {
            "rho": 1.2, "rhou": 2.4, "rhov": 2.4, "rhow": 2.4, "rhoE": 4.8
        }, "U_ref": 2.0, "L_ref": 1.0, "rho_ref": 1.2,
        "sha256": "synthetic_only", "scheme": "test_convective",
    } for case in cases}
    bounds = {case: {"transient_end_index": 24, "stationary_start_index": 35}
              for case in cases}
    return root, tmp_path / "results", refs, bounds


def test_qualify_sample_compare_without_changing_dns(synthetic_cache, monkeypatch) -> None:
    root, dest, refs, bounds = synthetic_cache
    m43.run_qualify(root, dest, {}, QualificationConfig(bootstrap_repeats=32))
    provisional = list(csv.DictReader((dest / "snapshot_classification_manifest.csv").open()))
    assert len(provisional) == 200
    assert not any(row["classification"] == "stationary" for row in provisional)
    assert all(row["reviewed"] == "False" for row in provisional)

    m43.run_qualify(root, dest, bounds, QualificationConfig(bootstrap_repeats=32))
    reviewed = list(csv.DictReader((dest / "snapshot_classification_manifest.csv").open()))
    assert sum(row["classification"] == "stationary" for row in reviewed) == 130

    monkeypatch.setattr(m43, "_unique_periodic_source_nodes", lambda _: np.arange(32))
    m43.run_sample(root, dest, bounds, refs, nodes_per_snapshot=16, seed=13)
    case_file = dest / "fields" / "Re54_Ma0.05" / "snapshot_035.npz"
    with np.load(case_file) as data:
        assert data["values"].shape == (16, 8)
        np.testing.assert_allclose(data["values"][:, 1] / data["values"][:, 0],
                                   data["values"][:, 5])
    m43.run_compare(root, dest, bounds, refs, "test_convective", seed=13)
    distances = list(csv.DictReader((dest / "pairwise_m3_3_wasserstein.csv").open()))
    assert len(distances) == len(m43.CHANNELS)
    lag = list(csv.DictReader((dest / "pairwise_m3_3_residual_lag.csv").open()))
    assert len(lag) == 1
    assert lag[0]["status"] == "matched_filter_width"
    assert (dest / "stationary_m3_3_energy_spectra.pdf").is_file()
    with pytest.raises(RuntimeError, match="incompatible"):
        m43.run_sample(root, dest, bounds, refs, nodes_per_snapshot=24, seed=13)


def test_missing_or_partial_review_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="six case"):
        path = tmp_path / "reviewed.json"
        path.write_text(json.dumps({"Re54/Ma0.05": {
            "transient_end_index": 10, "stationary_start_index": 30,
        }}))
        m43._reviewed_bounds(path)
