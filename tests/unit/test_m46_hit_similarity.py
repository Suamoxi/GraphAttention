"""Numerical and source-selection tests for the all-snapshot HIT audit."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts import m46_hit_similarity as audit


def _mesh(path: Path, *, side: int = 5, permute: bool = False) -> np.ndarray:
    xyz = np.stack(np.meshgrid(*(np.linspace(0, 1, side),) * 3, indexing="ij"), -1).reshape(-1, 3)
    permutation = np.random.default_rng(12).permutation(len(xyz)) if permute else np.arange(len(xyz))
    xyz = xyz[permutation]
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("Coordinates")
        for i, name in enumerate("xyz"):
            g.create_dataset(name, data=xyz[:, i])
        f.create_dataset("Connectivity/hex->node", data=np.ones((side-1)**3 * 8, dtype="i4"))
    return xyz


def _snapshot(path: Path, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rho = 1.2 + 0.01 * xyz[:, 0]
    vel = np.sin(2 * np.pi * xyz[:, 0])
    with h5py.File(path, "w") as f:
        g = f.create_group("GaseousPhase")
        for key, value in {
            "rho": rho, "rhou": rho * vel, "rhov": rho * vel / 2,
            "rhow": rho * vel / 3, "rhoE": rho * (100.0 + vel**2 / 2),
        }.items():
            g.create_dataset(key, data=value)


def _references(path: Path, case: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"case_id: {case}", "reference_scheme: hit_target_rms_box_reference", "references:"]
    for name, value, unit in (("rho_ref", 1.2, "kg/m^3"), ("U_ref", 10.0, "m/s"),
                              ("L_ref", 1.0, "m"), ("a_ref", 340.0, "m/s")):
        lines.extend((f"  {name}:", f"    value: {value}", f"    units: {unit}",
                      "    definition: synthetic_fixed_reference", "    provenance: synthetic_test",
                      "    inference_available: true"))
    path.write_text("\n".join(lines) + "\n")


def test_streaming_moments_are_full_population_moments() -> None:
    x = np.random.default_rng(3).normal(10, 1.7, 1234)
    acc = audit.Moments()
    for chunk in np.array_split(x, 17):
        acc.add(chunk)
    result = acc.record()
    centered = x - np.mean(x)
    assert result["mean"] == pytest.approx(np.mean(x), rel=1e-13)
    assert result["std"] == pytest.approx(np.std(x), rel=1e-13)
    assert result["skew"] == pytest.approx(np.mean(centered**3) / np.std(x)**3, abs=1e-12)
    assert result["excess_kurtosis"] == pytest.approx(
        np.mean(centered**4) / np.std(x)**4 - 3, abs=1e-12)


def test_spectral_helmholtz_partition_and_parseval() -> None:
    n = 12
    x = np.arange(n, dtype=np.float64)[:, None, None]
    sin = np.broadcast_to(np.sin(2*np.pi*x/n), (n, n, n))
    transverse = np.zeros((n, n, n, 3), dtype=np.float64)
    transverse[..., 1] = sin
    sol = audit._spectral(transverse)
    assert sol["TKE_spectral_star"] == pytest.approx(0.25, abs=1e-12)
    assert sol["dilatational_fraction"] == pytest.approx(0, abs=1e-12)
    longitudinal = np.zeros_like(transverse)
    longitudinal[..., 0] = sin
    dil = audit._spectral(longitudinal)
    assert dil["TKE_spectral_star"] == pytest.approx(0.25, abs=1e-12)
    assert dil["dilatational_fraction"] == pytest.approx(1, abs=1e-12)
    assert sum(dil["E_star"]) * 2 * np.pi == pytest.approx(0.25, abs=1e-12)


def test_cartesian_grid_renumbering_and_periodic_endpoint(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.h5"
    xyz = _mesh(mesh, permute=True)
    inverse, interior, forward, extent = audit._mesh_index(mesh, 5, chunk=17)
    assert inverse.shape == (4**3,)
    assert np.sum(interior) == 4**3
    assert extent == pytest.approx(1.0)
    ordered = xyz[inverse]
    assert np.all(ordered >= 0)
    assert np.all(ordered < 1.0)
    assert np.array_equal(forward[inverse], np.arange(4**3))


def test_analyze_sample_with_explicit_case_reference(tmp_path: Path) -> None:
    case = "ReT060/Mt0.59_Xi0.3"
    path = tmp_path / "test_case"
    mesh = path / "MESH" / "mesh.mesh.h5"
    xyz = _mesh(mesh, side=5, permute=True)
    sol = path / "RUN" / "SOLUT" / "solut_hit_00000001.h5"
    _snapshot(sol, xyz)
    ref = tmp_path / "ref.yaml"
    _references(ref, case)
    sample = {"sample_id": f"{case}:solut_hit_00000001", "case_id": case,
              "snapshot": str(sol), "mesh": str(mesh), "side": 5, "source": "reT_archive"}
    result, sampled = audit._analyze_sample(sample, ref, {}, 15, True, 46)
    assert result["node_count_unique"] == 64
    assert result["fields"]["rho"]["mean"] > 0.0
    assert result["mach_t_reference_sound_speed"] is not None
    assert result["spectral"]["TKE_spectral_star"] == pytest.approx(result["tke_star"], rel=1e-6)
    assert all(len(v) == 15 for v in sampled.values())


def test_inventory_keeps_all_unique_snapshots_and_prefers_archive(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from graph_attention.data.multicase_avbp import HITCaseSpec

    archive, pilot = tmp_path / "archive", tmp_path / "pilot"
    monkeypatch.setattr(audit, "DEFAULT_M43_CASES", (HITCaseSpec("Re54/Ma0.05", 5),))
    monkeypatch.setattr(audit, "PILOT_SIDES", {"ReT060": 5})
    for root, case, counters in (
        (archive, "Re54/Ma0.05", (1, 2)),
        (archive, "ReT060/Mt0.59_Xi0.3", (1, 2)),
        (pilot / "data", "ReT060/Mt0.59_Xi0.3", (2, 3)),
        (pilot / "data", "ReT060/Mt0.50_Xi0.0", (1,)),
    ):
        basedir = root / case
        xyz = _mesh(basedir / "MESH" / "mesh.mesh.h5")
        for k in counters:
            _snapshot(basedir / "RUN" / "SOLUT" / f"solut_hit_{k:08d}.h5", xyz)
    found = audit.inventory(archive, pilot)
    assert found["counts"] == {"cases": 3, "snapshots": 6, "duplicate_paths": 1}
    sample = next(s for s in found["samples"] if s["sample_id"] == "ReT060/Mt0.59_Xi0.3:solut_hit_00000002")
    assert sample["source"] == "reT_archive"
    assert found["stationarity_filter"] is False