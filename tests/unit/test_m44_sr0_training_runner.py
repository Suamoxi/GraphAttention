"""End-to-end CPU gate: snapshot disjointness, SR-0 training and exact resume.

Synthetic nonlinear conservative fields exercise the pipeline but make no CFD
physical-validity or generalization claim.
"""

from argparse import Namespace
from itertools import product
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.train_m44_sr0 import run


def _archive(tmp_path: Path) -> tuple[Path, Path]:
    side = 7
    root = tmp_path / "archive"
    case = root / "Re54" / "Ma0.3"
    mesh_path = case / "MESH" / "mesh.mesh.h5"
    mesh_path.parent.mkdir(parents=True)
    coords = np.asarray(list(product(range(side), repeat=3)), dtype=np.float64)
    cells = []
    for i, j, k in product(range(side-1), repeat=3):
        corners = [
            (i,j,k), (i+1,j,k), (i+1,j+1,k), (i,j+1,k),
            (i,j,k+1), (i+1,j,k+1), (i+1,j+1,k+1), (i,j+1,k+1),
        ]
        cells.extend([((a*side+b)*side+c)+1 for a,b,c in corners])
    with h5py.File(mesh_path, "w") as h5:
        axes = h5.create_group("Coordinates")
        for d, name in enumerate("xyz"):
            axes.create_dataset(name, data=coords[:,d])
        h5.create_group("Connectivity").create_dataset(
            "hex->node", data=np.asarray(cells,dtype=np.int32)
        )

    directory = case / "SOLUT"
    directory.mkdir()
    x, y, z = coords.T
    for n in range(8):
        with h5py.File(directory / f"solut_hit_{n:08d}.h5", "w") as h5:
            group = h5.create_group("GaseousPhase")
            data = {
                "rho": 1.2 + 0.15*np.sin(x*0.6+n*0.2) + 0.02*y,
                "rhou": 0.3*np.cos(y*0.4+n*0.1)+0.05*x*z,
                "rhov": 0.2*np.sin(z*0.7+n*0.11)+0.05*y*x,
                "rhow": 0.2*np.cos(x*0.5+z*0.1+n*0.13)+0.04*x*y,
                "rhoE": 5.0 + 0.2*np.sin((x+y+z)*0.45+n*0.17)+0.09*x,
            }
            for field, values in data.items():
                group.create_dataset(field, data=values)

    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "Re54_Ma0.3.yaml").write_text("""case_id: Re54/Ma0.3
reference_scheme: synthetic_convective
references:
  rho_ref:
    value: 1.2
    units: kg/m^3
    definition: synthetic_unit_density
    provenance: synthetic_hdf5_test
    inference_available: true
  U_ref:
    value: 2.0
    units: m/s
    definition: synthetic_unit_velocity
    provenance: synthetic_hdf5_test
    inference_available: true
  L_ref:
    value: 6.0
    units: m
    definition: synthetic_unit_box
    provenance: synthetic_hdf5_test
    inference_available: true
""")
    return root, refs


def test_sr01_synthetic_hdf5_train_validation_and_resume(tmp_path: Path) -> None:
    root, refs = _archive(tmp_path)
    out = tmp_path / "run"
    args = Namespace(
        root=root, reference_dir=refs, case="Re54/Ma0.3", side=7,
        patch_cells=4, refinement=2, patches_per_snapshot=1, boundary_margin=1,
        max_snapshots=8, train_fraction=0.70, validation_fraction=0.15,
        embargo_snapshots=1, epochs=1, resume=False,
        hidden_dim=16, layers=2, heads=2, lr=1.0e-3, seed=37,
        time_embedding_scale=1000.0, condition_mode="interpolated_only",
        absolute_coordinate_embedding=False, sparse_attention_backend="scatter",
        checkpointing=False, sample_steps=2, device="cpu", output=out,
    )
    first = run(args)
    assert first["status"] == "sr0_multisnapshot_train_complete"
    assert first["patch_counts"] == {"train": 4, "validation": 1, "test": 1}
    assert first["model_configuration"]["in_channels"] == 10
    assert first["model_configuration"]["out_channels"] == 5
    assert first["model_configuration"]["spatial_dim"] == 3
    assert first["model_configuration"]["use_absolute_coordinate_embedding"] is False
    assert math.isfinite(first["test_flow_velocity_mse_at_best_validation"])
    assert math.isfinite(first["conditional_sensitivity"]["mean_abs_velocity_change"])
    assert (out / "last.pt").is_file() and (out / "best.pt").is_file()
    assert (out / "scales.pt").is_file() and (out / "validation_sample.pt").is_file()
    manifest = json.loads((out / "manifest.json").read_text())
    ids = [set(x["sample_id"] for x in manifest["snapshots"][key])
           for key in ("train", "validation", "test", "embargo")]
    assert all(not ids[i] & ids[j] for i in range(4) for j in range(i + 1, 4))
    assert all(len(x) > 0 for x in ids[:3])

    first_saved = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
    assert first_saved["epoch"] == 0
    assert len(first_saved["train_scales"]["train_sample_ids"]) == 4

    resumed = run(
        Namespace(**{**vars(args), "epochs": 2, "resume": True})
    )
    assert resumed["epochs_completed"] == 2
    after = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
    assert after["epoch"] == 1
    assert after["history"][0] == first_saved["history"][0]
    assert all(math.isfinite(float(row["validation_loss"])) for row in after["history"])
