"""Read-only HIT_LES_FORCED snapshot inventory and stationarity diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from graph_attention.data import AVBPDirectoryHDF5Dataset
from graph_attention.utils.provenance import collect_runtime_provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot-dir", default="/scratch/coop/theret/HIT_LES_FORCED/RUN/SOLUT")
    p.add_argument("--mesh-file", default="/scratch/coop/theret/HIT_LES_FORCED/MESH/mesh.mesh.h5")
    p.add_argument("--case-file", default="cases/HIT_LES_FORCED.yaml")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = AVBPDirectoryHDF5Dataset(snapshot_dir=args.snapshot_dir, mesh_file=args.mesh_file,
        snapshot_pattern="*.h5", mesh_id="HIT_LES_FORCED:3d_fixed_mesh",
        case_file=args.case_file, case_id="HIT_LES_FORCED")
    rows = []
    digests = defaultdict(list)
    fields = ("rho", "rhou", "rhov", "rhow", "rhoE")
    for i, name in enumerate(dataset.sample_ids):
        sample = dataset[i]
        match = re.fullmatch(r"solut_hit_(\d+)", name)
        index = int(match.group(1)) if match else None
        digest = hashlib.sha256()
        raw = {}
        for field in fields:
            value = sample.fields[field].detach().cpu().contiguous()
            digest.update(field.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
            raw[field] = value.to(torch.float64).reshape(-1)
        key = digest.hexdigest()
        digests[key].append(name)
        row = dict(sample_id=name, filename_index=index, file=str(dataset.files[i]),
                   field_sha256=key, num_nodes=raw["rho"].numel())
        finite = all(bool(torch.isfinite(v).all()) for v in raw.values())
        rho = raw["rho"]
        row["finite"] = finite
        row["positive_density"] = bool((rho > 0).all())
        if finite and row["positive_density"]:
            velocity = torch.stack([raw[k]/rho for k in ("rhou", "rhov", "rhow")], dim=1)
            fluct = velocity - velocity.mean(dim=0)
            row["node_mean"] = {k: float(v.mean()) for k, v in raw.items()}
            row["node_std"] = {k: float(v.std(unbiased=False)) for k, v in raw.items()}
            row["velocity_mean"] = velocity.mean(dim=0).tolist()
            row["velocity_std"] = velocity.std(dim=0, unbiased=False).tolist()
            row["fluctuation_kinetic_energy_per_mass"] = float(0.5*fluct.square().sum(dim=1).mean())
            e_internal = raw["rhoE"]/rho - 0.5*velocity.square().sum(dim=1)
            row["specific_internal_energy_min"] = float(e_internal.min())
        rows.append(row)
        if (i+1) % 100 == 0:
            print(f"audited {i+1}/{len(dataset)}", flush=True)
    rows.sort(key=lambda r: (r["filename_index"] is None, r["filename_index"] or 0))
    payload = dict(provenance=collect_runtime_provenance(), count=len(rows),
                   time_basis="numeric filename index, not verified physical time",
                   statistics_weighting="unweighted stored nodes (periodic endpoint duplicates not removed)",
                   duplicate_definition="exact loaded conservative field tensor bytes, same mesh",
                   duplicate_groups=[names for names in digests.values() if len(names)>1],
                   nonnumbered_ids=[r["sample_id"] for r in rows if r["filename_index"] is None],
                   snapshots=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.output)


if __name__ == "__main__":
    main()
