"""CPU-only empirical PDFs of completed M49 validation generations.

Use all unique Cartesian nodes, not sampled moments or fitted Gaussians.
All populations share the reference mean, standard deviation and bin edges.
Record out-of-range counts instead of clipping or renormalizing the tails.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

CHANNELS = ("rho", "rhou", "rhov", "rhow", "rhoE")
EDGES = np.linspace(-10.0, 10.0, 201)


def histogram(values, center, scale, edges=EDGES):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("PDF requires nonempty finite values")
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid reference normalization")
    z = (values - center) / scale
    counts = np.histogram(z, bins=edges)[0]
    below = int(np.count_nonzero(z < edges[0]))
    above = int(np.count_nonzero(z > edges[-1]))
    if int(counts.sum()) + below + above != values.size:
        raise ValueError("histogram population mismatch")
    return dict(count=int(values.size), counts=counts.tolist(),
                below=below, above=above,
                mean=float(values.mean()), std=float(values.std()),
                minimum=float(values.min()), maximum=float(values.max()))


def columns(q):
    rho = q[..., 0].astype(np.float64)
    if not np.isfinite(rho).all() or np.any(rho <= 0):
        raise ValueError("positive finite density required for velocity PDFs")
    for i, name in enumerate(CHANNELS):
        yield name, q[..., i]
    for i, name in enumerate(("u", "v", "w"), 1):
        yield name, q[..., i] / rho


@lru_cache(maxsize=None)
def unique_mask(run, first_id, n):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
    from scripts.m47_generation_metrics import unique_node_mask

    source = OmegaConf.load(run / "resolved_config.yaml")
    policy = str(source.data.get("periodic_endpoint_policy", "keep"))
    if policy in ("drop_max", "periodic_quotient"):
        if n != 32768:
            raise ValueError("unexpected reduced HIT mesh size")
        return np.ones(n, dtype=bool), policy
    if policy != "keep" or n != 35937:
        raise ValueError("unsupported M49 endpoint policy or mesh size")
    dataset = instantiate(source.data)
    idx = {sid: i for i, sid in enumerate(dataset.sample_ids)}
    mesh = dataset[idx[first_id]].mesh
    grid = infer_cartesian_grid_3d(mesh.coords.cpu().numpy(),
                                  periodic_endpoint_mode="drop_max")
    mask = unique_node_mask(grid)
    if grid.source_shape != (33, 33, 33) or int(mask.sum()) != 32768:
        raise ValueError("unexpected endpoint-inclusive HIT mesh")
    return mask, policy


def main():
    import torch
    from graph_attention.utils.provenance import collect_runtime_provenance

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    reports = sorted(root.glob(
        "training/*/generations/m49_validation_epoch_*_heun*_seed5678/physical_metrics.json"
    ))
    if not reports:
        raise ValueError("no completed M49 validation evaluations")

    reference_ids = None
    normalization = {}
    canonical = {}
    evaluations = []
    for path in reports:
        dest, run = path.parent, path.parents[2]
        metrics = json.loads(path.read_text())
        manifest = json.loads((run / "dataset_split_manifest.json").read_text())
        artifact = torch.load(dest / "generated_validation.pt",
                              map_location="cpu", weights_only=True)
        ids = tuple(artifact["reference_ids"])
        if ids != tuple(manifest["validation_ids"]) or len(ids) != 180:
            raise ValueError(f"validation manifest mismatch: {dest}")
        if reference_ids is None:
            reference_ids = ids
        if ids != reference_ids:
            raise ValueError(f"different validation population: {dest}")
        if tuple(artifact["generated_ids"]) != tuple(f"gen_{i:06d}" for i in range(180)):
            raise ValueError(f"unexpected generated sample keys: {dest}")
        if tuple(c.split(".")[0] for c in artifact["channel_names"]) != CHANNELS:
            raise ValueError(f"unexpected conservative channel semantics: {dest}")
        if (artifact["sampling_seed"] != 5678
                or artifact["sampling_solver"] != "heun"
                or artifact["sampling_steps"] != metrics["sampling_steps"]):
            raise ValueError(f"sampling protocol mismatch: {dest}")
        node_counts = tuple(artifact["node_counts"])
        n = node_counts[0]
        if node_counts != (n,) * 180:
            raise ValueError(f"unexpected variable node counts: {dest}")
        mask, policy = unique_mask(run, ids[0], n)
        record = dict(run=run.name, epoch=metrics["epoch"],
                      steps=artifact["sampling_steps"], endpoint_policy=policy,
                      generation=str(dest.relative_to(root)))
        for label, key in (("reference", "target_nondimensional"),
                           ("generated", "generated_nondimensional")):
            q = artifact[key].reshape(180, n, len(CHANNELS)).numpy()[:, mask, :]
            stats = {}
            for name, values in columns(q):
                if name not in normalization:
                    if label != "reference":
                        raise ValueError("reference normalization not initialized")
                    x = np.asarray(values, dtype=np.float64)
                    normalization[name] = dict(mean=float(x.mean()), std=float(x.std()))
                norm = normalization[name]
                stats[name] = histogram(values, norm["mean"], norm["std"])
            record[label] = stats
            del q, values
        if not canonical:
            canonical = record["reference"]
        record["reference_histograms_match_first"] = all(
            all(record["reference"][name][k] == canonical[name][k]
                for k in ("count", "counts", "below", "above"))
            for name in canonical
        )
        if not record["reference_histograms_match_first"]:
            print(f"WARNING: reference histogram mismatch: {dest}", flush=True)
        evaluations.append(record)
        print(f"PDFs: {run.name} epoch={record['epoch']} steps={record['steps']}",
              flush=True)
        del artifact

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(protocol="m49_unique_node_pdf_v1",
                  reference_split="validation", reference_ids=reference_ids,
                  provenance=collect_runtime_provenance(),
                  node_weighting="uniform on unique 32^3 Cartesian nodes",
                  variable_units="physically nondimensional conservative variables and velocities",
                  normalized_coordinate="(q - reference_mean) / reference_std",
                  bin_edges=EDGES.tolist(), normalization=normalization,
                  density_convention="counts / (total_count * bin_width); tails recorded separately",
                  evaluations=evaluations)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(out)
    print(f"Saved {len(evaluations)} evaluations to {out}", flush=True)


if __name__ == "__main__":
    main()
