"""Compare identical Gaussian initial states across three Heun discretizations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from graph_attention.utils.provenance import collect_runtime_provenance
from scripts.evaluate_m49_generation import write
from scripts.m47_generation_metrics import unique_node_mask

STEPS = (50,100,200)


def matched_errors(a,b,mask,channels):
    """Per-field standardized RMS and relative L2 on unique physical nodes."""
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != len(channels):
        raise ValueError("matched tensors must have the same [samples,nodes,channels] shape")
    relative, rmse = [], []
    for lhs,rhs in zip(a,b,strict=True):
        x,y=lhs[mask].double(),rhs[mask].double()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("nonfinite generated state in matched comparison")
        d=x-y
        numerator=torch.linalg.vector_norm(d,dim=0)
        denominator=torch.linalg.vector_norm(y,dim=0)
        relative.append(torch.where(denominator>0,numerator/denominator,torch.nan).numpy())
        rmse.append(torch.sqrt(d.square().mean(0)).numpy())
    rel, rms=np.stack(relative),np.stack(rmse)
    out={}
    for i,c in enumerate(channels):
        r=rel[:,i]
        r=r[np.isfinite(r)]
        out[c]={"relative_l2_mean":float(r.mean()) if r.size else None,
                "relative_l2_q95":float(np.quantile(r,.95)) if r.size else None,
                "valid_relative_count":int(r.size),
                "standardized_rmse_mean":float(rms[:,i].mean()),
                "standardized_rmse_q95":float(np.quantile(rms[:,i],.95))}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    args=p.parse_args()
    run=args.run_dir.resolve()
    source=OmegaConf.load(run/"resolved_config.yaml")
    if source.data.get("periodic_endpoint_policy","keep") != "keep":
        raise ValueError("requires original node representation")
    manifest=json.loads((run/"dataset_split_manifest.json").read_text())
    ids=tuple(manifest["validation_ids"])
    keys=tuple(f"gen_{i:06d}" for i in range(len(ids)))
    if len(ids)!=180:
        raise ValueError("expected 180 frozen validation samples")
    dataset=instantiate(source.data)
    mesh=dataset[tuple(dataset.sample_ids).index(ids[0])].mesh
    grid=infer_cartesian_grid_3d(mesh.coords.cpu().numpy(),periodic_endpoint_mode="drop_max")
    if grid.source_shape!=(33,33,33):
        raise ValueError("unexpected mesh")
    mask=torch.from_numpy(unique_node_mask(grid))
    artifacts,metrics={},{}
    for steps in STEPS:
        path=run/"generations"/f"m49_sampling_epoch_1000_heun{steps}_seed5678"
        summary=json.loads((path/"summary.json").read_text())
        a=torch.load(path/"generated_validation.pt",map_location="cpu",weights_only=True)
        m=json.loads((path/"physical_metrics.json").read_text())
        if (tuple(a["reference_ids"])!=ids or tuple(a["generated_ids"])!=keys
            or tuple(a["node_counts"])!=(35937,)*180 or a["sampling_steps"]!=steps
            or a["sampling_seed"]!=5678 or a["sampling_solver"]!="heun"
            or summary["source_checkpoint"]!="epoch_1000.pt"
            or m["source_checkpoint"]!="epoch_1000.pt" or m["sampling_steps"]!=steps):
            raise ValueError("incompatible generation protocol")
        if artifacts:
            prev=artifacts[50]
            if a["channel_names"]!=prev["channel_names"] or not torch.equal(
                    a["target_nondimensional"],prev["target_nondimensional"]):
                raise ValueError("reference fields differ across integration arms")
        artifacts[steps]=a
        g=m["generated"]
        metrics[steps]={"mean_tke_m2_s2":g["mean_tke_m2_s2"],
                        "mean_dilatational_energy_fraction":g["mean_dilatational_energy_fraction"],
                        "relative_spectrum_l1":m["relative_spectrum_l1"],
                        "all_stored_nodes_admissible_count":g["all_stored_nodes_admissible_count"],
                        "endpoint_jump":g["mean_periodic_endpoint_jump_over_std"],
                        "generation_wall_seconds":m.get("generation_wall_seconds")}
    comparisons={}
    for n,m in ((50,100),(100,200),(50,200)):
        a,b=artifacts[n],artifacts[m]
        comparisons[f"{n}_to_{m}"]=matched_errors(
            a["generated_standardized"].reshape(180,35937,5),
            b["generated_standardized"].reshape(180,35937,5),
            mask,a["channel_names"])
    # A rerun at 50 steps also checks the historical evaluation's reproducibility.
    old=run/"generations/m49_validation_epoch_1000_heun50_seed5678/generated_validation.pt"
    if old.is_file():
        a=torch.load(old,map_location="cpu",weights_only=True)
        b=artifacts[50]
        if (a["generated_ids"]!=b["generated_ids"] or a["reference_ids"]!=b["reference_ids"]
                or a["channel_names"]!=b["channel_names"] or not torch.equal(
                    a["target_nondimensional"],b["target_nondimensional"])):
            raise ValueError("historical generation IDs/fields differ from new baseline")
        comparisons["historical_50_to_current_50"]=matched_errors(
            a["generated_standardized"].reshape(180,35937,5),
            b["generated_standardized"].reshape(180,35937,5),mask,b["channel_names"])
    with (run/"epoch_1000.pt").open("rb") as stream:
        checkpoint_sha256=hashlib.file_digest(stream,"sha256").hexdigest()
    out=run/"generations/m49_sampling_convergence_epoch_1000.json"
    write(out,{"protocol":"m49_heun_refinement_v1","source_checkpoint_sha256":checkpoint_sha256,
               "provenance":collect_runtime_provenance(),"reference_count":len(ids),
               "sampling_seed":5678,"steps":list(STEPS),"metrics":metrics,
               "matched_standardized_unique_node_errors":comparisons,
               "interpretation":"200 steps is a finer approximation, not an exact ODE solution"})
    print(f"Report={out}",flush=True)


if __name__=="__main__":
    main()
