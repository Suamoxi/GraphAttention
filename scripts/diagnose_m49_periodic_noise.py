"""Inference-only periodic source ablation on a fixed endpoint-inclusive M49 checkpoint.

No retraining, output projection, graph modification or test-population access.
The same original sample-keyed Gaussian draws are used in both arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from graph_attention.data.case_definition import load_case_definition
from graph_attention.data.periodic_endpoints import reduce_cartesian_endpoints, reduce_sample
from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from graph_attention.tasks import FlowMatchingTask
from graph_attention.utils.provenance import collect_runtime_provenance
from scripts.evaluate_m49_generation import population, write
from scripts.generate_generative import run_generative_generation
from scripts.m47_generation_metrics import population_wasserstein, relative_spectrum_l1

STEPS = 50
SEED = 5678
OBSERVE = (0, 5, 10, 25, 40, 45, 49, 50)


def representative_indices(retained, full_to_unique):
    return retained[full_to_unique]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--epoch", type=int, default=1000)
    p.add_argument("--tag", required=True, help="unique output suffix, e.g. Slurm job ID")
    args = p.parse_args()
    if args.epoch <= 0 or not args.tag.isalnum():
        p.error("positive epoch and alphanumeric unique tag required")
    run = args.run_dir.resolve()
    src = OmegaConf.load(run/"resolved_config.yaml")
    if src.data.get("periodic_endpoint_policy", "keep") != "keep":
        raise ValueError("this diagnostic requires the original stored endpoint-inclusive mesh")
    manifest = json.loads((run/"dataset_split_manifest.json").read_text())
    if len(manifest["validation_ids"]) != 180 or len(manifest["train_ids"]) != 800:
        raise ValueError("expected original blocked M49 validation membership")
    ckpt = f"epoch_{args.epoch:04d}.pt"
    checkpoint = torch.load(run/ckpt, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != args.epoch-1 or checkpoint["global_step"] != 50*args.epoch or checkpoint["model_class"] != "AlternatingDilatedGeometricDiT":
        raise ValueError("unexpected checkpoint")
    del checkpoint
    with (run/ckpt).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    dataset = instantiate(src.data)
    sample = dataset[tuple(dataset.sample_ids).index(manifest["validation_ids"][0])]
    reduction = reduce_cartesian_endpoints(sample.mesh, side=33)
    reduce_sample(sample, reduction)  # reject nonidentical source endpoints
    n = sample.mesh.num_nodes
    reps = representative_indices(reduction.retained, reduction.full_to_unique)
    duplicate = torch.nonzero(reps != torch.arange(n), as_tuple=True)[0]
    if len(duplicate) != 3169:
        raise ValueError("unexpected duplicate count")
    grid = infer_cartesian_grid_3d(sample.mesh.coords.numpy(), periodic_endpoint_mode="drop_max")
    u_ref = float(load_case_definition(src.data.case_file).reference_scales["U_ref"].value)
    prefix = f"m49_periodic_noise_e{args.epoch:04d}_{args.tag}"
    names = {arm:f"{prefix}_{arm}_heun50_seed5678" for arm in ("iid","tied")}
    paths = {arm:run/"generations"/name for arm,name in names.items()}
    report_path = run/"generations"/f"{prefix}_comparison.json"
    if report_path.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("use a fresh tag; previous diagnostic outputs are immutable")
    report = {"protocol":"m49_periodic_source_inference_only_v1",
              "checkpoint":ckpt,"checkpoint_sha256":digest,
              "provenance":collect_runtime_provenance(),"sampling_seed":SEED,
              "steps":STEPS,"reference_split":"validation",
              "absolute_coordinate_embedding":bool(src.model.get("use_absolute_coordinate_embedding",True)),
              "duplicate_count":len(duplicate),"arms":{},
              "trajectory_metric":"RMS standardized difference of each discarded node to its canonical retained representative; pooled over samples and duplicates",
              "caveat":"tied initial source is out of training distribution; no velocity or final-state projection"}
    ref = rn = ref_ids = ref_values = None
    for arm in ("iid","tied"):
        accum = {step:np.zeros(5,dtype=np.float64) for step in OBSERVE}
        counts = {step:0 for step in OBSERVE}
        def diagnostic_sampler(model,batch,*,task,**kwargs):
            if not isinstance(task, FlowMatchingTask):
                raise TypeError("expected FlowMatchingTask")
            b = batch.num_graphs
            if tuple((batch.ptr[1:]-batch.ptr[:-1]).tolist()) != (n,)*b:
                raise ValueError("mixed or unexpected mesh")
            coords = batch.source.coords.reshape(b,n,3)
            expected = sample.mesh.coords.to(device=coords.device,dtype=coords.dtype)
            if not torch.equal(coords,expected.expand_as(coords)):
                raise ValueError("mesh order differs from qualified representative map")
            local = reps.to(batch.inputs.device)
            dup = duplicate.to(batch.inputs.device)
            x = batch.inputs.reshape(b,n,5)
            if not torch.equal(x[:,dup],x[:,local[dup]]):
                raise ValueError("reference endpoints differ")
            packed = (local[None,:]+torch.arange(b,device=local.device)[:,None]*n).reshape(-1)
            def observe(step,state):
                if step not in accum:
                    return
                state = state.reshape(b,n,5)
                diff = state[:,dup].double()-state[:,local[dup]].double()
                accum[step] += diff.square().sum(dim=(0,1)).cpu().numpy()
                counts[step] += b*len(duplicate)
            return task.sample_standardized(model,batch,**kwargs,
                initial_source_index=packed if arm=="tied" else None,
                step_observer=observe)
        cfg = OmegaConf.create(dict(run_dir=str(run),checkpoint=ckpt,device="cuda",
            batch_size=8,num_workers=4,overwrite=False,output_name=names[arm],
            reference_split="validation",
            sampling=dict(steps=STEPS,method="flow_ode",solver="heun",
                          sampling_eps=0.0,final_denoise=False,seed=SEED)))
        run_generative_generation(cfg,diagnostic_sampler=diagnostic_sampler,
            diagnostic_source_distribution=("iid_standard_normal_stored_nodes" if arm=="iid"
                                            else "iid_gaussian_on_unique_nodes_lifted_to_stored_endpoints"))
        a = torch.load(paths[arm]/"generated_validation.pt",map_location="cpu",weights_only=True)
        ids = tuple(a["reference_ids"])
        if ids != tuple(manifest["validation_ids"]) or tuple(a["node_counts"]) != (n,)*180:
            raise ValueError("unexpected generated population")
        if tuple(a["generated_ids"]) != tuple(f"gen_{i:06d}" for i in range(180)):
            raise ValueError("unexpected noise keys")
        if any(count != 180*len(duplicate) for count in counts.values()):
            raise ValueError("incomplete trajectory coverage")
        channels = tuple(a["channel_names"])
        if ref is None:
            ref_ids = ids
            ref_values = a["target_nondimensional"]
            ref,rn = population(ref_values.reshape(180,n,5).numpy(),ids,grid,channels,u_ref)
        elif ids != ref_ids or not torch.equal(ref_values,a["target_nondimensional"]):
            raise ValueError("reference population changed")
        generated,gn = population(a["generated_nondimensional"].reshape(180,n,5).numpy(),
                                  a["generated_ids"],grid,channels,u_ref)
        metrics = {"reference":ref,"generated":generated,
                   "one_point_wasserstein":population_wasserstein(rn,gn),
                   "relative_spectrum_l1":relative_spectrum_l1(ref["velocity_energy_spectrum"],generated["velocity_energy_spectrum"])}
        trajectory = {str(step/STEPS):dict(zip(channels,np.sqrt(accum[step]/counts[step]).tolist(),strict=True))
                      for step in OBSERVE}
        if arm=="tied" and any(value != 0 for value in trajectory["0.0"].values()):
            raise ValueError("tied Gaussian endpoints were not identical at t=0")
        write(paths[arm]/"physical_metrics.json",dict(metrics,trajectory=trajectory,arm=arm))
        report["arms"][arm] = {"output":str(paths[arm]),"trajectory":trajectory,
            "metrics":{k:generated[k] for k in ("count","finite_count","all_stored_nodes_admissible_count",
                       "mean_tke_m2_s2","mean_dilatational_energy_fraction","mean_periodic_endpoint_jump_over_std")},
            "relative_spectrum_l1":metrics["relative_spectrum_l1"],
            "one_point_wasserstein":metrics["one_point_wasserstein"]}
        write(report_path,report)
    print(f"Report={report_path}",flush=True)


if __name__ == "__main__":
    main()
