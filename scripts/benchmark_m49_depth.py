"""Short real-mesh qualification of M49 depth / exact-hop configurations.

Uses original frozen training scalers, train IDs, FP32 and the production
equal-sample accumulated optimizer step. Timings exclude dataset loading and
host collation (reported separately); no scientific learning claim.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from graph_attention.training.data_pipeline import GraphTaskCollator
from graph_attention.training.model_factory import instantiate_controlled_model
from graph_attention.utils.provenance import collect_runtime_provenance
from scripts.evaluate_m49_generation import write
from scripts.train_generative import (
    _accumulated_optimizer_step, _evaluate, _load_standardizers, _task_adapter,
)

# Keep this ordering aligned with the Slurm array. Short compact controls first.
VARIANTS = {
    "10x12": (10,(1,2)),
    "9x123": (9,(1,2,3)),
    "8x123": (8,(1,2,3)),
    "20x12": (20,(1,2)),
    "15x123": (15,(1,2,3)),
}
EFFECTIVE_BATCH = 16
CANDIDATES = ((8,False),(4,False),(8,True),(4,True))
FALLBACKS = ((2,False),(2,True))


def choose_trial(trials,total_gib):
    """Conservative gate on measured peak, not theoretical memory estimates."""
    safe = [row for row in trials if row["status"] == "ok"
            and row["peak_allocated_gib"] <= .85*total_gib
            and row["peak_reserved_gib"] <= .90*total_gib
            and math.isfinite(row["median_update_seconds"])
            and row["median_update_seconds"]>0]
    return min(safe,key=lambda row:row["median_update_seconds"]) if safe else None


def trial(*,batch_size,checkpointing,samples,collator,task,standardizers,source,model_cfg,device,adapter):
    result={"microbatch":batch_size,"accumulation":EFFECTIVE_BATCH//batch_size,
            "activation_checkpointing":checkpointing}
    stage="collation"
    try:
        tick=time.perf_counter()
        batches=[collator(samples[i:i+batch_size]) for i in range(0,EFFECTIVE_BATCH,batch_size)]
        result["host_collation_seconds"]=time.perf_counter()-tick
        cfg=OmegaConf.create(OmegaConf.to_container(model_cfg,resolve=True))
        cfg.activation_checkpointing=checkpointing
        probe=task.make_model_probe(task.make_validation_problem(standardizers.transform(batches[0])))
        stage="model"
        model,meta=instantiate_controlled_model(cfg,probe,seed=int(source.seed))
        model=model.to(device=device,dtype=torch.float32)
        optimizer=instantiate(source.optimizer,params=model.parameters())
        scaler=standardizers.to(device=device,dtype=torch.float32)
        generator=torch.Generator(device=device).manual_seed(int(source.seed)+1000)
        result["parameters"]=sum(p.numel() for p in model.parameters())
        result["initialization"]=meta
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stage="train"
        times=[]
        model.train()
        # Two warm-up updates allocate AdamW state and compile the Triton kernels.
        for i in range(5):
            torch.cuda.synchronize()
            tick=time.perf_counter()
            loss,_,count=_accumulated_optimizer_step(adapter,model,optimizer,batches,
                task=task,standardizers=scaler,device=device,training_generator=generator)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-tick
            if count!=EFFECTIVE_BATCH or not math.isfinite(loss):
                raise ValueError("invalid accumulated training update")
            if i>=2:
                times.append(elapsed)
        stage="validation"
        # Exercise production evaluation path with the trained optimizer state resident.
        # Only train snapshots are used here, even in validation-mode execution.
        vals=[]
        for _ in range(3):
            torch.cuda.synchronize()
            tick=time.perf_counter()
            value=_evaluate(model,[batches[0]],task=task,adapter=adapter,
                            standardizers=scaler,device=device)
            torch.cuda.synchronize()
            vals.append(time.perf_counter()-tick)
            if not math.isfinite(value):
                raise ValueError("nonfinite forward evaluation")
        median=statistics.median(times)
        validation=statistics.median(vals)
        result.update(status="ok",update_seconds=times,median_update_seconds=median,
            samples_per_second=EFFECTIVE_BATCH/median,validation_microbatch_seconds=vals,
            median_validation_microbatch_seconds=validation,
            estimated_epoch_seconds=50*median+math.ceil(180/batch_size)*validation,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="cuda_oom",stage=stage,error=str(exc))
    return result


def benchmark(args):
    source=OmegaConf.load(args.baseline/"resolved_config.yaml")
    manifest=json.loads((args.baseline/"dataset_split_manifest.json").read_text())
    if len(manifest["train_ids"])!=800 or len(manifest["validation_ids"])!=180:
        raise ValueError("expected frozen blocked M49 membership")
    if (source.model.num_layers!=10 or not source.model.get("use_absolute_coordinate_embedding",True)
        or tuple(source.model.get("attention_hop_schedule",(1,2)))!=(1,2)
        or source.model.sparse_attention_backend!="torch_sparse_triton"
        or source.data.get("periodic_endpoint_policy","keep")!="keep"
        or int(source.seed)!=42):
        raise ValueError("expected original absolute-encoding M49 baseline")
    if (source.model.hidden_dim!=128 or source.model.num_heads!=4
        or source.model.mlp_ratio!=4 or source.model.condition_embed_dim!=128
        or float(source.optimizer.lr)!=1e-4 or float(source.optimizer.weight_decay)!=0):
        raise ValueError("unexpected baseline architecture/optimizer")
    standardizers=_load_standardizers(args.baseline/"standardizers.pt")
    if tuple(standardizers.train_sample_ids)!=tuple(manifest["train_ids"]):
        raise ValueError("frozen scaling and train membership disagree")
    device=torch.device("cuda")
    total=torch.cuda.get_device_properties(device).total_memory/1024**3
    provenance=collect_runtime_provenance()
    if not provenance["git"]["sha"] or provenance["git"]["dirty"] is not False:
        raise ValueError("benchmark requires clean accessible Git metadata")
    layers,schedule=VARIANTS[args.variant]
    model_cfg=OmegaConf.create(OmegaConf.to_container(source.model,resolve=True))
    model_cfg.num_layers=layers
    model_cfg.attention_hop_schedule=list(schedule)
    model_cfg.use_absolute_coordinate_embedding=True
    geometry=OmegaConf.load("configs/geometry/dinat_exact3.yaml" if 3 in schedule
                            else "configs/geometry/dinat_exact2.yaml")
    task=instantiate(source.task)
    adapter=_task_adapter(task)
    dataset=instantiate(source.data)
    index={sid:i for i,sid in enumerate(dataset.sample_ids)}
    ids=tuple(manifest["train_ids"][:EFFECTIVE_BATCH])
    tick=time.perf_counter()
    samples=[dataset[index[sid]] for sid in ids]
    read_seconds=time.perf_counter()-tick
    collator=GraphTaskCollator(task,dataset.field_catalog,geometry)
    tick=time.perf_counter()
    first=collator(samples[:1])
    topology_seconds=time.perf_counter()-tick
    edges={"local":first.edge_index.shape[1],**{k:v.shape[1] for k,v in first.attention_edge_indices.items()}}
    if first.inputs.shape!=(35937,5) or edges["local"]!=209088 or edges["dilated"]!=608058:
        raise ValueError("unexpected real M49 node/edge counts")
    if 3 in schedule and edges.get("dilated_3")!=1243828:
        raise ValueError("real M49 third-hop count differs from qualified Cartesian reference")
    del first
    report={"protocol":"m49_depth_real_accumulated_v1","status":"running",
            "provenance":provenance,"variant":args.variant,"layers":layers,"schedule":list(schedule),
            "absolute_coordinate_embedding":True,"nodes_per_graph":35937,"edges_per_graph":edges,
            "effective_batch":EFFECTIVE_BATCH,"gpu_total_gib":total,
            "train_ids":ids,"dataset_read_seconds":read_seconds,
            "first_collation_topology_seconds":topology_seconds,
            "memory_gate":{"allocated_fraction":.85,"reserved_fraction":.90},
            "trials":[],"selected":None}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    write(args.output,report)
    for batch_size,checkpointing in CANDIDATES+FALLBACKS:
        if batch_size==2 and choose_trial(report["trials"],total) is not None:
            break
        row=trial(batch_size=batch_size,checkpointing=checkpointing,samples=samples,
            collator=collator,task=task,standardizers=standardizers,source=source,
            model_cfg=model_cfg,device=device,adapter=adapter)
        report["trials"].append(row)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{args.variant}: {row}",flush=True)
        write(args.output,report)
    report["selected"]=choose_trial(report["trials"],total)
    report["status"]="ok" if report["selected"] is not None else "no_safe_config"
    write(args.output,report)
    print(f"Report={args.output} status={report['status']}",flush=True)


def selection(path,variant):
    report=json.loads(path.read_text())
    runtime=collect_runtime_provenance()
    current=runtime["git"]
    measured=report["provenance"]["git"]
    if (report["protocol"]!="m49_depth_real_accumulated_v1" or report["variant"]!=variant
        or report["status"]!="ok" or not report["absolute_coordinate_embedding"]
        or not measured["sha"] or measured["dirty"] is not False
        or not current["sha"] or current["dirty"] is not False):
        raise ValueError("incompatible or unqualified benchmark report")
    if measured["sha"] != current["sha"]:
        print(f"[M49] Reusing benchmark from {measured['sha']} on {current['sha']}; "
              "memory and throughput are historical measurements, not requalified "
              "for the current revision.", file=sys.stderr, flush=True)
    if runtime["runtime"] != report["provenance"]["runtime"] or runtime["hardware"] != report["provenance"]["hardware"]:
        raise ValueError("training runtime/hardware differs from benchmark")
    total=torch.cuda.get_device_properties("cuda").total_memory/1024**3
    if abs(total-report["gpu_total_gib"])>1e-6:
        raise ValueError("training GPU memory differs from benchmark")
    row=choose_trial(report["trials"],report["gpu_total_gib"])
    if row is None or row!=report["selected"]:
        raise ValueError("no safe reproducible benchmark selection")
    b,a,ac=row["microbatch"],row["accumulation"],row["activation_checkpointing"]
    if b not in (2,4,8) or b*a!=EFFECTIVE_BATCH or type(ac) is not bool:
        raise ValueError("unexpected benchmark-selected batch policy")
    layers,schedule=VARIANTS[variant]
    if report["layers"]!=layers or report["schedule"]!=list(schedule):
        raise ValueError("benchmark architecture mismatch")
    # Single machine-readable line for the Slurm launcher.
    print(layers, "exact3" if 3 in schedule else "exact2", b, a, str(ac).lower())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant",choices=tuple(VARIANTS),required=True)
    p.add_argument("--baseline",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--select",type=Path)
    args=p.parse_args()
    if args.select is not None:
        selection(args.select,args.variant)
    else:
        if args.baseline is None or args.output is None:
            p.error("benchmark requires --baseline and --output")
        benchmark(args)


if __name__=="__main__":
    main()
