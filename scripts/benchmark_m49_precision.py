"""Paired Full DiT and DiNAT FP32/BF16 AMP screening on actual numbered HIT snapshots.

GPU-side steps only; no long training, no test data, no precision changes to
the production runner. Both trajectories start from identical nonzero weights
and AdamW state after a short common FP32 warmup.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from graph_attention.data import AVBPDirectoryHDF5Dataset, make_grouped_split_manifest
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training import fit_train_standardizers
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from graph_attention.utils.provenance import collect_runtime_provenance
from scripts.probe_3d_max_batch import _forward_model, _model_and_geometry


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def vector(model, *, gradients=False):
    return torch.cat([
        (p.grad if gradients else p).detach().float().cpu().reshape(-1)
        for p in model.parameters()
    ])


def discrepancy(ref, other):
    a = ref.double().reshape(-1)
    b = other.double().reshape(-1)
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise RuntimeError("nonfinite comparison tensor")
    d = b - a
    an, bn = a.norm().item(), b.norm().item()
    return {
        "reference_l2": an, "candidate_l2": bn,
        "relative_l2": d.norm().item() / max(an, 1e-30),
        "rmse": d.square().mean().sqrt().item(),
        "max_abs": d.abs().max().item(),
        "cosine": float(torch.dot(a, b)) / (an * bn) if an and bn else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("full", "dinat"), default="full")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.batch < 1 or args.repeats < 1:
        p.error("positive batch and repeats required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 support required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {
        "status": "started", "provenance": collect_runtime_provenance(),
        "model": args.model,
        "backend": "sdpa" if args.model == "full" else "torch_sparse_triton",
        "batch": args.batch, "warmup_shared_fp32_steps": 3,
        "warmup_per_precision_steps": 2, "timed_steps": args.repeats,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "precision_policy": (
            "FP32 data, parameters, optimizer and loss; BF16 autocast model forward only"
            if args.model == "full" else
            "FP32 data, parameters, optimizer, loss and entire CSR geometric attention; "
            "BF16 autocast eligible non-attention operations and BF16 attention output"
        ),
        "snapshot_pattern": "solut_hit_*.h5",
    }

    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output)

    save()
    try:
        device = torch.device("cuda")
        ds = AVBPDirectoryHDF5Dataset(
            snapshot_dir="/scratch/coop/theret/HIT_LES_FORCED/RUN/SOLUT",
            mesh_file="/scratch/coop/theret/HIT_LES_FORCED/MESH/mesh.mesh.h5",
            snapshot_pattern=report["snapshot_pattern"],
            mesh_id="HIT_LES_FORCED:3d_fixed_mesh",
            case_file="cases/HIT_LES_FORCED.yaml", case_id="HIT_LES_FORCED",
        )
        ids = tuple(ds.sample_ids)
        split = make_grouped_split_manifest(ids, ids, seed=42, train_ratio=.7, validation_ratio=.15)
        if len(split.train_ids) < 3 * args.batch or not split.validation_ids:
            raise RuntimeError("insufficient train/validation snapshots")
        lookup = {name: i for i, name in enumerate(ids)}
        task = FlowMatchingTask(
            state_fields=("rho", "rhou", "rhov", "rhow", "rhoE"),
            physical_nondimensionalization=True, time_embedding_scale=1000.,
            validation_seed=1234,
        )
        cfg, geo = _model_and_geometry(args.model)
        if args.model == "dinat":
            cfg.sparse_attention_backend = "torch_sparse_triton"
        collate = GraphTaskCollator(task, ds.field_catalog, geo)
        scaler = fit_train_standardizers(task, (ds[lookup[i]] for i in split.train_ids),
                                        ds.field_catalog, split).to(device=device, dtype=torch.float32)
        def prepared(names):
            return scaler.transform(task_batch_to_device(
                collate([ds[lookup[i]] for i in names]), device=device, dtype=torch.float32))
        train_ids = split.train_ids[:3 * args.batch]
        batches = [prepared(train_ids[i:i+args.batch]) for i in range(0, len(train_ids), args.batch)]
        validation = task.make_validation_problem(prepared(split.validation_ids[:1]))
        report.update(train_count=len(split.train_ids), validation_count=len(split.validation_ids),
                      test_count=len(split.test_ids), selected_train_ids=list(train_ids),
                      validation_ids=list(split.validation_ids[:1]),
                      nodes_per_graph=int(batches[0].inputs.shape[0] // args.batch))
        save()
        def new_model():
            model, _ = instantiate_controlled_model(cfg, validation, seed=42)
            model = model.to(device=device, dtype=torch.float32)
            return model, torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.)

        def step(model, opt, problem, amp):
            model.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pred = _forward_model(model, problem)
            loss = task.training_loss(pred.float(), problem).mean
            loss.backward()
            opt.step()
            return pred.detach(), loss.detach()

        def generator(seed):
            return torch.Generator(device=device).manual_seed(seed)

        model, opt = new_model()
        rng = generator(1042)
        for batch in batches:
            _, loss = step(model, opt, task.make_training_problem(batch, generator=rng), False)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite shared warmup loss")
        common_weights = cpu_copy(model.state_dict())
        common_opt = cpu_copy(opt.state_dict())
        common_vector = vector(model)
        del model, opt, loss
        gc.collect()
        torch.cuda.empty_cache()

        # Both precisions see identical FP32 Flow Matching problems in identical order.
        comparison = task.make_training_problem(batches[0], generator=generator(2042))
        results = {}
        reference = None
        for mode in ("fp32", "bf16_amp"):
            amp = mode == "bf16_amp"
            model, opt = new_model()
            model.load_state_dict(common_weights)
            opt.load_state_dict(common_opt)
            pred, loss = step(model, opt, comparison, amp)
            first = dict(prediction=pred.float().cpu(), gradients=vector(model, gradients=True),
                         update=vector(model)-common_vector, loss=float(loss))
            if not all(bool(torch.isfinite(first[k]).all()) for k in ("prediction", "gradients", "update")):
                raise RuntimeError(f"nonfinite first step: {mode}")
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite first loss: {mode}")
            if reference is None:
                reference = first
            else:
                report["same_weights_parity"] = {
                    "prediction": discrepancy(reference["prediction"], first["prediction"]),
                    "gradient": discrepancy(reference["gradients"], first["gradients"]),
                    "adamw_update": discrepancy(reference["update"], first["update"]),
                    "fp32_loss": reference["loss"], "bf16_loss": first["loss"],
                    "loss_relative_abs": abs(first["loss"]-reference["loss"])/max(abs(reference["loss"]),1e-30),
                }
            del pred, loss, first
            rng = generator(3042)
            times, losses = [], []
            for i in range(2 + args.repeats):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                problem = task.make_training_problem(batches[i%len(batches)], generator=rng)
                pred, loss = step(model, opt, problem, amp)
                torch.cuda.synchronize()
                dt = time.perf_counter() - tick
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"nonfinite trajectory loss: {mode}")
                losses.append(float(loss))
                del pred, loss, problem
                if i == 1:
                    torch.cuda.reset_peak_memory_stats()
                if i >= 2:
                    times.append(dt)
            peak_allocated = torch.cuda.max_memory_allocated()/1024**3
            peak_reserved = torch.cuda.max_memory_reserved()/1024**3
            model.eval()
            with torch.no_grad():
                # Same FP32 evaluator isolates weight trajectory divergence.
                eval_pred = _forward_model(model, validation)
                val_loss = float(task.training_loss(eval_pred.float(), validation).mean)
            del eval_pred
            if not bool(torch.isfinite(torch.tensor(val_loss))):
                raise RuntimeError(f"nonfinite validation loss: {mode}")
            trajectory = vector(model)
            if not bool(torch.isfinite(trajectory).all()):
                raise RuntimeError(f"nonfinite weights: {mode}")
            rec = dict(seconds=times, median_step_seconds=statistics.median(times),
                       samples_per_second=args.batch/statistics.median(times),
                       peak_allocated_gib=peak_allocated, peak_reserved_gib=peak_reserved,
                       trajectory_train_losses=losses, fp32_validation_loss=val_loss)
            # Profile real aten dispatch outside the timing loop; do not count it as a measured step.
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    step(model, opt, comparison, amp)
                    torch.cuda.synchronize()
                rec["attention_operators"] = sorted({e.key for e in prof.key_averages()
                    if any(k in e.key.lower() for k in ("scaled_dot_product", "flash_attention", "efficient_attention", "cudnn_attention", "tritonsparsesddmm", "tritonsparsespmm", "sampled_addmm"))})
            except Exception as exc:
                rec["profiler_error"] = repr(exc)
            report[mode] = rec
            if mode == "fp32":
                results["fp32_trajectory"] = trajectory
            else:
                report["trajectory_delta"] = discrepancy(
                    results["fp32_trajectory"]-common_vector, trajectory-common_vector)
            save()
            del model, opt, trajectory
            gc.collect()
            torch.cuda.empty_cache()
        report["speedup_bf16_over_fp32"] = report["fp32"]["median_step_seconds"]/report["bf16_amp"]["median_step_seconds"]
        report["status"] = "ok"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
