"""M45.6 reproducibility and exact patch/full-owned optimizer diagnostics.

Use separate Python processes for independent trace replicas. All input paths,
patch selections, model seeds, and optimizer steps match the M45 pilot. The
same-process parity control compares identical owned-node objectives.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path

import torch

from graph_attention.data.two_scale_synthetic import (
    CASES, SyntheticQB, fit_synthetic_scales, make_synthetic_pair,
)
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from graph_attention.models.two_scale_dinat import TwoScaleDiNAT
from scripts.m45_synthetic_pilot import _patches, _path, _state, _to_device, _velocity


def _digest(tensors) -> str:
    digest = hashlib.sha256()
    for key, value in tensors:
        digest.update(str(key).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _maxdiff(a: torch.nn.Module, b: torch.nn.Module, *, gradients: bool) -> float:
    result = 0.0
    for (_, x), (_, y) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        lhs, rhs = (x.grad, y.grad) if gradients else (x.detach(), y.detach())
        if lhs is None and rhs is None:
            continue
        if lhs is None or rhs is None:
            return float("inf")
        result = max(result, float((lhs.detach() - rhs.detach()).abs().max()))
    return result


def _model(mode: str, seed: int, width: int, depth: int) -> TwoScaleDiNAT:
    kwargs = dict(coarse_width=width, fine_width=width, num_heads=4,
                  fine_pre_layers=depth, fine_post_layers=depth,
                  coarse_static_layers=depth, coarse_dynamic_layers=depth)
    torch.manual_seed(seed + 77)
    static = TwoScaleDiNAT(mode="static", **kwargs)
    if mode == "static":
        return static
    torch.manual_seed(seed + 77)
    dynamic = TwoScaleDiNAT(mode="dynamic", **kwargs)
    dynamic.load_state_dict({k: v for k, v in static.state_dict().items()
                             if k in dynamic.state_dict()}, strict=False)
    return dynamic


def _dataset(coarse_side: int, refinement: int, case: str,
             samples: int, seed: int, device: torch.device):
    mesh = build_hit_two_scale_geometry(coarse_side=coarse_side, refinement=refinement)
    qb = SyntheticQB(mesh, "block_average")
    train = [make_synthetic_pair(qb, case=case, sample_id=i,
                                 seed=seed + i * 3001) for i in range(samples)]
    return mesh, _to_device(mesh, device), qb, train, fit_synthetic_scales(train)


def _draws(train, patches, qb, scales, *, seed: int, steps: int):
    generator = torch.Generator(device="cpu").manual_seed(seed + 23000)
    sample_ids = torch.randint(len(train), (steps,), generator=generator)
    patch_ids = torch.randint(len(patches), (steps,), generator=generator)
    for step in range(steps):
        pair = train[int(sample_ids[step])]
        patch_cpu, patch_gpu = patches[int(patch_ids[step])]
        c, b, x1 = _state(pair, scales)
        x, target, t = _path(x1, seed + 500_000 + step * 997, qb)
        source_digest = _digest([("x", x), ("target", target), ("time", t)])
        yield step + 1, patch_cpu, patch_gpu, c, b, x, target, t, source_digest


def _loss(model, scope, mesh_gpu, patch_cpu, patch_gpu, c, b, x, target, t, device):
    if scope == "patch_owned":
        prediction = _velocity(model, x[patch_cpu.fine_global].to(device),
                               b[patch_cpu.fine_global].to(device),
                               c[patch_cpu.coarse_global].to(device),
                               patch_gpu, t.to(device))
        observed = target[patch_cpu.fine_global].to(device)
        owned = patch_gpu.owned_fine_local
    else:
        prediction = _velocity(model, x.to(device), b.to(device), c.to(device),
                               mesh_gpu, t.to(device))
        observed = target.to(device)
        owned = (patch_cpu.fine_global[patch_cpu.owned_fine_local].to(device)
                 if scope == "full_owned" else slice(None))
    return (prediction[owned] - observed[owned]).square().mean(), prediction[owned]


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no diagnostic rows")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _environment(device: torch.device) -> dict:
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def execute(*, task: str, output_dir: Path, cases: tuple[str, ...],
            modes: tuple[str, ...], depth: int, width: int, steps: int,
            train_samples: int, seed: int, coarse_side: int, refinement: int,
            core_side: int, device: torch.device, deterministic: str) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if task not in {"trace", "parity"}:
        raise ValueError("invalid task")
    torch.use_deterministic_algorithms(deterministic != "off", warn_only=deterministic == "warn")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic != "off"
    output_dir.mkdir(parents=True)
    result = {"schema": "graph_attention.m45_6_reproducibility.v1", "task": task,
              "settings": {"depth": depth, "width": width, "steps": steps,
                           "train_samples": train_samples, "seed": seed,
                           "coarse_side": coarse_side, "refinement": refinement,
                           "core_side": core_side, "cases": cases, "modes": modes,
                           "deterministic": deterministic},
              "environment": _environment(device), "runs": []}
    for case in cases:
        mesh, mesh_gpu, qb, train, scales = _dataset(coarse_side, refinement,
                                                      case, train_samples, seed, device)
        for mode in modes:
            model = _model(mode, seed, width, depth).to(device)
            patches = _patches(mesh, model, core_side, device)
            records = []
            initial_hash = _digest(model.named_parameters())
            if task == "trace":
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                for step, cpu, gpu, c, b, x, target, t, path_hash in _draws(
                        train, patches, qb, scales, seed=seed, steps=steps):
                    model.train()
                    loss, _ = _loss(model, "patch_owned", mesh_gpu, cpu, gpu,
                                    c, b, x, target, t, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_before_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                    optimizer.step()
                    records.append({"step": step, "path_sha256": path_hash,
                                    "loss": float(loss), "grad_norm_before_clip": grad_before_clip,
                                    "weights_sha256": _digest(model.named_parameters())})
            else:
                full = copy.deepcopy(model)
                optimizer_patch = torch.optim.AdamW(model.parameters(), lr=1e-3)
                optimizer_full = torch.optim.AdamW(full.parameters(), lr=1e-3)
                for step, cpu, gpu, c, b, x, target, t, path_hash in _draws(
                        train, patches, qb, scales, seed=seed, steps=steps):
                    model.train()
                    full.train()
                    lp, yp = _loss(model, "patch_owned", mesh_gpu, cpu, gpu,
                                   c, b, x, target, t, device)
                    lf, yf = _loss(full, "full_owned", mesh_gpu, cpu, gpu,
                                   c, b, x, target, t, device)
                    optimizer_patch.zero_grad(set_to_none=True)
                    optimizer_full.zero_grad(set_to_none=True)
                    lp.backward()
                    lf.backward()
                    gradient_gap = _maxdiff(model, full, gradients=True)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0)
                    optimizer_patch.step()
                    optimizer_full.step()
                    records.append({"step": step, "path_sha256": path_hash,
                                    "patch_loss": float(lp), "full_owned_loss": float(lf),
                                    "abs_loss_diff": abs(float(lp) - float(lf)),
                                    "max_owned_prediction_diff": float((yp - yf).abs().max()),
                                    "max_grad_diff": gradient_gap,
                                    "max_weight_diff": _maxdiff(model, full, gradients=False)})
            csv_path = f"{case}_{mode}_{task}.csv"
            _write_rows(output_dir / csv_path, records)
            result["runs"].append({"case": case, "mode": mode,
                                   "initial_sha256": initial_hash, "trace_file": csv_path,
                                   "final": records[-1]})
            print(f"{case} {mode} {task}: {records[-1]}", flush=True)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare(left: Path, right: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    a, b = [json.loads((path / "summary.json").read_text()) for path in (left, right)]
    if a["task"] != "trace" or b["task"] != "trace" or a["settings"] != b["settings"]:
        raise ValueError("expected matching, independent trace runs")
    output_dir.mkdir(parents=True)
    report = {"schema": "graph_attention.m45_6_compare.v1", "settings": a["settings"],
              "left_environment": a["environment"], "right_environment": b["environment"],
              "comparisons": []}
    for entry in a["runs"]:
        rhs = next(x for x in b["runs"] if x["case"] == entry["case"] and x["mode"] == entry["mode"])
        with (left / entry["trace_file"]).open() as fa, (right / rhs["trace_file"]).open() as fb:
            aa, bb = list(csv.DictReader(fa)), list(csv.DictReader(fb))
        if len(aa) != len(bb) or entry["initial_sha256"] != rhs["initial_sha256"]:
            raise ValueError("replicas have unequal training steps or initial parameters")
        if any(x["path_sha256"] != y["path_sha256"] for x, y in zip(aa, bb)):
            raise ValueError("replicas have different input/noise paths")
        first = next((int(x["step"]) for x, y in zip(aa, bb)
                      if x["weights_sha256"] != y["weights_sha256"]), None)
        max_loss = max(abs(float(x["loss"]) - float(y["loss"]))
                       for x, y in zip(aa, bb))
        report["comparisons"].append({"case": entry["case"], "mode": entry["mode"],
                                      "steps": len(aa), "bitwise_identical": first is None,
                                      "first_divergent_weight_step": first,
                                      "max_abs_loss_diff": max_loss})
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("trace", "parity", "compare"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=["local", "shared_mode"])
    parser.add_argument("--modes", nargs="+", choices=("static", "dynamic"), default=["static", "dynamic"])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--coarse-side", type=int, default=4)
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--core-side", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--deterministic", choices=("off", "warn", "strict"), default="off")
    args = parser.parse_args()
    if args.task == "compare":
        if args.left is None or args.right is None:
            parser.error("compare requires --left and --right")
        report = compare(args.left, args.right, args.output_dir)
        print(json.dumps(report, indent=2))
        return
    if min(args.depth, args.width, args.steps, args.train_samples,
           args.coarse_side, args.refinement, args.core_side) < 1:
        parser.error("positive sizes required")
    if args.width % 4:
        parser.error("width must be divisible by 4 heads")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    execute(task=args.task, output_dir=args.output_dir, cases=tuple(args.cases),
            modes=tuple(args.modes), depth=args.depth, width=args.width,
            steps=args.steps, train_samples=args.train_samples, seed=args.seed,
            coarse_side=args.coarse_side, refinement=args.refinement,
            core_side=args.core_side, device=torch.device(args.device),
            deterministic=args.deterministic)


if __name__ == "__main__":
    main()
