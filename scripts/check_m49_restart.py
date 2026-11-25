"""Prepare a real-data restart smoke subset and compare epoch-boundary artifacts.

No held-out test fields are evaluated. The production training runner is used
unchanged apart from the restart shuffle-RNG fix.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from graph_attention.data.splits import load_explicit_split_manifest

SOURCE = Path("configs/splits/m49_hit_les_forced_blocked50.json")
NAMES = {
    "full": "m49_full_dit_fp32_h128_l10_h4_b8_a2_seed42_blocked50",
    "dinat": "m49_dinat_dit_fp32_h128_l10_h4_b8_a2_seed42_blocked50",
}
# Short numerical smoke thresholds, not a proof of universal CUDA determinism.
LOSS_RTOL = 1e-3
PARAM_RTOL = 1e-4
UPDATE_RTOL = 1e-2
MOMENT_RTOL = 1e-2


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def prepare(output):
    source = json.loads(SOURCE.read_text())
    inventory = tuple(source["train_ids"] + source["validation_ids"] +
                      source["test_ids"] + source["excluded_ids"])
    def spaced(values, n):
        if len(values) < n:
            raise ValueError("insufficient source samples")
        return [values[i * len(values) // n] for i in range(n)]
    result = {"policy": "M49 restart smoke only",
              "train_ids": spaced(source["train_ids"], 32),
              "validation_ids": spaced(source["validation_ids"], 8),
              "test_ids": spaced(source["test_ids"], 8)}
    active = set(result["train_ids"] + result["validation_ids"] + result["test_ids"])
    result["excluded_ids"] = sorted(set(inventory) - active)
    write(output, result)
    load_explicit_split_manifest(output, inventory, inventory)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def tensor_stats(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("tensor contract mismatch")
    a, b = a.double(), b.double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("nonfinite comparison tensor")
    d = b - a
    return dict(diff_sq=float(d.square().sum()), ref_sq=float(a.square().sum()),
                max_abs=float(d.abs().max()) if d.numel() else 0.0)


def aggregate(pairs):
    stats = [tensor_stats(a, b) for a, b in pairs]
    if not stats:
        raise ValueError("empty tensor comparison")
    diff = math.sqrt(sum(s["diff_sq"] for s in stats))
    ref = math.sqrt(sum(s["ref_sq"] for s in stats))
    return {"relative_l2": diff / max(ref, 1e-30), "absolute_l2": diff,
            "max_abs": max(s["max_abs"] for s in stats)}


def exact_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact_tree(x, y) for x, y in zip(a, b))
    return a == b


def compare_checkpoints(a, b, base_a=None, base_b=None):
    checks = {}
    for key in ("epoch", "global_step", "model_class", "path_seed",
                "training_generator_state", "train_loader_generator_state"):
        checks[key] = key in a and key in b and exact_tree(a[key], b[key])
    pa, pb = a["model_state_dict"], b["model_state_dict"]
    if pa.keys() != pb.keys():
        raise ValueError("model parameter key mismatch")
    parameters = aggregate((pa[k], pb[k]) for k in pa)
    oa, ob = a["optimizer_state_dict"], b["optimizer_state_dict"]
    checks["optimizer_param_groups"] = exact_tree(oa["param_groups"], ob["param_groups"])
    sa, sb = oa["state"], ob["state"]
    if sa.keys() != sb.keys():
        raise ValueError("optimizer state keys mismatch")
    checks["optimizer_steps"] = all(exact_tree(sa[k]["step"], sb[k]["step"]) for k in sa)
    checks["optimizer_steps_match_global_step"] = all(
        float(sa[k]["step"]) == a["global_step"] and float(sb[k]["step"]) == b["global_step"]
        for k in sa)
    moments = {m: aggregate((sa[k][m], sb[k][m]) for k in sa) for m in ("exp_avg", "exp_avg_sq")}
    ha, hb = a["history"], b["history"]
    checks["history_epochs"] = [r["epoch"] for r in ha] == [r["epoch"] for r in hb]
    if len(ha) != len(hb):
        raise ValueError("history length mismatch")
    losses = {}
    for key in ("train_flow_velocity_mse", "validation_flow_velocity_mse"):
        values = [(float(x[key]), float(y[key])) for x, y in zip(ha, hb)]
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in values):
            raise ValueError("nonfinite history")
        losses[key] = {"max_relative": max(abs(x-y)/max(abs(x), 1e-30) for x,y in values)}
    checks["parameter_tolerance"] = parameters["relative_l2"] <= PARAM_RTOL
    checks["moment_tolerance"] = all(v["relative_l2"] <= MOMENT_RTOL for v in moments.values())
    checks["loss_tolerance"] = all(v["max_relative"] <= LOSS_RTOL for v in losses.values())
    result = {"checks": checks, "parameters": parameters, "moments": moments, "losses": losses}
    if base_a is not None:
        da, db = base_a["model_state_dict"], base_b["model_state_dict"]
        update = aggregate((pa[k]-da[k], pb[k]-db[k]) for k in pa)
        checks["continuation_update_tolerance"] = update["relative_l2"] <= UPDATE_RTOL
        result["continuation_update"] = update
    result["passed"] = all(checks.values())
    return result


def compare(root, arch):
    report = {"architecture": arch, "status": "started",
              "thresholds": {"loss_relative": LOSS_RTOL, "parameter_relative_l2": PARAM_RTOL,
                             "continuation_update_relative_l2": UPDATE_RTOL, "moment_relative_l2": MOMENT_RTOL}}
    out = root / "restart_report.json"
    try:
        a = root / "reference" / "training" / NAMES[arch]
        b = root / "restarted" / "training" / NAMES[arch]
        manifest_a, manifest_b = [json.loads((d/"dataset_split_manifest.json").read_text()) for d in (a,b)]
        report["manifest_equal"] = all(manifest_a[k] == manifest_b[k] for k in
            ("sample_ids", "train_ids", "validation_ids", "test_ids", "excluded_ids"))
        report["scalers_equal"] = exact_tree(load(a/"standardizers.pt"), load(b/"standardizers.pt"))
        ca, cb = load(a/"epoch_0002.pt"), load(b/"epoch_0002.pt")
        report["epoch2_before_restart"] = compare_checkpoints(ca, cb)
        ea, eb = load(a/"last.pt"), load(b/"last.pt")
        report["epoch4_after_restart"] = compare_checkpoints(ea, eb, ca, cb)
        report["expected_updates"] = (ca["global_step"] == cb["global_step"] == 4 and
                                      ea["global_step"] == eb["global_step"] == 8 and
                                      ca["epoch"] == cb["epoch"] == 1 and ea["epoch"] == eb["epoch"] == 3)
        report["best_epoch"] = [int(load(d/"best.pt")["epoch"]) for d in (a,b)]
        report["status"] = "passed" if all((
            report["manifest_equal"], report["scalers_equal"], report["expected_updates"],
            report["epoch2_before_restart"]["passed"], report["epoch4_after_restart"]["passed"],
        )) else "failed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        write(out, report)
        print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--arch", choices=tuple(NAMES), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    else:
        compare(args.root, args.arch)
