"""M47.8B setup: frozen M47.7 checkpoint, one validation case, no training."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, HITCaseSpec
from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from graph_attention.tasks import FlowMatchingTask
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m47_5_regime_diagnostics import ANCHOR_CASE
from scripts.m47_6_spatial_metrics import physical_velocity, scatter_unique
from scripts.m47_8_response_audit import CHANNELS, _verify_mode
from scripts.m47_8b_trajectory_math import spectrum_summary
from scripts.train_generative import _load_standardizers
from scripts.train_m43_b_multires import FIELDS, build_recipe


def load_context(args) -> SimpleNamespace:
    """Read-only loader using the scientific contract of M47.7B evaluation."""
    signature = json.loads((args.train_dir / "run_signature.json").read_text())
    split = json.loads((args.train_dir / "split.json").read_text())
    checkpoint = torch.load(
        args.train_dir / f"epoch{args.epoch:02d}.pt",
        map_location="cpu", weights_only=True,
    )
    contract = signature["content"]
    if (checkpoint["signature_sha256"] != signature["sha256"]
        or checkpoint["epoch"] != args.epoch - 1
        or len(checkpoint["history"]) != args.epoch
        or contract["case"] != ANCHOR_CASE or split["case"] != ANCHOR_CASE
        or tuple(split["train_ids"]) != tuple(contract["train_ids"])
        or tuple(split["validation_ids"]) != tuple(contract["validation_ids"])
        or len(split["validation_ids"]) != 5
        or contract["source_distribution"] != "independent_unit_Gaussian_standardized"):
        raise ValueError("M47.7B checkpoint/split/signature mismatch")
    sid = split["validation_ids"][args.validation_index]
    data = AVBPMultiCaseHITDataset(
        root=args.archive_root, cases=(HITCaseSpec(ANCHOR_CASE, 65),),
        reference_dir=args.reference_dir, require_references=True,
        expected_snapshots_per_case=100, selected_sample_ids=(sid,),
    )
    if tuple(data.sample_ids) != (sid,):
        raise ValueError("frozen validation snapshot unavailable")
    task = FlowMatchingTask(
        state_fields=FIELDS, conditioning_parameters=(),
        physical_nondimensionalization=True, validation_seed=1234,
        time_embedding_scale=1000.0,
    )
    collator = GraphTaskCollator(
        task, data.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    sample = data[0]
    grid = infer_cartesian_grid_3d(
        sample.mesh.coords.detach().cpu().numpy(), periodic_endpoint_mode="drop_max",
    )
    for mode in args.modes:
        _verify_mode(mode, grid.shape)
    device = torch.device("cuda:0")
    scalers = _load_standardizers(args.train_dir / "standardizers.pt")
    if (not scalers.physical_nondimensionalization
        or scalers.weighting != "sample_balanced"
        or tuple(scalers.train_sample_ids) != tuple(split["train_ids"])
        or tuple(c.split(".")[0] for c in scalers.inputs.channel_names) != CHANNELS):
        raise ValueError("M47.7B train-only scaler provenance mismatch")
    scalers = scalers.to(device=device, dtype=torch.float32)
    batch = scalers.transform(task_batch_to_device(
        collator([sample]), device=device, dtype=torch.float32,
    ))
    probe = task.make_validation_problem(batch)
    model, _ = instantiate_controlled_model(
        build_recipe(), probe, seed=contract["model_initialization_seed"],
    )
    model = model.to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    q_dns = scalers.inputs.inverse(batch.inputs, batch.input_channels)
    rho = q_dns[:, 0]
    if not bool(torch.isfinite(rho).all() and (rho > 0).all()):
        raise ValueError("invalid DNS reference density")
    fixed_density = scatter_unique(q_dns.detach().cpu().numpy(), grid)[..., 0]
    uref = float(data.case_definitions[ANCHOR_CASE].reference_scales["U_ref"].value)
    velocity = physical_velocity(q_dns.detach().cpu().numpy(), grid, uref)
    reference_rms = float(np.sqrt(np.mean(
        (velocity - velocity.mean(axis=(0, 1, 2), keepdims=True)) ** 2,
    )))
    if reference_rms <= 0:
        raise ValueError("DNS velocity reference fluctuations must be positive")
    return SimpleNamespace(
        sample_id=sid, training_revision=contract["code_revision"],
        signature_sha256=signature["sha256"], model=model, batch=batch,
        task=task, scalers=scalers, device=device, grid=grid, dns_rho=rho,
        fixed_density=fixed_density, u_ref=uref, reference_rms=reference_rms,
        dns_spectrum=spectrum_summary(velocity, grid),
    )
