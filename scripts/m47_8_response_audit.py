"""M47.8A: read-only cross-channel Fourier response of frozen M47.7B.

Probe the learned straight-path FM velocity at fixed DNS/Gaussian interpolants.
Perturb one physical velocity component at a time with +/- cos and +/- sin,
converting the perturbation into standardized conserved momentum at FIXED DNS
reference density. The measured output is equivalent momentum-rate velocity
(U_ref * d(rho*u)/dt / rho_DNS), NOT the actual velocity derivative at x_t.
No gradients, training, projection, checkpoint writes or sealed-test reads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from graph_attention.data.multicase_avbp import AVBPMultiCaseHITDataset, HITCaseSpec
from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from graph_attention.tasks import FlowMatchingTask
from graph_attention.tasks.flow_matching import _deterministic_sampling_source, _model_velocity
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from graph_attention.training.model_factory import instantiate_controlled_model
from scripts.m47_5_regime_diagnostics import ANCHOR_CASE
from scripts.m47_6_spatial_metrics import physical_velocity, scatter_unique
from scripts.m47_7_covariance_metrics import spectral_tensor_audit
from scripts.m47_8_response_math import mode_leakage, response_geometry, transfer_column
from scripts.train_generative import _load_standardizers
from scripts.train_m43_b_multires import FIELDS, build_recipe

PROTOCOL = "m47_8a_frozen_cross_channel_response_v2"
CHANNELS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def _mode_arg(text: str) -> tuple[int, int, int]:
    try:
        raw = tuple(int(v) for v in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mode must be ix,iy,iz") from exc
    if len(raw) != 3 or not any(raw):
        raise argparse.ArgumentTypeError("mode must contain three integers and be nonzero")
    return raw


def _verify_mode(mode: tuple[int, int, int], side: tuple[int, int, int]) -> None:
    if any(2 * abs(m) >= n for m, n in zip(mode, side)):
        raise ValueError("probe modes must be strictly below the FFT Nyquist index")


def _periodic_carriers(coords: torch.Tensor, mode: tuple[int, int, int],
                       grid) -> tuple[tuple[torch.Tensor, torch.Tensor], float]:
    """Create physical Fourier modes using the *nondimensional* model coordinates.

    The M47 task replaces physical coordinates with x/L_ref. Therefore
    multiplying model coords by physical rad/m wavenumbers is incorrect.
    The normalized periodic-box coordinate is independent of L_ref.
    Check the actual packed node-order FFT before every perturbation.
    """
    if grid.periodic_endpoint_mode != "drop_max" or any(
        full != unique + 1 for full, unique in zip(grid.source_shape, grid.shape)
    ):
        raise ValueError("Fourier probe requires endpoint-inclusive periodic mesh")
    if coords.ndim != 2 or coords.shape != (
        int(np.prod(grid.source_shape)), 3
    ) or not torch.isfinite(coords).all():
        raise ValueError("model coords must be finite [N,3] for this periodic mesh")
    minimum = coords.amin(dim=0)
    extent = coords.amax(dim=0) - minimum
    if not bool(torch.isfinite(extent).all()) or not bool((extent > 0).all()):
        raise ValueError("degenerate physical box extent in model coords")
    box_fraction = (coords - minimum) / extent
    phase = (2 * np.pi) * (box_fraction * coords.new_tensor(mode)).sum(dim=1)
    carriers = torch.cos(phase), torch.sin(phase)
    leakage = []
    for carrier in carriers:
        field = np.zeros((*grid.shape, 3), dtype=np.float64)
        field[..., 0] = scatter_unique(carrier.detach().cpu().numpy(), grid)
        leak = mode_leakage(field, mode)
        leakage.append(leak)
        if not np.isfinite(leak) or leak > 1e-6:
            raise ValueError(
                f"injected carrier is not Fourier mode {mode}; "
                f"FFT off-mode leakage={leak:.8g}; check mesh indexing and units"
            )
    return carriers, float(max(leakage))


def _to_momentum_rate_velocity(velocity_std: torch.Tensor,
                               rho_dns: torch.Tensor, scales: torch.Tensor,
                               u_ref: float) -> np.ndarray:
    # Fixed-density momentum-to-velocity equivalent, not d(u)/dt at x_t.
    return (u_ref * velocity_std[:, 1:4] * scales[1:4] /
            rho_dns[:, None]).detach().cpu().numpy()


def _response(model: torch.nn.Module, batch, state: torch.Tensor,
              time: float, rho: torch.Tensor,
              scale: torch.Tensor, u_ref: float, grid,
              mode: tuple[int, int, int], relative_amplitude: float,
              physical_rms_m_per_s: float) -> dict:
    """Central finite differences; each column is one physical input channel."""
    amplitude = physical_rms_m_per_s * relative_amplitude
    lengths = tuple(float(n * dx) for n, dx in zip(grid.shape, grid.spacing))
    carriers, input_mode_leakage = _periodic_carriers(batch.coords, mode, grid)
    t = torch.full((batch.num_graphs,), time, device=state.device, dtype=state.dtype)
    matrix = np.zeros((3, 3), dtype=np.complex128)
    leakages = []
    for input_component in range(3):
        responses = []
        input_amplitude = (rho * (amplitude / u_ref) / scale[input_component + 1])
        for carrier in carriers:
            delta = input_amplitude * carrier
            plus = state.clone()
            minus = state.clone()
            plus[:, input_component + 1] += delta
            minus[:, input_component + 1] -= delta
            positive = _model_velocity(model, batch, plus, t, time_embedding_scale=1000.0)
            negative = _model_velocity(model, batch, minus, t, time_embedding_scale=1000.0)
            diff = 0.5 * (
                _to_momentum_rate_velocity(positive, rho, scale, u_ref) -
                _to_momentum_rate_velocity(negative, rho, scale, u_ref)
            )
            unique = scatter_unique(diff, grid)
            responses.append(unique)
            leakages.append(mode_leakage(unique, mode))
        matrix[:, input_component] = transfer_column(responses[0], responses[1], mode, amplitude)
    stats = response_geometry(matrix, mode, lengths)
    return {
        "mode": list(mode),
        "k_rad_per_m": float(np.linalg.norm([2 * np.pi * n / length for n, length in zip(mode, lengths)])),
        "time": time,
        "relative_amplitude": relative_amplitude,
        "amplitude_m_per_s": amplitude,
        "input_mode_leakage_fraction_max": input_mode_leakage,
        "response_leakage_fraction_mean": float(np.mean(leakages)),
        **stats,
    }


def _estimate_diagnostics(estimate: torch.Tensor, target: torch.Tensor,
                          scalers, prepared, grid, u_ref: float) -> dict:
    squared_error = (estimate - target).square().mean(dim=0)
    q = scalers.inputs.inverse(estimate, prepared.input_channels).detach().cpu().numpy()
    rho_unique = scatter_unique(q, grid)[..., 0]
    invalid = float(np.mean(~np.isfinite(rho_unique) | (rho_unique <= 0)))
    result = {
        "standardized_clean_estimate_mse_by_channel": dict(zip(CHANNELS, squared_error.detach().cpu().tolist())),
        "invalid_unique_density_fraction": invalid,
        "longitudinal_fraction_if_valid": None,
        "cross_cancellation_if_valid": None,
    }
    if invalid == 0 and np.isfinite(q).all():
        s = spectral_tensor_audit(physical_velocity(q, grid, u_ref), grid)
        result["longitudinal_fraction_if_valid"] = s["longitudinal_energy_fraction_integrated"]
        result["cross_cancellation_if_valid"] = s["cross_cancellation_integrated"]
    return result


def _sample_control(model, task, prepared, scalers, grid, u_ref: float,
                    steps: tuple[int, ...], seed: int) -> list[dict]:
    key = f"M47.2/{ANCHOR_CASE}/replicate_0000"
    output = []
    for count in steps:
        z = task.sample_standardized(
            model, prepared, steps=count, method="flow_ode", solver="heun",
            sampling_eps=0.0, final_denoise=False,
            sampling_seed=seed, sampling_keys=(key,),
        )
        q = scalers.inputs.inverse(z, prepared.input_channels).detach().cpu().numpy()
        rho = scatter_unique(q, grid)[..., 0]
        invalid = float(np.mean(~np.isfinite(rho) | (rho <= 0)))
        row = {"heun_steps": count, "sampling_seed": seed, "sampling_key": key,
               "invalid_unique_density_fraction": invalid,
               "longitudinal_fraction_if_valid": None,
               "cross_cancellation_if_valid": None}
        if invalid == 0 and np.isfinite(q).all():
            spectrum = spectral_tensor_audit(physical_velocity(q, grid, u_ref), grid)
            row["longitudinal_fraction_if_valid"] = spectrum["longitudinal_energy_fraction_integrated"]
            row["cross_cancellation_if_valid"] = spectrum["cross_cancellation_integrated"]
        output.append(row)
        print(f"[M47.8A] sampling Heun={count} fL={row['longitudinal_fraction_if_valid']}", flush=True)
    return output


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("M47.8A full-field response audit requires CUDA")
    if args.epoch not in (10, 30) or args.seed < 0:
        raise ValueError("epoch must be 10 or 30, seed nonnegative")
    if not args.times or any(t <= 0 or t >= 1 for t in args.times):
        raise ValueError("interpolation times must be strictly inside (0,1)")
    if not args.amplitudes or any(e <= 0 or e >= 1 for e in args.amplitudes):
        raise ValueError("amplitude fractions must lie in (0,1)")
    if len(set(args.modes)) != len(args.modes):
        raise ValueError("duplicate probe mode")
    if any(s <= 0 for s in args.sampler_steps):
        raise ValueError("sampler steps must be positive")
    origin = json.loads((args.train_dir / "run_signature.json").read_text())
    split = json.loads((args.train_dir / "split.json").read_text())
    saved = torch.load(args.train_dir / f"epoch{args.epoch:02d}.pt", map_location="cpu", weights_only=True)
    if (saved["signature_sha256"] != origin["sha256"] or
        int(saved["epoch"]) != args.epoch - 1 or
        origin["content"]["case"] != ANCHOR_CASE or
        split["case"] != ANCHOR_CASE or
        tuple(split["validation_ids"]) != tuple(origin["content"]["validation_ids"])):
        raise ValueError("checkpoint, signature or frozen validation membership mismatch")
    if origin["content"]["source_distribution"] != "independent_unit_Gaussian_standardized":
        raise ValueError("unexpected FM source")
    if args.validation_index < 0 or args.validation_index >= len(split["validation_ids"]):
        raise ValueError("validation index outside frozen membership")
    sid = split["validation_ids"][args.validation_index]
    data = AVBPMultiCaseHITDataset(
        root=args.archive_root, cases=(HITCaseSpec(ANCHOR_CASE, 65),),
        reference_dir=args.reference_dir, require_references=True,
        expected_snapshots_per_case=100, selected_sample_ids=(sid,),
    )
    if tuple(data.sample_ids) != (sid,):
        raise ValueError("validation DNS sample not found")
    task = FlowMatchingTask(state_fields=FIELDS, conditioning_parameters=(),
                            physical_nondimensionalization=True,
                            validation_seed=1234, time_embedding_scale=1000.0)
    collator = GraphTaskCollator(
        task, data.field_catalog,
        {"attention_edge_indices": {"dilated": "exact_two_hop"}},
    )
    sample = data[0]
    grid = infer_cartesian_grid_3d(sample.mesh.coords.detach().cpu().numpy(),
                                   periodic_endpoint_mode="drop_max")
    for mode in args.modes:
        _verify_mode(mode, grid.shape)
    device = torch.device("cuda:0")
    scalers = _load_standardizers(args.train_dir / "standardizers.pt").to(device=device, dtype=torch.float32)
    if not scalers.physical_nondimensionalization or (
        tuple(scalers.train_sample_ids) != tuple(split["train_ids"]) or
        tuple(name.split(".")[0] for name in scalers.inputs.channel_names) != CHANNELS
    ):
        raise ValueError("train-only scaler provenance mismatch")
    prepared = scalers.transform(task_batch_to_device(collator([sample]), device=device, dtype=torch.float32))
    probe = task.make_validation_problem(prepared)
    model, _ = instantiate_controlled_model(
        build_recipe(), probe, seed=origin["content"]["model_initialization_seed"],
    )
    model = model.to(device=device, dtype=torch.float32)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    u_ref = float(data.case_definitions[ANCHOR_CASE].reference_scales["U_ref"].value)
    q_ref = scalers.inputs.inverse(prepared.inputs, prepared.input_channels)
    rho = q_ref[:, 0]
    if bool((rho <= 0).any()):
        raise ValueError("DNS reference density must be strictly positive")
    phys_ref = physical_velocity(q_ref.detach().cpu().numpy(), grid, u_ref)
    physical_rms = float(np.sqrt(np.mean((phys_ref - phys_ref.mean(axis=(0, 1, 2), keepdims=True)) ** 2)))
    if physical_rms <= 0:
        raise ValueError("zero DNS velocity fluctuations")
    source = _deterministic_sampling_source(prepared, args.seed, (f"M47.8/{sid}",))
    ref_cov = spectral_tensor_audit(phys_ref, grid)
    responses, clean_estimates = [], []
    for time in args.times:
        state = (1 - time) * source + time * prepared.inputs
        times = torch.full((prepared.num_graphs,), time, device=device, dtype=state.dtype)
        prediction = _model_velocity(model, prepared, state, times, time_embedding_scale=1000.0)
        est = state + (1 - time) * prediction
        clean_estimates.append({"time": time, **_estimate_diagnostics(
            est, prepared.inputs, scalers, prepared, grid, u_ref,
        )})
        for mode in args.modes:
            for eps in args.amplitudes:
                row = _response(model, prepared, state, time, rho,
                                scalers.inputs.scale, u_ref, grid,
                                mode, eps, physical_rms)
                responses.append(row)
                print(f"[M47.8A] t={time:.2f} mode={mode} eps={eps:.3g} "
                      f"gap={row['long_trans_gap_relative']:.3g} "
                      f"offdiag={row['offdiagonal_relative_frobenius']:.3g}", flush=True)
    sampler = _sample_control(model, task, prepared, scalers, grid, u_ref,
                              tuple(args.sampler_steps), args.seed) if args.sampler_steps else []
    result = {
        "protocol": PROTOCOL, "status": "complete", "training_signature_sha256": origin["sha256"],
        "training_revision": origin["content"]["code_revision"],
        "audit_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(args.train_dir / f"epoch{args.epoch:02d}.pt"),
        "validation_sample": sid, "test_data_accessed": False,
        "seed": args.seed, "time_points": list(args.times),
        "wavevectors_indices": [list(x) for x in args.modes],
        "relative_perturbation_amplitudes": list(args.amplitudes),
        "dns_component_velocity_rms_m_per_s": physical_rms,
        "dns_longitudinal_fraction": ref_cov["longitudinal_energy_fraction_integrated"],
        "reference_scaling": "U_ref, inverse train-only scaler, fixed DNS density",
        "response_definition": "d(U_ref*d(rho*u)/dt / rho_DNS)/d(delta physical u); fixed rho_DNS; NOT instantaneous physical velocity derivative at interpolated Gaussian state",
        "mode_transfer_definition": "complex +k diagonal Fourier response estimated by central +/- perturbations for cos and sin; offmode power measured separately",
        "coordinate_convention": "model coordinates are nondimensionalized x/L_ref; input carriers use (coords-min)/(max-min) times 2pi integer modes and must pass physical FFT purity check",
        "v1_correction": "v1 incorrectly multiplied nondimensional coordinates by physical rad/m wavenumbers; v1 response matrices are invalid",
        "note": "A single DNS realization and finite perturbations yield local Jacobians, not a full ensemble-average covariance or proof of spatial invariance. Time endpoints are excluded.",
        "clean_state_estimates": clean_estimates,
        "responses": responses,
        "heun_control": sampler,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--archive-root", type=Path, default=Path("/archivenfs/phlusim/HIT_DNS_COMP"))
    p.add_argument("--reference-dir", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references"))
    p.add_argument("--epoch", type=int, choices=(10, 30), default=30)
    p.add_argument("--validation-index", type=int, default=0)
    p.add_argument("--times", type=float, nargs="+", default=(0.25, 0.5, 0.75))
    p.add_argument("--amplitudes", type=float, nargs="+", default=(0.04, 0.08))
    p.add_argument("--modes", type=_mode_arg, nargs="+", default=((2, 2, 0), (4, 3, 1), (8, 6, 5)))
    p.add_argument("--sampler-steps", type=int, nargs="*", default=())
    p.add_argument("--seed", type=int, default=20261116)
    result = run(p.parse_args())
    print(f"[M47.8A] complete: {len(result['responses'])} response matrices, "
          f"{len(result['heun_control'])} Heun controls", flush=True)


if __name__ == "__main__":
    main()
