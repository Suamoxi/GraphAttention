# M47.8 — Frozen M47.7B spectral cross-channel response audit

## Motivation and existing controls

- **M47.3** already measured per-channel Flow Matching velocity MSE at several times and recorded 50-step Heun trajectories. It did **not** resolve the wavevector-dependent 3x3 momentum response.
- **M45.6/M45.7** already performed conditional Gaussian-oracle/Heun controls for **super-resolution residuals**. These are not an unconditional full-field covariance-transport result.
- **M47.7B** epoch 30, Re54/Ma0.2, reports DNS longitudinal fraction 0.0187313, generated 0.3252221, with cross-term cancellation 0.90234 versus -0.00076.

The first M47.8 target is therefore **not to train a new model**. It is to interrogate the frozen epoch-30 DiNAT-DiT vector field, the same reference standardizer and the same five validation DNS snapshots. No test snapshots are used.

## Probe: what each response matrix means

At a fixed straight-path interpolation of **one DNS validation sample** and an independent seeded Gaussian, inject a small sinusoidal velocity perturbation in one Cartesian direction j at physical wavevector k. Keep the DNS density fixed when converting this perturbation into nondimensional conserved momentum, and then into the original standardized input channel:

`delta_x_standardized,rho_u_j = rho_DNS * (epsilon_m_per_s/U_ref) * cos(k.x) / standardizer.scale[rho_u_j]`.

Evaluate the **frozen** FM vector field at plus and minus perturbations. Convert only the predicted **momentum** components to the fixed-density velocity-equivalent rate:

`g_i(x,t) = U_ref * standardizer.scale[rho_u_i] * v_theta,rho_u_i(x,t) / rho_DNS(x)`.

Importantly, this is **not** d(u_i)/dt at the Gaussian/data interpolated state: that state may not correspond to physically admissible positive density, and a true physical velocity derivative would also include a density-velocity product term. The measurement is a controlled momentum-channel transfer coefficient with physically comparable amplitudes.

Run both real perturbations `cos(k.x)` and `sin(k.x)` in order to isolate the complex positive-k Fourier response (a single real perturbation cannot do so for a position-dependent nonlinear model). With central differences `delta_g_cos=(g_plus-g_minus)/2` and similarly for sin, the transfer matrix column is

`J[:,j](k,t) = [ FFT(delta_g_cos)[+k] + i*FFT(delta_g_sin)[+k] ] / epsilon`

with forward-normalized FFT. Report the complex 3x3 matrix, its off-diagonal norm, the difference between longitudinal and transverse responses and its relative deviation from a fitted `a_T P_T+a_L P_L`. Report **off-mode Fourier energy leakage** separately rather than assuming translation invariance.

These matrices are **local linear responses**, not measured output covariances. An off-diagonal response is necessary for a general direction-dependent spectral projector but is not sufficient to reproduce the DNS covariance. Even an exact Gaussian transport can have a nearly diagonal Jacobian close to t=0 or t=1; the informative times are strictly interior.

## Run on Calypso (GPU required)

From the GraphAttention repository root, activate its existing working Python environment, then:

```bash
git pull --rebase origin main
python -m pytest -q tests/unit/test_m47_8_response.py tests/unit/test_m47_8_gaussian_oracle.py tests/unit/test_m47_7_covariance.py
TRAIN=/scratch/coop/theret/GraphAttention_runs/m47_7_single_regime_training_fast_v1
OUT=/scratch/coop/theret/GraphAttention_runs/m47_8a_cross_channel_epoch30_v1
python -m scripts.m47_8_response_audit \
  --train-dir "$TRAIN" \
  --epoch 30 \
  --validation-index 0 \
  --times 0.25 0.5 0.75 \
  --modes 2,2,0 4,3,1 8,6,5 \
  --amplitudes 0.04 0.08 \
  --sampler-steps 25 50 100 \
  --output-dir "$OUT"
```

The command **must execute on a Calypso GPU allocation**; it is not an interactive-login-node workload. To compare with additional frozen validation DNS, choose a new output path and `--validation-index 1` (through 4). The training split and code revision are checked from `run_signature.json`; checkpoint weights, split, scalers and DNS files are read only. A pre-existing output directory is rejected. The audit forward uses the established `torch_sparse` implementation, as did the frozen M47.7 evaluation; the original accelerated training checkpoint used `torch_sparse_triton`.

Audit output: `$OUT/audit.json`, containing 3 time points x 3 Fourier wavevectors x 2 perturbation amplitudes = **18 complex response matrices**, plus three clean-state estimate diagnostics, and an optional fixed-source Heun control sweep. Matrix units are dimensionless (physical velocity-equivalent rate per physical input velocity, and the FM time coordinate is dimensionless).

The two epsilon values are a finite-difference linearity check. A large disagreement between their matrices invalidates the linear-response approximation for that mode/time and motivates smaller perturbations or float64 diagnostics.

## Independent full-field Gaussian oracle (CPU)

To demonstrate that an independent Gaussian source is not a fundamental mathematical obstruction, run:

```bash
python -m scripts.m47_8_gaussian_oracle \
  --side 33 --fraction 0.018731345684936694 \
  --samples 3 --heun-steps 25 50 100 200 \
  --output /tmp/m47_8_gaussian_oracle.json
```

The exact Gaussian optimal straight-path FM vector field is used, with a flat isotropic spatial spectrum and transverse/longitudinal covariance eigenvalues such that their longitudinal fraction is 1.8731%. **No DiNAT model is trained.** A separate future experiment must train unchanged M37 on these synthetic full-field targets, using its original Gaussian source and FM loss, before drawing an architectural conclusion.

## Interpretation checklist

1. Check whether **25/50/100-step Heun** changes the longitudinal fraction materially for the same sampling key. This is a *small M47-specific control*, not a repeat of the broad M45 studies.
2. Review clean-estimate MSE and density validity at each time. Clean estimates at t<1 are generally conditional means, not DNS draws; their covariance need not equal DNS covariance.
3. Compare each complex J(k,t) with a longitudinal/transverse projector model. A matrix near a scalar identity, across several non-axis wavevectors and interior times, would support the **missing spectral coupling** hypothesis. A large projector-like anisotropy would mean coupling exists at this local linear-response level, so generation dynamics should be inspected next.
4. Check off-mode leakage and finite-difference amplitude sensitivity; high leakage/instability limits interpreting each 3x3 diagonal-in-k matrix.
5. Interpret only as an audit of **one frozen checkpoint on validation data**. It does not identify the cause among graph reach, periodicity, dataset diversity and the learned objective without subsequent controlled ablations.
