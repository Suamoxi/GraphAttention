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

## Submit through a Grace sbatch job (mandatory)

**Do not execute Python, pytest, or the Gaussian oracle on the login node.**
All Python workloads, including syntax validation and unit tests, run within
`scripts/slurm/m47_8_response_audit.slurm` on the `grace` Slurm partition.

This launcher reuses the verified M47.7 environment:

- `#SBATCH --partition=grace`, `--gres=gpu:1`, 4 CPUs, 64 GiB host RAM.
- Singularity image: `/softs/local_arm/singularity/images/pyg25.03.sif`.
- Python executable inside the container: `/scratch/coop/theret/avbpML_env/bin/python3`.
- Bind mounts: `/home`, `/scratch`, and available archives.
- Frozen source: `m47_7_single_regime_training_fast_v1/epoch30.pt`
  (the **native single-graph** Triton-backed checkpoint; not the newer
  segmented training trajectory).

From the cluster login node, use only repository and Slurm commands:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --ff-only origin main
git status --short

JOB=$(sbatch --parsable scripts/slurm/m47_8_response_audit.slurm)
echo "Submitted M47.8 Grace job $JOB"
squeue -j "$JOB"
```

The job itself runs, in this order:

1. `python -m py_compile` for the M47.8 scripts and unit tests.
2. `python -m pytest` for the response, Gaussian-oracle, and M47.7 covariance tests.
3. `python -m scripts.m47_8_gaussian_oracle`, with 33^3 grid,
   3 samples and Heun 25/50/100/200 steps.
4. `python -m scripts.m47_8_response_audit` on the frozen epoch-30 checkpoint,
   using three times, three wavevectors and two perturbation amplitudes,
   plus a fixed-source Heun 25/50/100-step control.

Each Python command runs through `singularity exec --nv` **inside the Grace
sbatch allocation**, never in an interactive shell.

For another validation DNS snapshot (indices 0 to 4), submit a separate
job and let it create an isolated Slurm-job-specific output directory:

```bash
JOB=$(sbatch --parsable --export=ALL,M47_8_VAL_INDEX=1 \
  scripts/slurm/m47_8_response_audit.slurm)
echo "Submitted validation index 1 as job $JOB"
```

Optional `M47_8_EPOCH=10` selects the earlier saved checkpoint; epoch 30
is the default. `M47_8_TRAIN_DIR`, `M47_REPO`, `M47_IMAGE`, `M47_PYTHON`,
`M47_ARCHIVE_ROOT`, `M47_CONDITION_REFS`, and `M47_8_OUT_ROOT`
override paths through `sbatch --export=ALL,...` when necessary.

Each job creates:

```text
/scratch/coop/theret/GraphAttention_runs/m47_8a_response_audit/
    job_<SLURM_JOB_ID>_epoch30_val0/
        full_field_gaussian_oracle.json
        frozen_checkpoint_response/
            audit.json
```

The Grace logs are stored at
`/scratch/coop/theret/GraphAttention_runs/m47_8_response_<JOB_ID>.out`
and `.err`, following the same logging policy as M47.7.

The launcher refuses missing checkpoint/references, a dirty repository,
or a pre-existing output directory. M47.7 model weights, split and
standardizers are only read. The audit uses the established `torch_sparse`
model execution, consistent with the frozen M47.7 evaluation, even though
the original training used `torch_sparse_triton`.

The response output contains **18 complex response matrices**, three
clean-state diagnostics and the Heun control. The two amplitude values
are a finite-difference linearity test; disagreement warrants smaller
amplitudes or higher-precision diagnostics.

## Independent full-field Gaussian oracle (run inside the same sbatch job)

The CPU-only analytical Gaussian oracle is intentionally grouped with
the GPU audit so its tests and computation also run on Grace using the
same Python environment. It demonstrates that a unit-Gaussian source
can reach a longitudinal fraction near 1.873% under exact FM transport.
**It does not train DiNAT-DiT.** The separate neural full-field Gaussian
learning experiment is deferred until the read-only audit is evaluated.

## Interpretation checklist

1. Check whether **25/50/100-step Heun** changes the longitudinal fraction materially for the same sampling key. This is a *small M47-specific control*, not a repeat of the broad M45 studies.
2. Review clean-estimate MSE and density validity at each time. Clean estimates at t<1 are generally conditional means, not DNS draws; their covariance need not equal DNS covariance.
3. Compare each complex J(k,t) with a longitudinal/transverse projector model. A matrix near a scalar identity, across several non-axis wavevectors and interior times, would support the **missing spectral coupling** hypothesis. A large projector-like anisotropy would mean coupling exists at this local linear-response level, so generation dynamics should be inspected next.
4. Check off-mode leakage and finite-difference amplitude sensitivity; high leakage/instability limits interpreting each 3x3 diagonal-in-k matrix.
5. Interpret only as an audit of **one frozen checkpoint on validation data**. It does not identify the cause among graph reach, periodicity, dataset diversity and the learned objective without subsequent controlled ablations.
