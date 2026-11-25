# M47.8B — Matched teacher and self-generated trajectories

## Question

Does the frozen M47.7B DiNAT-DiT learn adequate wavevector-dependent
cross-momentum coupling along **DNS-containing training interpolants**, but
not along its **own trajectories**? Or is its coupling weak on both?

This is a **read-only diagnostic** of the existing Re54/Ma0.2 checkpoint,
not new training, architecture modification, or a correlated Gaussian prior.

## Controlled construction

For each of the five **frozen validation** DNS fields, use one identical
deterministic Gaussian initial field `x0` for two paths:

- Teacher: `x_teacher(t)=(1-t)*x0+t*x_DNS`.
- Generated: `x_generated(t)=HeunODE(v_theta,x0,t)`.

Use the previously established M47.2 key
`M47.2/Re54/Ma0.2/replicate_0000`, seed `20261116`.
A strict equality guard confirms that the two paths share the same `x0`.

**100 Heun steps** exactly visit `t=0.25,0.50,0.75` at steps 25,50,75.
The probe rejects non-grid times instead of rounding, so comparisons
really occur at the same generative time.

At each time compare:

1. Standardized state discrepancy, absolute and relative RMS, per channel.
2. Longitudinal fraction and cross-term cancellation of the **momentum divided
   by fixed DNS density** (same denominator for both paths). This is a
   velocity-equivalent momentum **proxy**, not physical velocity unless the
   instantaneous density equals DNS density.
3. True physical velocity spectrum `u=rhou/rho` only if all nodes have
   positive density, with separate internal-energy validity flag. Early
   Gaussian-mixed states must not be assumed thermodynamically admissible.
4. Longitudinal fraction and cancellation of the **predicted FM momentum-rate
   field**, converted with the same fixed DNS density (not fluid velocity
   time derivative).
5. Full complex 3x3 Fourier response matrices `J(k,t)` using the **corrected
   M47.8A v2** physical Fourier carriers and the same density, amplitude,
   position indexing and scaling on both paths. Report same-mode response
   differences, longitudinal/transverse gains, projector fit, off-diagonal
   response and off-mode leakage. The pure input carrier must pass the
   existing <1e-6 FFT leakage guard on both paths.

Wavevectors are `(2,2,0)`, `(4,3,1)`, `(8,6,5)`. Perturbation
amplitudes are 4% and 8% of DNS velocity-component RMS.

Each Slurm task creates **18 matched Fourier-response pairs** (36 complex
matrices), three timewise state/covariance comparisons and the final
generated-state covariance. All output includes training signature SHA,
checkpoint, Git revision, validation snapshot, source seed and numerical
protocol.

## Cluster submission

All Python (including syntax checks, unit tests and model inference) runs
through **Grace Slurm + pyg25.03 Singularity + avbpML_env**. Do **not**
run Python from the login node.

```bash
cd /scratch/coop/theret/GraphAttention
git pull --ff-only origin main
git status --short

# Qualify the protocol on one frozen validation snapshot first.
sbatch --array=0 scripts/slurm/m47_8b_trajectory_response.slurm

# After reviewing this job, repeat across the other four snapshots.
sbatch --array=1-4%2 scripts/slurm/m47_8b_trajectory_response.slurm
```

The launcher requests one GPU and four CPUs from the `grace` partition,
**no explicit --mem limit** (Slurm defaults apply), using

- `/softs/local_arm/singularity/images/pyg25.03.sif`
- `/scratch/coop/theret/avbpML_env/bin/python3`
- `/scratch/coop/theret/GraphAttention_runs/m47_7_single_regime_training_fast_v1/epoch30.pt`.

Results:
`/scratch/coop/theret/GraphAttention_runs/m47_8b_trajectory_response/job_<ARRAY_JOB_ID>_epoch30_val<i>/audit.json`

Logs:
`/scratch/coop/theret/GraphAttention_runs/m47_8b_paths_<ARRAY_JOB_ID>_<TASK_ID>.out`
and `.err`.

The launcher refuses dirty checkouts, missing checkpoints and pre-existing
results directories. No sealed test DNS fields are read.

## Interpretation

- **Similar near-identity gains on both paths:** supports deficient
  polarization selectivity of the learned FM field, but does not isolate
  architecture vs finite data vs loss.
- **Projector-like gains on teacher path but not generated path:** supports
  a state-distribution/trajectory mismatch hypothesis and motivates
  conditional-vs-generated state analyses.
- **Strong projected gains on both paths, wrong generated covariance:**
  suggests studying the cumulative vector-field transport rather than
  assuming a single-mode Jacobian diagnosis is decisive.
- **Large off-mode leakage:** the 3x3 *same-mode* matrix is incomplete,
  especially at high k. Inspect full spatial output response separately.

These are **local Jacobians at one state per run**, not direct covariance
operators or an ensemble guarantee. Across five validation samples, compare
between-sample variation before making claims about the whole regime.
The experiment cannot by itself establish causality or compare the learned
field with the unknown exact DNS transport. The M45 Gaussian super-resolution
oracle is a different, conditional-residual experiment.
