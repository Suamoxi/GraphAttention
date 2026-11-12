# M45.8.1 — Locally conditioned HIT-like synthetic super-resolution

**Scope:** a statistical *velocity-only* benchmark for conditional Flow Matching
and the TwoScaleDiNAT architecture. The fields are not DNS, do not solve
Navier--Stokes, and do not yet enforce fine-grid incompressibility.

This replaces the artificial M45.7 domain-wide shared random mode as the
main benchmark target. **M45.7 remains unchanged and usable as an isolated
global-communication stress test.** M42/dnstoles processing is out of scope.

## Scientific objective

For a fine-grid residual \(R_f=U_f-B(U_c)\), test whether the generated
distribution reproduces local conditional second moments, spectra and
correlations when the unresolved environment is determined by the nearby
resolved coarse velocity.

The local hypothesis is tested with two genuine graph-hop radii:

- **Radius 1:** strain/rotation from the parent coarse cell and its 6
  nearest neighbors (centered finite differences).
- **Radius 2:** additionally smooth strain/rotation over one coarse
  neighbor hop. The maximum dependency radius is two graph hops.
- **Radius 0 (generator control only):** parent coarse velocity magnitude,
  without coarse gradients or strain orientation.

The conditional law is **not** a collection of independent pixel noises.
For each fixed coarse state, a *spatially correlated three-component
Gaussian field* is transformed by the locally conditioned amplitude and
strain anisotropy, then projected into the exact nullspace of the
block-average operator.

## Generator: stage A (implemented)

1. Generate a periodic, spectrally solenoidal coarse velocity field with
   a prescribed smooth, HIT-inspired spectral envelope. Coarse realizations
   use independent random Fourier phases and amplitudes.
2. Compute local coarse velocity gradients, strain, rotation and a bounded
   subgrid fluctuation scale \(\sigma(U_c,\mathcal N_r)\), using exactly
   the requested graph-hop conditioning radius.
3. Generate independent fine-grid Gaussian noise, project it into
   solenoidal Fourier modes and apply a bandpass/small-scale spectral filter.
4. Multiply the correlated fine noise by trilinearly interpolated local
   \(\sigma\) and a bounded strain-aligned anisotropy transformation.
5. Project the candidate to the **block-average Q nullspace**:
   \(R_f=(I-B_0 Q)\,\tilde R_f\), where \(B_0\) repeats coarse block values.
6. Construct a locally interpolated, block-corrected baseline
   \(B(U_c)\) satisfying \(Q B=I\), and form \(U_f=B(U_c)+R_f\).

The fine residual is **Gaussian conditioned on U_c**: every operation
applied to independent standard Gaussian fine noise is linear for a
fixed coarse state. However its covariance *depends nonlinearly* on
the nearby coarse velocity gradients through \(\sigma\) and
anisotropy. This is the intended controlled conditional benchmark.

There is no closed-form Flow Matching velocity implemented for this
general high-dimensional covariance. **Standard FM** is trained directly
on source/target pairs; the exact-Gaussian oracle from M45.7 does not apply
to this new generator.

The coarse velocity is solenoidal in the spectral sense (up to numerical
precision); applying local amplitude modulation and the block-average
nullspace projection does **not** preserve fine-grid solenoidality.
We therefore report fine-grid divergence and do not claim to generate
incompressible fine fields. The smooth baseline also uses local
block-mean corrections; this is not a physical LES reconstruction
filter. Spectra are physically dimensionalized by the specified
periodic box length (default 1), but the spectrum is not claimed to
be an empirical DNS spectrum.

## Network

Both modes reuse the existing `TwoScaleDiNAT` and the same Flow
Matching solver:

- **Static:** encode coarse state once, then up-condition the fine
  network during the generative trajectory.
- **Dynamic:** additionally propagate evolving fine features down to
  the coarse network and back up.
- Fine input/output is **three velocity channels**. The existing
  architecture now supports an optional `channels=3`; the historical
  `channels=5` default and its checkpoints are unchanged.
- All training is full-domain (no new partition implementation).
- A shared model-init seed, common dataset, normalization subset,
  residual stream, schedule and validation split are used for
  matched static/dynamic comparisons.

**Important:** to study stochastic optimization reliably, later
experiments must vary `M45_8_INIT_SEED` while holding the data and
schedule seeds constant. The four-run matrix below is **one matched
initialization**, not a general claim about seed robustness.

## Initial study matrix and sizes

| Parameter | Default |
| --- | --- |
| Coarse / fine | \(8^3\) / \(16^3\) (refinement 2) |
| Train coarse fields | 2048 distinct, deterministic |
| Normalization coarse subset | 128 training fields, frozen |
| Training updates | 20,000 |
| FM residual policy | Fresh conditional residual per update |
| Model | Width 32, 4 heads, depth 4 each fine/coarse block |
| Validation coarse fields | 16 separate resolved environments |
| Inference draws / coarse field | 16 |
| Sampling | Projected 50-step Heun |
| Architectures | Static and dynamic |
| Conditioning radius | 1 or 2 coarse graph hops |

No synthetic fine-truth snapshots are stored persistently; they are
generated reproducibly from explicit seeds. The 2,048 coarse states are
cached in CPU memory for training, while residual samples are regenerated
on demand. Normalization is fitted **only** from 128 training pairs and
then held fixed; validation does not affect scaling.

Default evaluation is an initial benchmark, **not** a converged sampling
study. Check 50/100/200-step convergence before claiming final generated
statistics, especially if the velocity field varies sharply near FM
endpoints.

## Statistical diagnostics

For both generated and independent target ensembles, each validation
coarse field has:

- Conditional subgrid kinetic energy \(0.5\mathbb E|R_f|^2\), stratified
  into five bins of the **observed local coarse strain**. Bin edges come
  from the held-out coarse fields only; no generated fine state enters
  the bin definition. Compare raw curves and amplitude-normalized shape.
- Spatial correlations on the x/y/z axes at lags 0–4, centering each
  fixed-coarse ensemble before measuring covariance.
- Shell-binned 3D kinetic energy spectral density \(E(k)\) of the full
  fine velocity, with **physical angular wavenumber**
  \(k=2\pi n/L\). The normalization is
  \(\sum_k E(k)\Delta k=0.5\langle|u_f|^2\rangle\).
- Unresolved and total kinetic energy, per-channel residual RMS,
  exact coarse-consistency defect \(\max|QR_f|\), and
  fine-grid divergence RMS.

The comparison scorecard reports the residual and total energy ratios,
conditional energy error, normalized conditional-curve shape RMSE,
high-wavenumber log-spectrum RMSE, and spatial-correlation RMSE.
**Loss alone does not determine quality** of the generative statistics.

## Running on Calypso (SLURM only)

Pull latest changes before launching, then check shell syntax. No
Python/pytest runs on login nodes.

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
bash -n scripts/slurm/m45_8_local.slurm
bash -n scripts/slurm/m45_8_compare.slurm
```

### Smoke run

Four small tasks (static/dynamic × 1/2 coarse hops), 3^3 coarse
and 6^3 fine nodes, 2 training updates, 2 conditional test draws,
2 Heun steps. The Slurm script itself executes Python compilation,
unit tests and an end-to-end training/evaluation.

```bash
smoke_id=$(sbatch --parsable --array=0-3%2 \
  --export=ALL,M45_8_COARSE_SIDE=3,M45_8_TRAIN_FIELDS=8,M45_8_VAL_FIELDS=2,M45_8_NORMAL_FIELDS=4,M45_8_STEPS=2,M45_8_DEPTH=1,M45_8_WIDTH=8,M45_8_HEADS=2,M45_8_DRAWS=2,M45_8_HEUN_STEPS=2 \
  scripts/slurm/m45_8_local.slurm)

sbatch --dependency=afterok:"$smoke_id" \
  --export=ALL,M45_8_PARENT_JOB="$smoke_id" \
  scripts/slurm/m45_8_compare.slurm
```

**Note:** the comparison step does not assume a particular training
budget and works with both smoke and full runs. Smoke output tests
software correctness only, not physical/statistical convergence.

### Main run

Once all four smoke jobs and their CPU aggregation succeed:

```bash
array_id=$(sbatch --parsable --array=0-3%2 \
  scripts/slurm/m45_8_local.slurm)

sbatch --dependency=afterok:"$array_id" \
  --export=ALL,M45_8_PARENT_JOB="$array_id" \
  scripts/slurm/m45_8_compare.slurm

sacct -j "$array_id" --format=JobID,State,ExitCode,Elapsed
```

Defaults are overridable with `--export` environment variables
(as shown in `scripts/slurm/m45_8_local.slurm`). CPU/GPU
requirements and walltime must be verified by the first smoke task.
A full run is substantially more expensive than M45.7.

Output contract:

```text
/scratch/coop/theret/GraphAttention_runs/m45_8_local/
  job_<ARRAY_ID>/
    task_0_static_r1/
    task_1_dynamic_r1/
    task_2_static_r2/
    task_3_dynamic_r2/
    aggregate/
      summary.json
      scorecard.csv
      conditional_curves.csv
```

Each task contains `summary.json`, a `local_hit/block_average/<mode>/`
checkpoint and training history, and `diagnostics/` with
`summary.json`, `per_field.csv`, `conditional_energy.csv`,
`conditional_summary.csv`, `spectra.csv`, and
`spatial_correlations.csv`.

Logs go to:

```text
/scratch/coop/theret/GraphAttention/m45_8_local_<ARRAY_ID>_<TASK>.out
/scratch/coop/theret/GraphAttention/m45_8_local_<ARRAY_ID>_<TASK>.err
/scratch/coop/theret/GraphAttention/m45_8_compare_<AGG_JOB_ID>.out
/scratch/coop/theret/GraphAttention/m45_8_compare_<AGG_JOB_ID>.err
```

## Stage B (not implemented)

After this conditional Gaussian benchmark passes, construct the more
physical alternative:

1. Generate non-Gaussian, spectrally structured fine fields with
   controlled intermittency and cross-scale interactions;
2. Produce coarse fields by filtering and restriction
   \(U_c=QU_f\), instead of drawing them first;
3. Recover \(R_f=U_f-BU_c\) exactly and audit the conditional law;
4. Add joint solenoidal/coarse consistency, physically meaningful
   energy spectra, increment PDFs, structure functions, and
   anisotropy diagnostics calibrated against DNS;
5. Then move to actual dnstoles DNS/LES filtering experiments.

Stage B requires a separate benchmark specification. Do **not** infer
physical Navier--Stokes realism from the Gaussian stage-A data.
