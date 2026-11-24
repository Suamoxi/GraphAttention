# M47.7 — Fourier tensor covariance and controlled single-regime training

## Goal

M47.4–M47.6 established that the conditional DiNAT-DiT models recover
regime-dependent mean nondimensional total energy and develop spatial
coherence during Flow Matching, but do NOT reproduce the turbulent
velocity component coupling: their bulk RMS(div u)/RMS(curl u) stays
near 0.7 instead of DNS values ranging from 0.02 to 0.25.

M47.7 separates (A) spectral evidence for missing cross-component
coupling from (B) the possibility that one-regime training for longer
improves this coupling. It does NOT yet change graph topology, use a
correlated Gaussian prior, or isolate long-range context from prior
choice. Those require subsequent controlled ablations if needed.

## M47.7B segmented Triton execution (new training runs only)

The 65^3 real-HDF5 benchmark (Slurm 511501) measured 2.717 s/step
for packed-index original CSR versus 0.428 s/step for native single-graph
broadcast plus Triton CSR backward. Later two-graph tests (Slurm 511557
and 511565) passed complete first-step gradient/AdamW parity within the
published tolerances. These are computational probes, not guaranteed
long-run bitwise parity or an empirical M47.7B speed prediction.

M47.7B now uses `torch_sparse_triton` backward and **packed segmented
graph-conditioning broadcast**. For any packed batch forwarded by
`scripts/train_m43_b_multires.py::_forward` using Triton, graph sizes
are derived from the existing contiguous packed-batch offsets and passed
as `graph_sizes` into the unchanged DiNAT-DiT implementation. Its
`_broadcast_graph_features` selects per-graph expand+cat, avoiding
contended indexed backward. For exactly one graph, the implementation
uses `expand` without a `cat`, but retains the packed coordinate
normalization path. This is distinct from and not guaranteed as fast
as the separately measured native `batch_index=None` fast path.

The shared M43/M47 `torch_sparse` training default is unchanged;
`torch_sparse_triton` scientific forwards use segmented broadcasting
by default. The explicit `single_graph_forward=True` path remains
available when chosen; standalone kernel/packing benchmarks call the
model directly and are unaffected. M47.7B retains the same original
`train_epoch` and `evaluate` algorithms, task, model architecture,
graph topology, optimizer, source noise, train-only normalization
and per-sample objective. M47.7A remains completely unchanged.

Training signatures record both the optimized backend and the
single-graph mode. A new default output root prevents accidentally
resuming or overwriting legacy runs:
`m47_7_single_regime_training_segmented_v1/` for training and
`m47_7_single_regime_evaluation_segmented_v1/` for evaluation.
Old checkpoints must not be resumed into the new optimized execution
trajectory; changing the training code or git commit between epoch 10
and 30 is rejected by the existing resume-signature check.

The train Slurm preflight now runs the M47.7 physics/split tests,
CPU/CUDA native single-graph and segmented-packed gradient/AdamW parity
tests, and the deterministic checkpointed Triton full-DiNAT backward
test in the production image before training.
This is **not** a claim that the optimized 30-epoch run has executed.
Inspect Slurm logs and JSON artifacts before accepting the comparison.

To submit stages and evaluations without interactive dependency handling:

~~~bash
JOB10=$(sbatch --parsable --export=ALL,M47_7_STAGE=10 \
  scripts/slurm/m47_7_single_regime.slurm | cut -d ';' -f1)
JOB30=$(sbatch --parsable --dependency=afterok:$JOB10 \
  --export=ALL,M47_7_STAGE=30 \
  scripts/slurm/m47_7_single_regime.slurm | cut -d ';' -f1)
EVAL10=$(sbatch --parsable --dependency=afterok:$JOB10 \
  --export=ALL,M47_7_EPOCH=10 \
  scripts/slurm/m47_7_single_eval.slurm | cut -d ';' -f1)
EVAL30=$(sbatch --parsable --dependency=afterok:$JOB30 \
  --export=ALL,M47_7_EPOCH=30 \
  scripts/slurm/m47_7_single_eval.slurm | cut -d ';' -f1)
printf 'train10=%s train30=%s eval10=%s eval30=%s\n' \
  "$JOB10" "$JOB30" "$EVAL10" "$EVAL30"
~~~

If stage 10 times out or fails, afterok prevents stage 30; diagnose
the error and resume stage 10 explicitly from `last.pt`. Do not
change the repository revision between resumes.

## A: Read-only Fourier covariance audit

Three frozen M47.4 epoch-10 checkpoints (none, mach, mach_re), each
evaluated at Re54/Ma0.05, Re54/Ma0.2 and Re54/Ma0.3.

Use exactly the original M47.1 validation population (5 DNS fields per
case), 4 independent generations, 50-step Heun, seed 20261116, and the
M47.2 Gaussian keys M47.2/<case>/replicate_<index>.

The velocity in m/s is calculated using the audited U_ref and the
conservative nondimensional fields:
  u_i = U_ref * rho_u_i / rho.

For the original periodic-FFT unique grid (64^3, duplicate maximum
planes dropped) and the shell around physical wavenumber k in rad/m,
compute auto-spectra, ordinary cross-cospectra, and the *weighted*
longitudinal diagonal and cross spectra:

  D(k) = sum_shell sum_i [0.5 |uhat_i|^2 k_i^2/(N |k|^2)] / dk
  C(k) = sum_shell sum_i<j [k_i k_j Re(uhat_i conj(uhat_j))/
                              (N |k|^2)] / dk
  E_longitudinal(k) = D(k) + C(k)
  E_transverse(k) = E_total(k) - E_longitudinal(k).

A perfectly solenoidal Fourier velocity has E_longitudinal=0, and
the cross cancellation ratio -C/D = 1 wherever D>0. For three
independent isotropic velocity components, the weighted cross term
tends to zero, so the longitudinal energy fraction tends to 1/3.

Raw off-diagonal shell cospectra by themselves are NOT a reliable
measure of solenoidality: they can cancel in isotropic turbulence.

Additionally record longitudinal fractions in non-overlapping angular
sectors (axis-like, face-diagonal-like, body-diagonal-like, other),
full spectral tensor curves, and per-field values. Fourier results
remain periodic-seam-sensitive; interpret them alongside M47.6
interior divergence/vorticity statistics.

## B: Controlled single-regime training and evaluation

Case: Re54/Ma0.2, original 65^3 stored mesh and 64^3 unique FFT grid.

- Use original M47.1 stationarity-qualified train/validation sample IDs.
  The test split stays sealed; no new random or temporal split.
- Same M37 DiNAT-DiT architecture, local/exact-two-hop attention,
  no periodic graph modification, and independent Gaussian FM source.
  M47.7B uses opt-in Triton CSR backward and segmented packed
  conditioning broadcast; the mathematical model/objective is unchanged.
- Same model seed (42), learning rate (1e-4), AdamW weight decay 0,
  Flow Matching objective, 1000 time embedding scale, and Heun sampling.
- Fit a NEW train-only sample-balanced scaler on ONLY this physical
  regime, shared by both M47.7B checkpoints.
- One sample per optimizer update, with single-case shuffled order.
  This differs from the six-case mixed-regime batch-accumulation budget;
  therefore per-epoch MSE between mixed and single scalers is not
  directly comparable.
- Save epoch10.pt, then CONTINUE the SAME training RNG, optimizer and
  model weights to epoch30.pt. Do not restart epoch30 from scratch.
  Resume signatures lock source split, reference YAML, model config,
  training hyperparameters and code revision.
- Generate four independent fields at both checkpoints, matching the
  original M47.2 keys and seed, and compute covariance plus the
  M47.6 physical velocity diagnostics. Do not select checkpoints using
  generated physical metrics.
- Never clip negative density. If it occurs, record invalid
  realizations and exclude them explicitly from velocity diagnostics.

**Important:** a positive single-case epoch30 result would indicate that
training budget and/or multi-regime interference matter, but comparing
epochs 10 and 30 in one run does not isolate architecture or prior.
If both remain at about 1/3 dilatational energy and near-zero
cross-cancellation, the next experiment is a matched long-range
context and/or correlated-source ablation, not an unsupported claim
that DiNAT is intrinsically incapable.

## Calypso: execution order

Use a clean checkout of main, with scientific source code unchanged
between single-case training stages, because the resume signature
includes the Git revision.

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase origin main
git status --short
unset M47_CONDITION

# Stage A: first one model, then remaining two
sbatch --array=0 scripts/slurm/m47_7_covariance.slurm
sbatch --array=1-2 scripts/slurm/m47_7_covariance.slurm

# Stage B1: train fresh single regime through epoch 10
sbatch --export=ALL,M47_7_STAGE=10 scripts/slurm/m47_7_single_regime.slurm

# After B1 succeeds: evaluate frozen epoch 10
sbatch --export=ALL,M47_7_EPOCH=10 scripts/slurm/m47_7_single_eval.slurm

# Stage B2: continue exact training from 10 to 30
sbatch --export=ALL,M47_7_STAGE=30 scripts/slurm/m47_7_single_regime.slurm

# After B2 succeeds: evaluate epoch 30
sbatch --export=ALL,M47_7_EPOCH=30 scripts/slurm/m47_7_single_eval.slurm
~~~

Do not submit the two B training jobs simultaneously. B1 and B2
are resumable after job termination (resubmit the SAME stage);
if run_dir exists but has no last.pt because interruption occurred
during initial scaler fitting, use a new M47_7B_TRAIN_DIR.

Default location mapping is under /scratch/coop/theret inside Singularity.
Do not feed host /gpfs-calypso/scratch paths into the container.
The existing M47 Singularity image and Python environment are reused.
Optional environment roots: M47_REPO, M47_IMAGE, M47_PYTHON,
M47_ARCHIVE_ROOT, M47_CONDITION_REFS, M47_4_OUT_ROOT,
M47_7A_OUT_ROOT, M47_7B_TRAIN_DIR, M47_7B_EVAL_ROOT.

Each job first runs
  tests/unit/test_m47_7_covariance.py
on the production container Python before loading GPU checkpoint.
Tests include analytic solenoidal/dilatational Fourier wave modes,
independent Gaussian vectors, angular bins, closure and source split.

## Artifacts to make available for analysis

### Stage A (three models)

Per mode none, mach, mach_re:

~~~text
/scratch/coop/theret/GraphAttention_runs/m47_7_covariance_audit/
  <mode>/epoch10/
    summary.json
    case_Re54_Ma0.05_covariance.json
    case_Re54_Ma0.2_covariance.json
    case_Re54_Ma0.3_covariance.json
~~~

Include Slurm .out/.err for m47_7_tensor_<jobid>_<task>.

### Stage B (one training run and two evaluations)

~~~text
/scratch/coop/theret/GraphAttention_runs/
  m47_7_single_regime_training_segmented_v1/
    run_signature.json          # required
    split.json                  # required
    normalization.json          # required
    history.json                # required
    summary_epoch10.json        # required
    summary_epoch30.json        # required after stage B2
    epoch10.pt                  # do NOT upload unless requested
    epoch30.pt                  # do NOT upload unless requested
    standardizers.pt            # do NOT upload unless requested
  m47_7_single_regime_evaluation_segmented_v1/
    epoch10/
      summary.json              # required
      case_Re54_Ma0.2_covariance.json
    epoch30/
      summary.json              # required
      case_Re54_Ma0.2_covariance.json
~~~

Include both Slurm .out/.err logs from:
  m47_7_single_<jobid>.out/.err
  m47_7_single_eval_<jobid>.out/.err

The JSONs include per-sample spectra, density-invalid counts, physical
velocity metrics, and training provenance. No raw DNS snapshots,
checkpoint binaries or full generated fields are needed for initial review.
In case of a failed stage, share .out/.err and whatever JSONs were written;
do NOT delete completed checkpoint stages.

## Evaluation decisions

1. Check if DNS longitudinal cross cancellation approaches 1 in the
   low-Mach flow, while generated models remain near 0.
2. Check angular sector dependence and the spectral scales where
   cross-component cancellation is absent. Distinguish low-k and high-k.
3. Compare single-case epoch10 vs epoch30 cross cancellation,
   longitudinal-energy fraction, interior RMS(div)/RMS(curl),
   physical spectrum, boundary jumps, and validity.
4. Compare the single-case results with the original mixed-regime
   baseline **in physical units**; the normalizers and training
   populations differ, so raw standardized MSE is not comparable.
5. If little changes by epoch30, design subsequent matched ablations:
   longer graph receptive field; proper periodic connectivity;
   solenoidal/dilatational covariance-conditioned Gaussian prior.
   Do not add all three simultaneously.

No M47.7 result is evidence of extrapolation to an unseen physical
regime, since the reference cases were included in the prior training.
