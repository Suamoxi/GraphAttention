# M44 SR-0.2 — Checkpoint-based conditional-utility diagnostics

## Evidence and intended inference

First multi-snapshot point-sampled DNS proxy experiment (Slurm 483921;
revision `81fc845`) completed with 28/28 preflight tests and 5 epochs:
train Flow Matching MSE 1.8755 -> 1.2524, validation 1.6427 -> 1.1789,
best at epoch 4, held-out test 1.7936, and a nonphysical spatial
condition-permutation velocity change 0.4541. Training has not converged;
none of these numbers measure SGS fidelity or validated LES super-resolution.

**Do not introduce SR-1, physical filtering, coordinate changes, or new
architectural capacity from these measurements alone.** Preserve the 9^3 fine /
5^3 coarse patch, 10 input field channels, 4-layer geometric DiNAT-DiT,
relative displacement attention and disabled absolute-coordinate addition.

## Implementation

- `src/graph_attention/tasks/sr0_evaluation.py`: pure tensor functions for
  per-conservative-field velocity MSE, exact injection anchor identification,
  point-sample node replacement and multi-draw descriptive statistics.
- `scripts/diagnose_m44_sr0.py`: **read-only** audit of an existing `best.pt`.
  Reuses its manifest, source snapshots, geometry, M3.3 case references, saved
  train-only statistics, task, and controlled model factory. Checks SHA256 of
  manifest and reference file, native mesh size+mtime, source snapshot IDs and
  size+mtime, saved scaling equality, and actual model input/output dimensions.
  It does not train, choose an epoch, alter the source run, or re-fit statistics.
- `tests/unit/test_m44_sr0_evaluation.py`: standalone CPU numerical gates.
- `tests/unit/test_m44_sr0_diagnostics_runner.py`: synthetic HDF5 end-to-end
  audit including best-checkpoint read-only integrity and missing-donor handling.
- `scripts/slurm/m44_sr02_diagnostics.slurm`: cluster launcher with preflight.

## Three tests that distinguish learning and context

### A. Fixed-time/noise per-field reference

For each train, validation, and test patch, use the EXACT deterministic
Flow Matching validation problem and evaluate the trained model:

    matched_MSE_c = mean_{fine_nodes} [v_theta(z_t,t | C_true)_c - v*_c]^2
    zero_MSE_c    = mean_{fine_nodes} [0 - v*_c]^2

Evaluate once with fixed source snapshot/patch IDs. Output per-field and
per-snapshot losses, and gain over the exact zero predictor. The zero
reference is approximately 2 in expectation, not universally 2 for each
finite deterministic validation realization. A falling aggregate FM loss
without stratification is insufficient to assess generalization.

### B. Condition usefulness with matched vs wrong coarse snapshots

At the SAME fine location, take `C_wrong` from the NEXT distinct source DNS
snapshot of the SAME chronological split; retain exactly the original target,
noise `z_t`, FM time, fine graph and reference velocity. Calculate:

    condition_gain = MSE(v_theta(z_t,t | C_wrong), v*)
                   - MSE(v_theta(z_t,t | C_true), v*)

A positive gap means the correct condition improves the selected FM
prediction problem against this counterfactual. It does NOT imply correct
physical conditioning or validated conditional PDFs. A non-positive gap
challenges the claim that conditioning helps. For a split with only one
snapshot, wrong-coarse metrics are NULL, never fabricated.

This differs from the existing within-field node permutation: wrong coarse
values are *real* coarse fields from another snapshot, at the same patch
position. It is still not a physically realizable coarse/fine pair.

### C. Held-out multi-draw diagnostics and point-observation consistency

For up to two distinct validation and two distinct test DNS snapshots,
generate four stochastic draws each using 50 Heun steps by default. Report
per-field generated/resolved fine residual RMS, the single available DNS
residual RMS, sample-to-sample standard deviation, and the fraction of nodes
with nonpositive density. These are descriptive diagnostics, NOT conditional
PDF calibration. A 9^3 patch is too small for a credible inertial-range
energy-spectrum assessment.

Point-sample anchor correction is **diagnostic-only**:

    U_corrected[i] = Uc[j]   at each observed coarse-vertex anchor i<->j
    U_corrected[i] = U_generated[i]    at every other fine node

This enforces only `Q_sample U_corrected=Uc`. It does NOT enforce conservation
of finite-volume integrals, positive pressure, discretized equations or
physical LES consistency. It is never applied to the training target or loss,
and does not silently replace the raw generated field. Both raw and corrected
fields are saved in a small example artifact.

## How to run on Calypso

First diagnose the existing 5-epoch model **without any retraining**:

```bash
cd /scratch/coop/theret/GraphAttention
git switch main
git pull --rebase
sbatch --export=ALL,M44_SR02_SOURCE=/scratch/coop/theret/GraphAttention_runs/m44_sr01/job_483921 \
  scripts/slurm/m44_sr02_diagnostics.slurm
```

Default output: `/scratch/coop/theret/GraphAttention_runs/m44_sr02_diagnostics/job_<NEW_JOB_ID>/`.
Contains `diagnostic_summary.json`, `flow_matching_per_patch.json`, and
`ensemble_examples.pt`. The original run remains unchanged.

In parallel or afterward, continue the **same** SR-0.1 training run from
5 to 50 total epochs (do not restart from random weights):

```bash
sbatch --export=ALL,M44_SR01_RESUME=1,\
M44_SR01_OUTPUT=/scratch/coop/theret/GraphAttention_runs/m44_sr01/job_483921,\
M44_SR01_EPOCHS=50 \
  scripts/slurm/m44_sr0_multisnapshot.slurm
```

The original runner validates the manifest and restores model, AdamW and
Flow Matching RNG state. It writes `best.pt`, `last.pt`, `history.json`,
`summary.json` in the **same** source run directory. Re-run SR-0.2 diagnostics
against the new best checkpoint afterward, using a fresh diagnostic output
folder; never compare checkpoints without recording their best epochs.

**Important:** Do not run diagnostics concurrently with resume on the same
source directory. The resume job updates `best.pt`; wait for it to finish or
run diagnostics on a frozen copy of the earlier checkpoint.

## Decision gates before SR-1

1. Validation FM loss should stabilize; compare the saved best epoch, not
   merely final epoch. Check per-snapshot variability and test/validation gap.
2. MSE against the exactly defined zero predictor should be lower on held-out
   data, and correctly matched condition should generally outperform
   wrong-coarse controls. A positive permutation sensitivity is insufficient.
3. Generated residual RMS and conditional variability should be compared to
   the DNS reference across held-out patches. Check anchor violations before
   and after diagnostic projection and density positivity.
4. Only after these checks consider (a) explicit coarse-spacing conditioning,
   (b) deeper-layer condition skips, and (c) SR-1 learned coarse encoder in
   isolated controlled ablations. Physical LES restriction remains owned by
   `dnstoles` and is not replaced by the point-sample proxy.
