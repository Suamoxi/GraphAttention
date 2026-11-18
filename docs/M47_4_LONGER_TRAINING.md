# M47.4 — matched longer-training continuation of M47.1

## Aim and scientific constraints

The five-snapshot-per-regime M47.3 audit confirmed that physical conditioning
reduces teacher-forced Flow Matching (FM) MSE, while two-epoch generated fields
remain physically unsatisfactory. M47.4 tests whether **longer training alone**
improves independently generated compressible HIT fields.

- Three arms: none, mach, mach_re; original model, physical normalization,
  chronological split, optimizer, learning rate, seed 42 and equal-case accumulation.
- Original population: 240 train, 30 validation and 42 existing test snapshots;
  48 excluded by temporal embargo.
- Resume each run from the original epoch-index-1 **last.pt**, not best.pt.
- Train to total 5 epochs, freeze checkpoint, then continue the same state to
  total 10 epochs. Never train the epoch-10 run independently from scratch.
- Original training revision: 5d472796828a8abeb51747751b777dc8cc837f35.
- Original M47.1 folders and M47.2/M47.3 results are immutable.
- Epoch-5 and epoch-10 generated-field diagnostics use the original **validation**
  DNS population; no test field values are supplied to the new evaluator.

The Slurm training job executes a local, detached clone pinned to the **exact
M47.1 code revision**. It thereby uses the original checkpoint signature and
full saved optimizer and Flow Matching generator state, without bypassing the
strict M43 training-run resume gate or modifying its training implementation.
The staging script checks original SHA256 hashes before copying and freezing.

### Legacy trainer test-metric caveat

The original M47.1 training runner unconditionally computes its held-out
'test_at_best_validation' metric when a command finishes. Therefore, using
that exact original code at epochs 5 and 10 also records *legacy test metrics*
in its training summaries. **These are not used by M47.4 evaluation or
comparison, and must not be used for model selection.** This is not a
completely sealed test protocol; a never-evaluated final test requires an
additional independent holdout or an audited modification to the trainer.
We retain original training numerical behavior instead of silently changing
the legacy runner.

## Calypso execution

From a clean updated main checkout:

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase origin main
git status --short
unset M47_CONDITION

# 1. Three full-state continuations: 2 -> 5 total epochs
sbatch --array=0-2 --export=ALL,M47_4_STAGE=5 scripts/slurm/m47_4_continue.slurm
~~~

After **all three** stage-5 jobs succeed and each new epoch05/summary.json exists:

~~~bash
# 2. Three epoch-5 validation and generation evaluations
sbatch --array=0-2 --export=ALL,M47_4_STAGE=5 scripts/slurm/m47_4_evaluate.slurm

# 3. Three 5 -> 10-epoch continuations (stage 5 must be frozen)
sbatch --array=0-2 --export=ALL,M47_4_STAGE=10 scripts/slurm/m47_4_continue.slurm

# 4. Three epoch-10 validation and generation evaluations
sbatch --array=0-2 --export=ALL,M47_4_STAGE=10 scripts/slurm/m47_4_evaluate.slurm

# 5. Once all six evaluation summaries are complete
python3 -m scripts.m47_4_compare
~~~

Observe successful completion of each preceding stage before starting its
dependent stage. The stage-10 gate verifies that the live last.pt and history
are identical to the frozen epoch05 snapshot.

The evaluator processes **all 30 original validation snapshots** at
FM time t = 0, 0.25, 0.5, 0.75, 1, reporting per-regime and per-conservative-
channel MSE against the same Gaussian source. Default independent generation
uses four 50-step-Heun Gaussian samples on Re54/Ma0.2 (65^3 source grid).
Identical latent keys and seed 20261116 are used across modes and epochs.

To generate on all six regimes, set M47_4_EVAL_ALL_CASES=YES for *both*
evaluation stages. Adjust M47_4_EVAL_SAMPLES and M47_4_EVAL_STEPS only
consistently across all six evaluations.

### Diagnostic coverage

Independent generated fields are compared distributionally, not paired to
individual DNS targets. Reported quantities include:

- One-point Wasserstein-1 of rho, rhou, rhov, rhow, rhoE, normalized by
  reference standard deviation;
- density positivity and internal-energy positivity (necessary EOS proxy);
- physical-wavenumber velocity-energy spectra and Helmholtz dilatational
  energy fraction, where whole-field density is strictly positive;
- velocity spatial two-point correlations and periodic endpoint seams;
- raw per-sample statistics in the detailed per-case JSON outputs.

Invalid-density velocity spectra remain null. No floor, clipping, projection
or rejection is used to conceal nonphysical samples.

## Output layout

~~~text
/scratch/coop/theret/GraphAttention_runs/
  m47_conditioned/{none,mach,mach_re}/train_s42/  # ORIGINAL; untouched
  m47_4_longer_training/
    none/
      m47_4_source_provenance.json
      train_s42/                            # live continuation
      epoch05/                              # frozen 5-epoch checkpoint
      epoch10/                              # frozen 10-epoch checkpoint
      eval_epoch05_h50_n4_seed20261116/
      eval_epoch10_h50_n4_seed20261116/
    mach/...
    mach_re/...
    comparison_epoch05_epoch10.json
    comparison_epoch05_epoch10.csv
~~~

Each training stage carries the original signature, split, scaler and reference
metadata, optimizer checkpoints, training history and summary. Each evaluation
contains summary.json, validation_details.json and case_<regime>.json.
Full generated volumes are not stored by default. The CPU-only comparison
refuses any missing arm, mismatched validation cohort, mismatched Gaussian key,
different physical sampling settings, or divergent 5-to-10 epoch histories.

## Calypso host vs Singularity scratch mount

M47.4 staging runs under host Python and records the immutable M47.1 source
path after Path.resolve(), for example:

- Host: /gpfs-calypso/scratch/coop/theret/GraphAttention_runs/m47_conditioned/mach/train_s42
- Singularity: /scratch/coop/theret/GraphAttention_runs/m47_conditioned/mach/train_s42

Inside the Singularity evaluator the host-prefixed path may not exist.
The evaluator maps **only** the exact /gpfs-calypso/scratch -> /scratch
mount alias and verifies that the remapped directory contains the same
original source files (SHA256 hashes, training signature, split and revision).
It does not rewrite the original provenance JSON or accept a different
source root. Do not edit the provenance or recreate the epoch checkpoints.

If evaluation previously failed before creating its output directory, pull
the fix and resubmit the same evaluation Slurm array. This alias fix is
independent of occasional Singularity CUDA compatibility-path read-only
warnings, which are not the provenance exception.

## Interpretation and limitations

- A smaller FM velocity MSE is not proof of physical generation fidelity.
  Judge longer training using thermodynamics, distributions, spectra,
  solenoidal/dilatational fraction, correlations and periodic seams.
- All six regimes were trained within-case; no held-out-condition
  interpolation or extrapolation is demonstrated.
- Re54 versus Re94 remains confounded with 65^3 versus 129^3 source grids.
  Consecutive validation snapshots are not guaranteed independent.
- Original checkpoint restores model, AdamW and the explicit training
  generator. Legacy code does not serialize every process RNG state; dropout
  is zero, but cross-device bitwise equivalence is not guaranteed.
- Slurm jobs refuse overwriting existing results. If a job is interrupted,
  do not delete the partial continuation optimizer state. Inspect its
  history before controlled recovery; a fresh stage-5 preparation will
  correctly refuse an already-created workspace.
- No GPU execution has yet been performed for M47.4. The unit tests in
  tests/unit/test_m47_4.py are intended to run on Calypso before evaluation.

If epoch 5 and 10 improve generated physics, expand the validation ensemble
before finalizing a checkpoint. If only teacher-forced MSE improves, inspect
regime scaling, energy representation and missing periodic graph connectivity
before increasing the training budget further.
