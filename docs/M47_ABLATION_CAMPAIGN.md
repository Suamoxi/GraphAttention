# M47.1: matched conditioning ablation (six-case HIT)

## Scientific scope

Compare three M37 DiNAT-DiT Flow Matching models with identical model depth,
physical M3.3 scaling, stationary-candidate snapshots, split, optimizer settings,
training seed and validation protocol. Only the **global physical conditioning**
changes. All models retain the Flow Matching time input to adaLN-Zero.

| Slurm array index | `M47_CONDITION` | Additional physical inputs |
|---|---|---|
| 0 | `none` | None (Flow Matching time remains conditioned) |
| 1 | `mach` | `Ma_ref/0.1` |
| 2 | `mach_re` | `Ma_ref/0.1`, `ln(Re_ref)` |

The physical references are derived from original immutable M3.3 scales and
the common archived-as-repository generic AVBP AIR mixture database:
`configs/m47_physical_conditions_re54_re94_air_sutherland_300K.yaml`.
The Reynolds definition uses the box length, not the label `Re54`/`Re94`.

## Execution on Calypso

```
cd /scratch/coop/theret/GraphAttention
git pull --rebase origin main

# Step 1: Prepare audited reference YAMLs (only once; refuses overwrite).
sbatch --export=ALL,M47_REF_MODE=prepare,M47_PHYSICAL_INPUT="$PWD/configs/m47_physical_conditions_re54_re94_air_sutherland_300K.yaml" \
  scripts/slurm/m47_reference_conditions.slurm

# Step 2: Once preparation succeeded, smoke all three modes.
sbatch --array=0-2 scripts/slurm/m47_conditioned_train.slurm

# Step 3: Review all three smoke summary.json files before enabling training.
# Explicit long-training opt-in, same two epochs / learning rate / seed:
sbatch --array=0-2 --export=ALL,M47_MODE=train,M47_CONFIRM_LONG_TRAIN=YES,M47_EPOCHS=2 \
  scripts/slurm/m47_conditioned_train.slurm

# Step 4: After all three long-training jobs finish successfully:
sbatch --dependency=afterok:<TRAIN_ARRAY_JOB_ID> scripts/slurm/m47_compare.slurm
```

For *single* M47 training, omit `--array` and specify
`--export=ALL,M47_CONDITION=mach_re,...` (or leave the condition unset for the
historical `mach_re` default). Do **not** set `M47_CONDITION` for an array job:
array index determines it, and an ambiguous override is rejected.

The three persistent training directories are:

```
/scratch/coop/theret/GraphAttention_runs/m47_conditioned/
  none/train_s42/
  mach/train_s42/
  mach_re/train_s42/
```

The comparison is written to `m47_conditioned/comparison_s42.json`. It refuses
to write unless all three jobs have completed, their checkpoints exist, and
reference signatures, exact train/validation/test/embargo IDs, code revision,
physical preprocessing, seed, learning rate and completed epoch counts match.
If any arm fails, do not infer that the other two constitute a three-arm test.

### Stationarity and weighting

The six cases have 60 selected snapshots each (counters 41-100). The reviewed
chronological split with a two-snapshot embargo on either side of both
boundaries gives 240 train, 30 validation, 42 test, and 48 excluded snapshots.
Gradient accumulation is six: one sample from each case per optimizer update.
The frozen reference YAML, dataset files and independent training seeds are
recorded in run artifacts. These data are only *exploratory stationary
candidates*, not proven statistically independent samples.

## What this first experiment can and cannot establish

`comparison_s42.json` reports validation/test **Flow Matching velocity MSE**
by mode and case. Choose checkpoints using validation metrics; test results
are reported only after selection. Reduced FM velocity MSE is **not evidence**
of correct generated density PDFs, kinetic-energy spectra, solenoidal/dilatational
energy fractions, spatial correlations or physically admissible states.

The next experiment must **generate ensembles** with the same sampling seeds,
Heun steps and independent selected conditions, and compare (at minimum):
normalized PDFs of `rho`, `rhou/rhov/rhow`, `rhoE`; physical-wavenumber
velocity spectra with consistent shell normalization; dilatational energy
fraction; density/pressure positivity; casewise conditional moments.
Sampling only one generated field per condition is inadequate for stochastic
second moments. Holdout regimes will require a separate train/validation
protocol, not the current within-case temporal test split.

**Confound:** All Re54 cases use unique `64^3` grids and all Re94 cases use
unique `128^3` grids. A gain from Reynolds conditioning does not prove that
the model learned viscosity independent of resolution. The one-seed campaign
is a pilot and must be replicated before claiming robust improvements.