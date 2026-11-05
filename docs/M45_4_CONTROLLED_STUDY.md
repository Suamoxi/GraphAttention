# M45.4 — Controlled training and exact-Gaussian sampling positive control

## Questions addressed

Keep M45.2's **two-layer** `TwoScaleDiNAT` architecture unchanged and determine:

1. Does integrating the **exact** conditional Gaussian FM velocity reproduce the known synthetic covariance? If not at 8/25/50 Heun steps, how many steps are required?
2. Does the patch-owned training implementation match full-graph prediction under the *same owned-node loss*? Does using a loss on **all** fine nodes change convergence and distribution statistics?
3. How do independent training realizations (`24` vs `256`) and independent random training seeds (`2026`, `2027`, `2028`) affect the learned conditional FM velocity and stochastic mode?

This is synthetic block-average `Q/B` qualification, **not** a real DNS/LES validation. No changes to `dnstoles` or the neural architecture.

## Training scopes: three distinct comparisons

The synthetic training runner now accepts:

- `--training-scope patch_owned` (legacy default): use the exact dependency-complete subgraph and compute the loss on the selected 64 owned fine nodes.
- `--training-scope full_owned`: evaluate the entire fine/coarse graph but compute the loss on **the same 64 owned nodes**, using the same training sample, FM noise and sampled time as `patch_owned`. With complete dependency closure and deterministic operators, predictions and parameter gradients should be numerically equivalent. This is a correctness and scheduling control, **not** a different learning objective.
- `--training-scope full_all`: evaluate the full graph and compute the FM loss on **all 512 fine nodes**. One update observes 8 times as many target nodes as an owned-core update. This is a *different stochastic gradient estimator and compute budget per update*, so improvements cannot be attributed to partitioning alone.

The trainer records `training_scope`, `train_samples`, read/owned node counts, training loss, held-out full-graph FM loss, parameters, training duration, and saved checkpoints. The flags `--fine-pre-layers`, `--fine-post-layers`, `--coarse-static-layers` and `--coarse-dynamic-layers` remain unchanged and are **all fixed to 2** for M45.4.

### Fairness across training seeds

Set `--val-seed 7002026` to use the same held-out eight synthetic coarse/fine realizations for every training seed and dataset size. If `--val-seed` is omitted, the original M45.2 behavior remains intact: validation starts at `--seed + 7000000`.

For a fixed `--seed`, static and dynamic models share the same train/val snapshots, FM paths, and initialization of common parameters. Across training seeds, the train snapshots, model initialization, FM stochastic path and optimizer sequence vary; the validation split is held fixed using `--val-seed`. The per-channel training normalization is recomputed for each training set and saved inside each checkpoint.

The old 24-sample / 2,000-update results remain directly reproducible with the original Slurm script. They are not overwritten.

## Exact Gaussian oracle as a numerical positive control

`BlockGaussianOracle` calculates the optimal conditional FM velocity for projected Gaussian source `X0=P N(0,I)` and the known block-average Gaussian synthetic target. We now integrate that *exact* velocity using the **same Heun formula and noise initialization** as the model, then reverse the residual normalization. There is no trained neural model in this positive-control path.

`--oracle-sampling` enables this control in `scripts/m45_checkpoint_diagnostics.py`. `--oracle-heun-steps 8 25 50 100 200` specifies oracle integration resolutions **independently** from `--heun-steps 25` for the trained models.

Why separate step lists? The shared rank-one covariance introduces modes with a large source/target variance ratio. Initial local tests show the *exact* Gaussian velocity can require more Heun steps than the trained velocity to recover its theoretical variance. Therefore, neural 25-vs-50-step stability does **not** prove 25 steps would suffice for the correct velocity field. Compare the positive-control result against the analytical modal second moment and the independent synthetic sampler at fixed coarse input.

`generation_metrics.csv`, `mode_amplitudes.csv`, and `spatial_correlations.csv` add rows with `mode=oracle, variant=exact_velocity`; existing saved diagnostic rows retain their schema. The Gaussian oracle is supported only for `block_average` Q/B and the projected source.

For all trained checkpoints the diagnostic also reports time-resolved MSE against the Gaussian velocity oracle at `t=0,0.1,0.25,0.5,0.75,0.9,1`, mode-specific velocity errors, conditional 128-sample generation at one held-out coarse state, and a dynamic-feedback ablation.

## Grace execution

From the repository:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
```

One complete **24-sample / patch-owned / seed-2026** training+diagnostic+oracle pilot:

```bash
sbatch scripts/slurm/m45_4_training_controls.slurm
```

The **twelve-job controlled matrix** (2 loss scopes x 2 dataset sizes x 3 training seeds):

```bash
sbatch --array=0-11 scripts/slurm/m45_4_training_controls.slurm
```

Matrix mapping:

| Task indices | Training scope | Train realizations | Seeds |
|---|---|---:|---|
| 0–2 | `patch_owned` | 24 | 2026–2028 |
| 3–5 | `patch_owned` | 256 | 2026–2028 |
| 6–8 | `full_all` | 24 | 2026–2028 |
| 9–11 | `full_all` | 256 | 2026–2028 |

The matrix excludes the costly exact-oracle sampler by default (the standalone pilot includes it). Every matrix task **does** run checkpoint time-resolved evaluation and fixed-coarse generation by default. `--array` overrides manual scope/seed/sample values; do not supply it for a specific custom configuration.

Numerical parity control, with identical data, FM paths, and owned-node loss but full-graph forward:

```bash
sbatch --export=ALL,M45_4_SCOPE=full_owned \
  scripts/slurm/m45_4_training_controls.slurm
```

Faster test settings, without diagnostics:

```bash
sbatch --export=ALL,M45_4_STEPS=32,M45_4_RUN_DIAGNOSTICS=0 \
  scripts/slurm/m45_4_training_controls.slurm
```

Override `M45_4_TRAIN_SAMPLES`, `M45_4_SCOPE`, `M45_4_SEED`, `M45_4_VAL_SEED`, `M45_4_STEPS`, `M45_4_RUN_DIAGNOSTICS`, `M45_4_RUN_ORACLE`, `M45_4_GENERATED_SAMPLES` or `M45_4_VELOCITY_DRAWS` for a single job. Each job runs the unit tests before training.

## Files, logs, outputs

| Role | Path |
|---|---|
| Existing runner, now with scope/validation-seed flags | `scripts/m45_synthetic_pilot.py` |
| Existing checkpoint diagnostics, now with exact Heun oracle generation | `scripts/m45_checkpoint_diagnostics.py` |
| Regression tests for oracle and loss-scope equivalence | `tests/unit/test_m45_4_controls.py` |
| Reused base tests, including depth-4 checks | `tests/unit/test_two_scale_hit.py`, `tests/unit/test_m45_synthetic.py` |
| Grace training + diagnostics | `scripts/slurm/m45_4_training_controls.slurm` |
| Slurm stdout/stderr | `/scratch/coop/theret/GraphAttention/m45_4_controls_<JOBID>.out/.err` |
| Results root | `/scratch/coop/theret/GraphAttention_runs/m45_4_controls/job_<JOBID>/` |
| Original training report | `<root>/summary.json`, `<root>/metrics.csv` |
| Per-case model training | `<root>/<case>/block_average/<mode>/train_history.csv`, `checkpoint.pt`, `metrics.json` |
| Evaluation | `<root>/diagnostics/{summary.json,time_summary.csv,time_resolved.csv,generation_metrics.csv,mode_amplitudes.csv,spatial_correlations.csv}` |

**Results for comparison:** Slurm logs; each job's training `summary.json`, `metrics.csv`, and per-model `train_history.csv`; each job's `diagnostics/time_summary.csv`, `generation_metrics.csv`, `spatial_correlations.csv`, and `mode_amplitudes.csv`.

## Scientific interpretation and limitations

- A full-owned loss should **match** patch-owned numerical gradients, rather than produce a scientifically distinct trained model.
- Full-all supervision uses more targets per update, so keep both optimizer-step and total supervised-node budgets in mind.
- Changing training sample count also changes the fitted per-channel normalization and the variety of coarse conditions observed.
- Three seeds give limited uncertainty on training outcomes; treat differences as exploratory.
- The fixed-coarse generation diagnostic uses only one held-out coarse condition per model. Confirm on more coarse conditions before generalizing.
- The analytical reference is only defined for the synthetic Gaussian distribution, not actual turbulent DNS fields.
