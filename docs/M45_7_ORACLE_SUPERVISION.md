# M45.7.1 — matched FM versus analytical Gaussian velocity supervision

**Scope:** synthetic block-average Q/B residuals only. This does not qualify physical LES/DNS.
**Architecture:** existing `TwoScaleDiNAT`, unchanged (static/dynamic).
**Question:** is the missing shared residual covariance primarily a noisy-FM-supervision/optimization issue, or does it persist when the network directly learns the exact conditional-mean FM velocity?

## Experiment

Four Slurm array tasks:

| Array task | Mode | Training target |
|---|---|---|
| 0 | static | sampled FM velocity \(R-\epsilon\) |
| 1 | static | exact Gaussian oracle \(E[R-\epsilon\mid X_t,U_c]\) |
| 2 | dynamic | sampled FM velocity |
| 3 | dynamic | exact Gaussian oracle |

Both objectives use the **raw network velocity**, without projecting network output before loss. This matches M45.6 full-field training; the source noise and residual are both in `ker(Q)`. Oracle supervision is **not** direct residual regression. Only the target field differs.

The ordinary FM and oracle experiments share the same independently constructed coarse/fine dataset, train/validation split, train normalization, sampled FM times and noise, model initialization, architecture, AdamW, gradient clipping, seed, and number of updates. Dynamic and static share initial parameters where present, as in M45.6. The oracle is constructed for each distinct coarse training example (using the existing `BlockGaussianOracle`) and its velocity is evaluated at exactly the sampled FM state and time.

Initial pilot: block-average, r=2, coarse side 4 (fine side 8), `shared_mode`, 24 training fields, 8 validation fields, 2000 updates, depth=4, width=16, 4 attention heads, seed=2028, validation seed=7002026. The test uses full-field supervision to avoid mixing an architecture diagnosis with incomplete global patch context.

Use existing `scripts/m45_checkpoint_diagnostics.py` after each train run. It evaluates errors against the oracle at times [0, .1, .25, .5, .75, .9, 1], independently reports the irreducible oracle FM loss and excess velocity error, and generates an ensemble for the same fixed validation coarse field using the same random initial states for all four variants. The generated mode second moment, channel RMS, spatial correlations, and `max_Q_residual` are the primary generative metrics. An ensemble of 128 is a pilot only, not a precise tail-statistics evaluation.

## Execution: only inside Slurm

**Never execute Python scripts or pytest directly on the login node.**
Syntax compilation, unit tests, training, and diagnostics all run inside the Slurm job using the same container and Python interpreter.

From the HPC login node:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch --array=0-3%2 scripts/slurm/m45_7_oracle_supervision.slurm
```

Optional first-run smoke test (all Python still runs in Slurm):

```bash
sbatch --array=0-3%2 --export=ALL,M45_7_STEPS=16,M45_7_GENERATED_SAMPLES=8,M45_7_VELOCITY_DRAWS=1 \
  scripts/slurm/m45_7_oracle_supervision.slurm
```

Do not interpret the 16-update smoke test as a modeling result. Every rerun receives a new Slurm job ID and output root, so results are not overwritten.

To inspect:

```bash
sacct -j <ARRAY_JOB_ID> --format=JobID,State,ExitCode,Elapsed
find /scratch/coop/theret/GraphAttention_runs/m45_7_oracle -type f | sort
```

Standard output:
`/scratch/coop/theret/GraphAttention/m45_7_oracle_<ARRAY_JOB_ID>_<TASK_ID>.out`

Standard error:
`/scratch/coop/theret/GraphAttention/m45_7_oracle_<ARRAY_JOB_ID>_<TASK_ID>.err`

Each task writes under:

```text
/scratch/coop/theret/GraphAttention_runs/m45_7_oracle/
  job_<ARRAY_JOB_ID>_<TASK_ID>_<MODE>_<SUPERVISION>/
    summary.json
    environment.json
    shared_mode/block_average/<MODE>/
      checkpoint.pt
      metrics.json
      train_history.csv
    diagnostics/
      summary.json
      time_resolved.csv
      time_summary.csv
      generation_metrics.csv
      mode_amplitudes.csv
      spatial_correlations.csv
```

Share/upload `summary.json`, `environment.json`, `metrics.json`, `train_history.csv` and all CSV/JSON files in `diagnostics/` for all four tasks. The `checkpoint.pt` files are not needed for the first numerical review, but retain them for reproducibility.

## Scientific interpretation

Compare held-out oracle excess MSE and mode-velocity MSE at each time, plus the **recovered fraction of the analytically known mode second moment**. Do not equate raw FM loss with oracle-supervised training loss; they have different irreducible minima. Improvements in velocity error must also translate into conditional generation quality.

A poor oracle-supervised result does not by itself prove an architectural limitation. The subsequent experiment must distinguish optimization from expressivity through longer training, small-set overfitting, and/or controlled architectural changes.

**M45.6 depth aggregation remains separate**, and its results do not block this test.
