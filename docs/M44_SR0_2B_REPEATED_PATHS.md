# M44 SR-0.2b — Repeated-time/noise checkpoint evaluation

## Why this extension was added

After resuming the unchanged SR-0.1 model from 5 to 50 epochs, the best
snapshot-held-out Flow Matching validation MSE improved from 1.1789200 to
0.9243461 (epoch 48); test MSE improved from 1.7936070 to 1.2908618.
The generated 50-step Heun residual RMS ratios on one validation patch were
uneven (rho 1.10, rhou 0.73, rhov 1.47, rhow 1.03, rhoE 1.07). Those are
single-draw diagnostics, **not** distributional fidelity metrics.

SR-0.2 originally used a single deterministic Flow Matching time/noise path
per patch to compare correct and counterfactual coarse conditions. A positive
gap there supports conditional sensitivity, but its magnitude is sensitive to
the randomly chosen evaluation path.

## Read-only extension

`scripts/diagnose_m44_sr0.py` now accepts `--path-repeats N` (default 1,
fully backward compatible). For N >= 2, it also evaluates deterministic
repeated FM time/noise paths for **train, validation and test**, using:

- the **same best.pt**, M3.3 reference scales, source snapshots, fine topology,
  training-only scaling and spatial condition protocol;
- the **same fixed noise, FM time and fine target** for matched and wrong-coarse
  predictions within each repeat; wrong conditions come from a different DNS
  snapshot at the **same patch origin**;
- a deterministic seed of `base_validation_seed + repeat*1000003`; repeat
  zero reproduces exactly the original saved validation-problem construction,
  and a parity gate rejects any mismatch.

Output `diagnostic_summary.json` has an additional
`repeated_flow_matching` section by split, with mean per-field MSE for
matched and zero velocity, wrong-minus-matched counterfactual gap, and the
number of source snapshots whose *repeat-averaged* gap is positive.
`per_snapshot` retains the per-field breakdown. The standard one-path
`flow_matching_counterfactual` and multi-draw `generation_ensemble` fields
are unchanged.

A **path repetition is not a new physical DNS sample**. In particular, 16
paths on 3 validation snapshots are still only 3 independent validation
snapshots. The reported snapshot-standard-deviation is descriptive and must
not be interpreted as a confidence interval or a test of statistical
significance. The coarse data are still **point-sampled DNS, not LES**.

## Run the existing 50-epoch best checkpoint

```bash
cd /scratch/coop/theret/GraphAttention
git switch main
git pull --rebase

sbatch --export=ALL,M44_SR02_SOURCE=/scratch/coop/theret/GraphAttention_runs/m44_sr01/job_483921,M44_SR02_PATH_REPEATS=16 scripts/slurm/m44_sr02_diagnostics.slurm
```

The launcher runs the SR-0.2 tensor and synthetic HDF5 integration gates
before read-only GPU evaluation. The result is written to
`/scratch/coop/theret/GraphAttention_runs/m44_sr02_diagnostics/job_<JOB_ID>/diagnostic_summary.json`.
The source best and last checkpoints are not modified.

## Decision rule

Evaluate the signs and per-field magnitudes of the **repeat-averaged**
wrong-minus-matched gap on independent validation and test snapshots, and
the improvement of matched/zero MSE relative to the 5-epoch diagnostics.
If the matched condition consistently helps, investigate physically
meaningful coarse restriction and coarse-consistent generation next.
If the result remains highly variable across snapshots, increase independent
source snapshots/patch diversity before adding a coarse learned encoder.

Do **not** interpret a favorable Flow Matching MSE or its conditioning gap
as proof of conditional PDF quality, spatial spectral accuracy, conservation,
or SGS validity.
