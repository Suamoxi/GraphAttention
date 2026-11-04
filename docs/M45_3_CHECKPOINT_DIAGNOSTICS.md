# M45.3 — checkpoint-based Gaussian Flow Matching diagnostics

## Scope

This is **evaluation only** of the saved M45.2 static and dynamic two-layer
checkpoints (job `491707`). No training, model alteration, or changes to
`dnstoles` are performed. The exact Gaussian velocity oracle implemented here
is restricted to the synthetic `block_average` Q/B pair and projected-Gaussian
source; it does not define physical LES/DNS filtering.

## Mathematical setup

The synthetic conditional residual satisfies `Q R = 0` with `P=I-BQ`. For the
block-averaging operator, `P` is an orthogonal projector in each `r**3` fine-site
block. The normalized FM source is `X0=P standard_normal` and target
`X1=R/rms_R`, using **checkpoint** normalization fitted on training data only.

For fixed coarse state `Uc`, the synthetic target is an exactly zero-mean
Gaussian, with covariance

    Sigma1(Uc) = L(Uc) + g g^T.

`L` is the block-diagonal, channel-specific covariance of local fluctuations
after `P` and training RMS normalization. The `shared_mode` case adds a *single*
rank-one factor `g g^T` spanning all fine nodes **and all five channels**. In
`local`, `g=0`. The source covariance is `Sigma0=P`, independently per channel.
The optimal conditional mean FM velocity is

    v*(x,t,Uc) = [t Sigma1-(1-t) Sigma0]
                  [(1-t)**2 Sigma0+t**2 Sigma1]^+ x.

`BlockGaussianOracle` calculates this without constructing a dense global
covariance: it solves independent small `(r**3) x (r**3)` block systems, adds a
block-mean stabilizer (which does not affect nullspace outputs), then applies a
rank-one Woodbury update for the shared random coefficient. The solves use
float64 for improved endpoint accuracy. The oracle requires no neural model.

The analytic expected second moment of the **physical `rhou` residual**
projected onto the spatial template is

    E[a_phi**2] = sum((phi_i * s_i * c_rhou)**2)/(sum(phi_i**2))**2
                  + [case == shared_mode] * (1.7*c_rhou)**2

for the orthogonal block-average projector. It is compared with independently
drawn reference and generated ensembles under the **same fixed coarse field**.

## Diagnostic experiments

1. **Time-resolved FM accuracy:** across all eight held-out coarse fields, with
   multiple fixed noise draws per field, evaluate `t = 0, .1, .25, .5, .75, .9, 1`.
   Record raw network MSE against the FM sampled target, projected-network MSE,
   the exact Gaussian oracle's irreducible sampled-target MSE, the projected
   network's excess MSE against the oracle, and projected `rhou` mode-velocity
   errors. A separate `time_summary.csv` averages noise draws within each
   coarse field before calculating the mean and standard error across held-out
   coarse fields. The oracle itself has **zero** error against the true
   conditional mean, but nonzero sampled-target MSE is expected.
2. **Conditional generation:** choose validation coarse field `val-index=0`;
   generate `128` independently seeded residuals for each static/dynamic model
   using Heun `8`, `25`, and `50` steps, with the exact same initial projected
   noise across models/step counts. Draw 128 independent synthetic **reference**
   residuals for that *same* coarse field, rather than treating unrelated
   validation conditions as one conditional ensemble.
3. **Distribution metrics:** record the `rhou` template's mean and second moment,
   analytical target second moment, unbiased sample variance, pooled RMS per
   physical channel using the **same estimator** for reference/generation, and
   maximum `|Q R|` defect. Record every modal amplitude for independent
   uncertainty/bootstrap analysis. Measure centered two-point `rhou` spatial
   covariance along each periodic Cartesian axis for lags `0..n_f//2`.
4. **Dynamic-feedback ablation:** at the largest Heun step count, evaluate the
   trained dynamic model again with only the fine-to-coarse cross-attention
   contribution replaced by zero. Its dynamic coarse DiNAT and time conditioning
   remain active; this ablation is **not the static architecture**.

No model parameters are modified or optimized. The diagnostic reads the saved
`summary.json` for the exact original synthetic split and architecture settings,
reconstructs validation fields from their recorded PRNG convention, loads the
saved per-model normalization and weights, and rejects non-block-average or
unprojected-source checkpoints for the exact oracle.

## Execution on Grace

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m45_3_checkpoint_diagnostics.slurm
```

Default checkpoint root:

    /scratch/coop/theret/GraphAttention_runs/m45_2_two_layer/job_491707/

Default M45.3 output root:

    /scratch/coop/theret/GraphAttention_runs/m45_3_diagnostics/job_<JOBID>/

A short smoke run (same checkpoint weights; only 8 generation draws, 2 velocity
noise draws) is:

```bash
sbatch --export=ALL,M45_3_GENERATED_SAMPLES=8,M45_3_VELOCITY_DRAWS=2 \
    scripts/slurm/m45_3_checkpoint_diagnostics.slurm
```

For other saved runs use `M45_3_INPUT`; for a different output root use
`M45_3_OUT`; change `M45_3_GENERATED_SAMPLES`, `M45_3_VELOCITY_DRAWS`, or
`M45_3_SEED` to configure the diagnostic. The CLI supports other times, Heun
step lists and validation conditions. Reference data are reconstructed without
accessing target fine values during generation.

## Exact files needed to evaluate

| Purpose | Path |
|---|---|
| Checkpoint diagnostics and analytical oracle | `scripts/m45_checkpoint_diagnostics.py` |
| Fixed-coarse conditional sampler | `src/graph_attention/data/two_scale_synthetic.py` |
| Reused Heun/normalization/mesh functions | `scripts/m45_synthetic_pilot.py` |
| CPU tests | `tests/unit/test_m45_checkpoint_diagnostics.py` |
| Regression tests | `tests/unit/test_m45_synthetic.py`, `tests/unit/test_two_scale_hit.py` |
| **Slurm script** | `scripts/slurm/m45_3_checkpoint_diagnostics.slurm` |
| Documentation | `docs/M45_3_CHECKPOINT_DIAGNOSTICS.md` |
| **Input checkpoints** | `<M45_3_INPUT>/<case>/block_average/<mode>/checkpoint.pt` |
| **Input run metadata** | `<M45_3_INPUT>/summary.json` |
| **Slurm stdout/stderr** | `/scratch/coop/theret/GraphAttention/m45_3_checkpoint_diag_<JOBID>.out/.err` |
| Diagnostic summary | `<M45_3_OUT>/summary.json` |
| Per-time measurements | `<M45_3_OUT>/time_resolved.csv` |
| Per-time means and SEM | `<M45_3_OUT>/time_summary.csv` |
| Ensemble metrics | `<M45_3_OUT>/generation_metrics.csv` |
| Spatial covariance | `<M45_3_OUT>/spatial_correlations.csv` |
| Individual signed mode amplitudes | `<M45_3_OUT>/mode_amplitudes.csv` |

**Return for analysis:** stdout/stderr, `summary.json`, `time_summary.csv`,
`generation_metrics.csv`, `spatial_correlations.csv`, and `mode_amplitudes.csv`.
`time_resolved.csv` is useful for uncertainty analyses but can be read from
Drive if available. Existing checkpoint weights are not rewritten.

## Remaining limitations

- One trained seed per architecture and only eight validation coarse fields;
  no cross-training-seed uncertainty or parameter matching.
- Fixed-coarse conditional ensemble uses a single validation coarse field;
  repeat other coarse conditions to test generality.
- Both training and Heun generation are on the tiny qualification meshes;
  production synchronized patchwise inference is not implemented here.
- This Gaussian reference is a synthetic distribution-level baseline, not a
  proof of Navier–Stokes fidelity or physical super-resolution quality.
- Eight/25/50-step Heun convergence is empirical. A low Heun error does not
  establish that the learned vector field is correct.
