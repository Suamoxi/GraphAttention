# M45: Synthetic operator-consistent Flow Matching pilot

## Motivation and limits

This small 3D periodic HIT-like pilot tests the **static** and **dynamic**
TwoScaleDiNAT architectures before `dnstoles` has finalized the physical
restriction Q and right-inverse reconstruction B. Synthetic data are not DNS;
there is no physical LES filtering, Navier–Stokes evolution or real turbulent
spectrum. Results are architecture and numerical-engineering diagnostics, not
claims of physical super-resolution skill.

No modifications to `Suamoxi/dnstoles` are made by this experiment.

## Geometry and Q/B

For unique periodic Cartesian nodes `n_f=r*n_c` in each dimension (no duplicated
periodic endpoint), the pilot supports two **explicit, alternative** Q/B pairs:

- `block_average`: Q averages the r^3 fine sites in each coarse block;
  B repeats the associated coarse value on the fine sites.
- `anchor_trilinear`: Q picks fine nodes that coincide with coarse anchors;
  B interpolates with the existing M45 periodic trilinear support.

Both satisfy `Q B=I`, but neither should be mistaken for dnstoles physical
coarse-graining. The same Q/B is used to synthesize targets, to reconstruct
the baseline and to measure generated coarse consistency. The learned
cross-attention edges are **independent** of the physical Q/B operator.

Five dimensionless coarse conservative-like channels are constructed as smooth
random-phase low-wavenumber periodic fields. Fine truth is

    Uf = B Uc + Rf,     Rf = (I - B Q) candidate.

Two synthetic cases:

- `local`: candidate consists of coarse-amplitude-modulated, locally sampled
  Gaussian subgrid noise. The Q/B projection introduces local constraints.
- `shared_mode`: the same local component plus one **single random scalar**
  multiplying a prescribed high-frequency spatial template over the whole
  domain. This introduces an unresolved, long-range stochastic correlation
  which is invisible to Uc because Q Rf=0.

The same training fields, splits, normalization, FM stochastic paths, selected
fine owned cores, optimization settings and initial shared parameters are used
for static and dynamic models. Dynamic has extra parameters and compute; this
pilot is **not** a parameter- or compute-matched scientific benchmark.

Train samples and validation samples use disjoint PRNG seeds. Normalization
`mean_c`, `std_c`, and per-channel `rms_R` is fitted only on train samples.

## FM and coarse consistency

Default `--source-subspace nullspace` draws epsilon in `ker Q` using

    Pi = I - B Q,  epsilon_0 = Pi standard_normal,
    X_t = (1-t)*epsilon_0 + t*(Rf / rms_R),  target_v = X_1-epsilon_0.

Channelwise `rms_R` commutes with Q. Training still uses an ordinary owned-node
MSE on the **raw** velocity output, without inserting a Q/B projection into the
model's local backward graph. During small-graph Heun generation, both initial
noise and the complete **global** predicted velocity are projected by Pi, so
`Q R_generated = 0` up to numerical precision. This projection is local for
these synthetic operator families but a real dnstoles B may not be local.

Use `--source-subspace full` for an ablation using ordinary full Gaussian
source and no projected inference velocity. In that mode coarse consistency
is **not** enforced and its defect is a diagnostic.

## Partitioned training and inference

Fine owned cubes are split using `--core-side` (must divide `n_f`). Each
architecture computes its exact M45 dependency closure and slices both graphs
and bipartite supports. Only the fine owned nodes contribute to the training
loss. Coarse context may overlap between independently trained fine cores.
Each local training path draws a global Gaussian vector **only for the small
qualification mesh** and then crops it, preserving consistent shared halo
inputs. The production-sized target-driven M41 planning is still pending.

Validation and the small full-graph Heun generator evaluate the **whole small
mesh**; no stage-synchronized production inference is claimed.

## Measured outcomes

- Held-out fine FM velocity MSE (paired with independently drawn validation
  time and Gaussian source). This is an appropriate supervised FM vector-field
  metric, *not* a paired generative field reconstruction metric.
- Oracle versus generated ensemble second moment of the spatial amplitude of
  the prescribed shared high-frequency mode (in one momentum channel). A
  diagnostic of long-range stochastic coherence.
- Oracle/generated channel RMS and max `|Q R_generated|`.
- Number of parameters, training time, and fine/core read-set ratio.

Tiny default training runs are **pipeline smoke tests**. They should not be
ranked scientifically. The global-mode ensemble second moment has high Monte
Carlo uncertainty with four generated samples.

## Run on Grace

Both architecture modes and both synthetic cases are run within one job:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m45_synthetic_pilot.slurm
```

Optionally change Q/B family (a distinct Slurm job):

```bash
sbatch --export=ALL,M45_SYNTHETIC_OPERATOR=anchor_trilinear scripts/slurm/m45_synthetic_pilot.slurm
```

The job uses `grace`, `pyg25.03.sif`, and
`/scratch/coop/theret/avbpML_env/bin/python3`. It first runs the real M45
model tests and the synthetic Q/B tests, then performs training and small
Heun generation on CUDA.

Set `M45_SYNTHETIC_REFINEMENT=3` for the r=3 experiment; use an appropriate
fine `M45_SYNTHETIC_CORE_SIDE` (default 4 divides `3*4=12`).
Set `M45_SYNTHETIC_STEPS=...` to increase the number of optimizer updates
**per architecture and synthetic case**, or `M45_SYNTHETIC_SOURCE_SUBSPACE=full`
for the unconstrained ablation. Do not use the login node for training.

## Exact files for evaluation

| Purpose | Repository file / Grace output |
|---|---|
| Synthetic Q/B and fields | `src/graph_attention/data/two_scale_synthetic.py` |
| Paired pilot trainer/generator | `scripts/m45_synthetic_pilot.py` |
| Unit tests | `tests/unit/test_m45_synthetic.py` |
| Prior M45 tests | `tests/unit/test_two_scale_hit.py` |
| Slurm entry point | `scripts/slurm/m45_synthetic_pilot.slurm` |
| Slurm stdout/stderr | `/scratch/coop/theret/GraphAttention/m45_synthetic_pilot_<JOBID>.out` / `.err` |
| Experiment root | `/scratch/coop/theret/GraphAttention_runs/m45_synthetic/job_<JOBID>/` |
| Overall results | `<experiment-root>/summary.json`, `<experiment-root>/metrics.csv` |
| Individual metrics | `<experiment-root>/<case>/<operator>/<mode>/metrics.json` |
| Training curves | `<experiment-root>/<case>/<operator>/<mode>/train_history.csv` |
| Checkpoints | `<experiment-root>/<case>/<operator>/<mode>/checkpoint.pt` |

**Send back** both Slurm logs, `summary.json`, `metrics.csv`, and the four
`train_history.csv` files. Checkpoints are not needed for an initial analysis
unless we want to resume the same pilot or inspect learned features.

## Required next scientific gates

1. Qualify synthetic pilot forward/backward and inference on the real Grace GPU.
2. Increase train/val snapshot counts, optimizer steps, and sampling ensemble;
   report uncertainty across seeds and capacity-matched baselines.
3. Test the same model under r=2 and r=3 rather than assume scale extrapolation.
4. Replace the synthetic Q/B pair with the audited, physical `dnstoles` pair,
   without changing the attention or Flow Matching architectural contract.
5. Implement production synchronized partitioned Heun and Q/B projections for
   each physical operator, or explicitly retain approximate unconstrained SR.
