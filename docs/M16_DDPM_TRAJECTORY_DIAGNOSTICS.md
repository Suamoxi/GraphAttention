# M16 — DDPM reverse-trajectory diagnostics

## 1. Purpose

M15 showed that fixed-timestep epsilon prediction can look numerically accurate while the implied clean-state estimate becomes extremely ill-conditioned at the high-noise end of the cosine schedule. M16 diagnoses the next question: whether and where that local error becomes an actual instability of the reverse sampling trajectory.

M16 does not retrain the model and does not introduce a new sampler. The production `DiffusionDenoisingTask.sample_standardized` implementation is executed unchanged. A model wrapper records the state presented to the denoiser at every reverse timestep and therefore observes the exact states produced by the existing sampler.

## 2. Reference experiment

The default configuration traces the full `T=1000` chain with

- `steps = T`;
- `eta = 1`;
- independent generated sampling keys;
- the persisted trained checkpoint and frozen standardizers;
- 32 generated trajectories in one packed fixed-mesh batch.

With the M13 baseline this is exact ancestral DDPM according to the repository sampler classification.

The number of traced samples is deliberately smaller than the generation benchmark population because the purpose is mechanism diagnosis, not a population-quality claim. It can be increased if the observed trajectory statistics are too noisy.

## 3. No clean-state initialization

Held-out test samples provide the graph geometry, packed structure, and any base conditioning required to call the trained model. Their clean field values do not initialize the generated trajectory.

The reverse state is initialized by the existing sampler from independent sample-keyed Gaussian noise using generated keys

`trajectory_000000`, `trajectory_000001`, ...

The held-out clean test population is used only for diagnostic calibration of expected forward-process moments.

## 4. Expected forward marginal calibration

Let the held-out standardized clean population have channel mean `mu_0` and standard deviation `sigma_0`. For

\[
x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
\]

the corresponding population moments are

\[
\mu_t^{q} = \sqrt{\bar\alpha_t}\,\mu_0,
\]

and, assuming the injected noise is independent of `x0`,

\[
(\sigma_t^{q})^2
= \bar\alpha_t\sigma_0^2 + 1-\bar\alpha_t.
\]

M16 records

\[
\frac{\sigma(x_t^{reverse})}{\sigma_t^{q}}
\]

and the mean discrepancy normalized by `sigma_t^q`. These are calibration diagnostics: the test statistics are never fed back into the reverse update.

The current HIT-slice workflow uses the same number of nodes in every sample, so node-pooled channel moments and equal-sample/equal-node weighting coincide for these marginal statistics. M16 does not generalize this statement to variable-size meshes.

## 5. What is recorded at every model evaluation

For every reverse timestep and every channel, `trajectory_states.csv` records:

- timestep, normalized time, `alpha_bar`, SNR, and `log10(SNR)`;
- generalized-DDIM `sigma` and direction coefficient implied by the configured `eta`;
- expected forward marginal mean and standard deviation;
- reverse-state mean, standard deviation, RMS, and maximum absolute value;
- reverse-state mean and standard-deviation errors relative to the expected forward marginal;
- predicted-epsilon mean, standard deviation, RMS, and maximum absolute value;
- implied `x0_hat` mean, standard deviation, RMS, and maximum absolute value;
- `x0_hat` mean and standard-deviation errors relative to the held-out standardized clean population.

An additional `__mean__` row at each timestep is the arithmetic mean over channels for compact plotting. `trajectory_mean.csv` contains only these rows.

## 6. Actual transition diagnostics

The recorder stores the state seen at consecutive model calls. Therefore the difference

\[
\Delta x_t = x_{t-1}-x_t
\]

is the update produced by the actual production sampler, including both deterministic and stochastic contributions.

`trajectory_transitions.csv` records per-channel:

- RMS and maximum absolute update;
- state RMS/std/max before and after the transition;
- the ratio of post-transition to pre-transition standard deviation.

The final `1 -> 0` transition is measured from the sampler return value, since no model call occurs at `t=0`.

This directly answers whether the first `T -> T-1` update already destabilizes the state or whether instability accumulates later in the chain.

## 7. Tensor snapshots

The full reverse chain is represented in CSV, while selected state tensors are persisted in `trajectory_snapshots.pt` for later field visualization.

For `T=1000`, the default saved timesteps are

`[1000, 999, 998, 995, 990, 980, 950, 900, 750, 500, 250, 100, 50, 10, 1, 0]`.

At model-evaluated snapshot timesteps the artifact stores:

- current standardized sampler state;
- predicted epsilon;
- implied standardized `x0_hat`.

At `t=0` it stores the final standardized generated state.

## 8. Implementation

- CLI: `python -m scripts.diagnose_diffusion_trajectory`
- Config: `configs/diagnose_diffusion_trajectory.yaml`
- Unit tests: `tests/unit/test_diffusion_trajectory_diagnostics.py`
- Existing production sampler: `graph_attention.tasks.DiffusionDenoisingTask.sample_standardized`

The key implementation choice is that M16 wraps the trained model rather than copying the reverse loop. Consequently the traced states are the states produced by the same sampler used for ordinary generation.

The wrapper reconstructs `x0_hat` only for observation. It does not alter the epsilon returned to the production sampler.

## 9. Interpretation ladder

The main cases are:

1. **State scale departs strongly from expected `q(x_t)` immediately at `T -> T-1`.** The terminal low-SNR update is the leading suspect; investigate terminal schedule behavior and prediction parameterization.
2. **The first updates remain calibrated but state/update magnitudes grow progressively.** Error accumulation across the reverse chain is the leading mechanism.
3. **State remains calibrated until a restricted timestep/SNR band.** Focus loss weighting, time conditioning, and prediction accuracy on that band.
4. **State trajectory remains well calibrated but the final generated distribution is still poor.** The problem is then statistical/structural quality rather than numerical trajectory explosion.

M16 reports measurements only. It does not automatically recommend clipping, SNR weighting, `v` prediction, or a modified schedule; each would be a separate scientifically meaningful experiment.

## 10. Assumptions and limitations

- The CLI targets the current fixed 2-D `PrecomputedSlicePTDataset` diffusion workflow.
- The diagnostic is single-process FP32, matching the current generation path.
- The default trace uses 32 generated trajectories; this is a mechanism diagnostic rather than a final distribution benchmark.
- Expected `q(x_t)` moments use held-out test statistics for evaluation only and therefore must not be reused as generation-time corrections.
- Snapshot persistence increases host synchronization and I/O at selected timesteps, so M16 is not a sampler-performance benchmark.
- CUDA/cluster execution and the new unit tests remain unvalidated until the Calypso test gate and target diagnostic job are run.
