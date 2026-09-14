# M18 — DDPM restart-sweep diagnostic

## Purpose

M18 is a diagnostic experiment for the M13 epsilon-prediction DDPM. It does not
change training, the production sampler, the noise schedule, or the model.

The preceding fixed-timestep and full-trajectory diagnostics showed that the
reverse process departs from the expected marginal scale immediately near the
terminal cosine-schedule region. M18 tests how strongly this failure depends on
the timestep at which the reverse chain is started.

## Source model

The diagnostic is intended for a frozen M13 checkpoint using the existing
`DiffusionDenoisingTask` and the exact adjacent-step ancestral reverse law.

For a forward timestep `t`,

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon.
$$

For each adjacent reverse transition `t -> t-1`, M18 uses the same generalized
DDIM equation as the production task with `eta=1`. For adjacent steps this is
the ancestral DDPM law.

## Restart modes

### Exact forward marginal

For every held-out clean sample, one deterministic Gaussian field is generated
and reused for every selected restart timestep. The diagnostic state is
constructed exactly as

$$
x_{t_s} = \sqrt{\bar\alpha_{t_s}}\,x_0
+ \sqrt{1-\bar\alpha_{t_s}}\,\epsilon.
$$

The reverse chain is then run from `t_s` to zero. This mode answers:

> If the model is placed on a mathematically correct forward marginal at this
> noise level, can its learned reverse dynamics recover a plausible clean
> state?

Because the clean held-out field is used to construct the starting state, the
resulting paired clean-state MSE is a denoising/reconstruction diagnostic. It is
not an unconditional generation metric.

### Gaussian restart

One deterministic `N(0, I)` field per sample is reused as the starting state
for all selected restart timesteps.

At `t=T` this matches the ordinary unconditional initialization. Below `T` it is
only a diagnostic approximation and is not an exact draw from `q(x_t)`. It
answers:

> Does omitting the most terminal reverse transitions remove or reduce the
> observed instability?

This mode must not be presented as a valid replacement sampler without a
separate derivation and generative benchmark.

## Controlled randomness

The two restart modes use distinct deterministic initial random fields.

For the reverse chain, ancestral Gaussian noise is keyed by:

- the global reverse seed;
- sample ID;
- the absolute reverse timestep.

Therefore any two restart runs that share a transition `t -> t-1` receive the
same stochastic perturbation for that transition. This removes an avoidable
source of Monte-Carlo variation when comparing different restart times.

This keyed-per-timestep stream has the same Gaussian transition law as the
production ancestral sampler, but it is not bitwise identical to the
production sampler's sequential RNG stream.

## Default sweep

For `T=1000`, the default configuration evaluates:

$$
1000,\ 999,\ 995,\ 990,\ 980,\ 950,\ 900.
$$

Both restart modes use the same 32 held-out graph templates by default.

## Reported quantities

`restart_sweep.csv` reports per channel and channel-mean quantities including:

- initial mean and standard deviation;
- expected forward-marginal mean and standard deviation;
- initial standard-deviation ratio relative to the expected marginal;
- final mean, standard deviation, RMS, and extrema;
- final mean error normalized by reference standard deviation;
- final standard-deviation ratio and its error from one;
- paired clean-state MSE for the exact-forward mode.

`restart_samples.pt` preserves initial and final standardized tensors for every
mode and restart timestep.

## Interpretation

A sharp improvement when moving the start from `T` to a slightly smaller
`timestep` is evidence that the terminal low-SNR region is a dominant failure
source.

If exact-forward starts remain poor even at substantially lower noise levels,
the problem is broader than terminal initialization and instead implicates the
learned reverse dynamics over a wider SNR range.

If exact-forward starts are healthy while Gaussian restarts remain poor, the
model may be capable of local denoising but unable to bootstrap a valid sample
from an unstructured state at that noise level.

## Implementation and validation

Implementation:

- `scripts/diagnose_diffusion_restart_sweep.py`
- `configs/diagnose_diffusion_restart_sweep.yaml`

Tests:

- `tests/unit/test_diffusion_restart_sweep.py`

M18 is a diagnostic only. No performance or improved-generation claim is made
until target-cluster results are available.
