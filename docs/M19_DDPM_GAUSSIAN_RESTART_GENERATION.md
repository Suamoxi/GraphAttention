# M19 DDPM Gaussian-Restart Generation

## 1. Purpose

M19 turns the M18 restart diagnostic into an explicit, benchmarkable sampling option for the
existing M13 epsilon-prediction diffusion checkpoint.

The scientific motivation is the measured M18 result that the terminal `1000 -> 999`
transition destabilizes the trained M13 model, while restarting at `t=999` from a standard
Gaussian produces a numerically well-scaled reverse trajectory.

M19 changes **sampling only**. It does not change training, the cosine schedule, epsilon
prediction, model weights, preprocessing, or the reverse update used at retained timesteps.

## 2. Sampling definition

Let the trained diffusion process contain `T` discrete timesteps. The original M13 sampler
initializes

$$
x_T \sim \mathcal N(0,I)
$$

and evaluates a selected reverse-time grid beginning at `T`.

M19 exposes an explicit `start_timestep = t_s <= T`. Sampling instead initializes

$$
\boxed{x_{t_s} \sim \mathcal N(0,I)}
$$

and evaluates only reverse timesteps at or below `t_s`.

For the first controlled experiment,

```text
T = 1000
start_timestep = 999
steps = 999
eta = 1
```

so every adjacent reverse transition

```text
999 -> 998 -> ... -> 1 -> 0
```

is retained and only the terminal `1000 -> 999` transition is removed.

## 3. Scientific status

For `start_timestep = T`, standard-normal initialization is the ordinary terminal diffusion
prior used by M13.

For `start_timestep < T`, standard-normal initialization is **not claimed to be an exact draw**
from the learned/forward marginal `q(x_t)`. It is a project sampling hypothesis motivated by
M18. The approximation is especially plausible near the terminal time because

$$
x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon
$$

has a very small clean-state coefficient when `bar_alpha_t << 1`.

The production generation artifact therefore records:

- `sampling_start_timestep`;
- `sampling_start_t_over_T`;
- `initial_state_distribution = standard_normal`;
- `gaussian_restart_below_terminal`;
- explicit Gaussian-restart semantics.

A restart below `T` must not be reported as exact ancestral DDPM from the terminal prior.
The sampler label is `ddpm_ancestral_gaussian_restart` when every adjacent step from the
restart time is used with `eta=1`.

## 4. Configuration

`configs/generate_slice_diffusion.yaml` adds

```yaml
sampling:
  start_timestep: null
```

`null` preserves the original M13 behavior and resolves to `T`.

The controlled M19 setting is

```yaml
sampling:
  start_timestep: 999
  steps: 999
  eta: 1.0
  seed: 5678
```

Constraints are

```text
1 <= start_timestep <= T
1 <= steps <= start_timestep
0 <= eta <= 1
```

## 5. Implementation

Task-level sampling semantics:

```text
src/graph_attention/tasks/diffusion.py
```

Generation configuration/artifact provenance:

```text
configs/generate_slice_diffusion.yaml
scripts/generate_slice_diffusion.py
```

The original behavior is preserved when `start_timestep` is omitted or null.

## 6. Validation

Unit tests verify:

- the first model-visible normalized time equals the requested absolute restart time;
- adjacent `eta=1` restart sampling is labelled separately from exact terminal ancestral DDPM;
- restart generation remains independent of held-out clean reference values;
- invalid restart/step combinations fail explicitly;
- generation names encode non-terminal restart time.

Relevant tests:

```text
tests/unit/test_diffusion_task.py
tests/unit/test_diffusion_generation.py
```

Target-cluster `ruff` and `pytest` validation remains required after implementation.

## 7. First benchmark experiment

The first benchmark must use the existing M13 L10/1000-epoch `best.pt` checkpoint and the
full held-out test population. This isolates sampling-start behavior from training changes.

Primary comparison:

```text
M13 best.pt, start=1000, steps=1000, eta=1
vs
M13 best.pt, start=999,  steps=999,  eta=1
```

The generation artifact is evaluated by the unchanged `generation_distribution_v2`
benchmark. Primary outputs include marginal normalized errors, local statistics,
correlations, spectra, nearest-reference diagnostics, and physical metrics.

A later `start=990` experiment is allowed as a separate empirical restart candidate, but it
must not be conflated with the minimal one-step removal experiment at `start=999`.

## 8. Assumptions and edge cases

Assumptions:

- model time conditioning remains normalized by the original training `T`, not by the restart
  time;
- the trained model is queried only at timesteps seen during training;
- the standard Gaussian start is independent for each generated sample key;
- sampling below `T` is evaluated as a new inference procedure, not as a training-equivalent
  generative process.

Handled edge cases:

- restart time above `T` fails;
- number of selected steps greater than restart time fails;
- `start_timestep=null` exactly preserves the original terminal start convention;
- non-terminal full adjacent `eta=1` sampling receives an explicit restart sampler label.

Deferred:

- Improved-DDPM learned-variance restart generation;
- restart-aware DDIM cost/quality sweeps;
- principled sampling directly from an estimated non-terminal marginal;
- retraining with a modified terminal schedule;
- any claim that `start=999` improves generation quality before the full benchmark is run.
