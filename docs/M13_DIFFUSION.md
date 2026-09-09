# M13 — diffusion baseline

M13 adds unconditional discrete diffusion for the existing packed HIT 2-D slice
pipeline without changing the M9/M12 graph Transformer architectures.

## Frozen training formulation

Training is epsilon prediction in the statistically standardized,
physically-nondimensionalized conservative state
`[rho, rhou, rhov, rhow, rhoE]`.

For one integer timestep `t` sampled uniformly per physical graph from
`1, ..., T`,

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

with `epsilon ~ N(0, I)`. The model predicts `epsilon`. The diffusion timestep is
supplied through the existing graph-level conditioning path as the normalized
scalar `t / T`, so changing from flow matching to diffusion does not also change
the Transformer time-conditioning architecture.

The first baseline uses:

```text
T = 1000
schedule = cosine
cosine_s = 0.008
prediction = epsilon
```

Validation uses deterministic timestep/noise pairs keyed by validation seed and
sample ID. The optimization objective remains the existing equal-sample spatial
MSE reduction.

## Training and generation are separate

Training does not run the reverse diffusion sampler. It writes checkpoints,
training history, the exact data split, and fitted train-only standardizers.

Example M12 training command:

```bash
python -m scripts.train_slice_diffusion \
  data=hit_slice_pt \
  geometry=dinat_exact2 \
  model=alternating_dilated_geometric_sparse_transformer \
  task=hit_diffusion \
  +generative=hit_slice_diffusion \
  generative.max_epochs=200 \
  run_name=m13_m12_diffusion_200e
```

Generation is performed later from an existing run:

```bash
python -m scripts.generate_slice_diffusion \
  run_dir=/scratch/coop/theret/GraphAttention_runs/m13_m12_diffusion_200e \
  sampling.steps=100 \
  sampling.eta=0.0 \
  sampling.seed=5678
```

This separation allows one trained model to be sampled many times with different
reverse-process costs and stochasticities.

## DDIM eta and ancestral DDPM

The reverse update uses the generalized DDIM parameter `eta`.

```text
eta = 0                 deterministic DDIM
0 < eta < 1             stochastic DDIM
eta = 1, steps < T      stochastic accelerated DDIM
eta = 1, steps = T      exact ancestral-DDPM limit
```

For the controlled flow-matching comparison, use DDIM100 with `eta=0`. This is
100 model evaluations, matching the 100 model evaluations of Heun50 flow
matching.

The full ancestral reference is:

```bash
python -m scripts.generate_slice_diffusion \
  run_dir=/scratch/coop/theret/GraphAttention_runs/m13_m12_diffusion_200e \
  sampling.steps=1000 \
  sampling.eta=1.0 \
  sampling.seed=5678
```

## Generation layout

Each sampling configuration gets an isolated directory under the trained run:

```text
m13_m12_diffusion_200e/
├── best.pt
├── last.pt
├── history.csv
├── summary.json
├── resolved_config.yaml
├── standardizers.pt
├── dataset_split_manifest.json
└── generations/
    ├── ddim_steps100_eta0_seed5678/
    │   ├── generated_test.pt
    │   ├── summary.json
    │   ├── resolved_config.yaml
    │   ├── generation_config.yaml
    │   └── dataset_split_manifest.json
    └── ddpm_ancestral_steps1000_eta1_seed5678/
        └── ...
```

Generation directories are benchmark-ready: point `scripts.benchmark_generation`
at the generation directory itself.

## Unpaired population semantics

Generated IDs are independent deterministic sampling keys such as
`gen_000000`, `gen_000001`, and so on. Reference IDs remain the held-out test
sample IDs. They are stored separately in `generated_test.pt` and do not imply a
sample-to-sample target pairing.

The generation benchmark remains backward compatible with the earlier M9/M12
flow-matching artifacts that used the legacy shared `sample_ids` field.

## Validation gate

Before running a long diffusion experiment:

```bash
pytest
ruff check .
ruff format --check .
```

Then run a one-epoch M12 smoke training and a small-step generation before the
200-epoch controlled comparison.
