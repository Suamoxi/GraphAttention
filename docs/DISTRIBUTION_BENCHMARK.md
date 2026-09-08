# Generation distribution benchmark

The generation benchmark is post-processing only. It consumes an existing
`generated_test.pt` artifact and never reruns training or sampling.

## Run layout

Given a source run such as

```text
/scratch/coop/theret/GraphAttention_runs/m12_dinat_exact2_flow_200e_405553
```

run

```bash
python -m scripts.benchmark_generation \
  run_dir=/scratch/coop/theret/GraphAttention_runs/m12_dinat_exact2_flow_200e_405553
```

The default output is isolated by source run name:

```text
/scratch/coop/theret/GraphAttention_runs/benchmark/
└── m12_dinat_exact2_flow_200e_405553/
    ├── benchmark_config.yaml
    ├── summary.json
    ├── channel_metrics.csv
    ├── sample_statistics.csv
    ├── correlation_matrix.csv
    ├── spectra.csv
    ├── spectral_bands.csv
    ├── physical_metrics.csv
    └── plots/
```

Existing benchmark directories are never replaced unless `overwrite=true` is
set explicitly.

## V1 scientific scope

V1 is deliberately limited to the fixed Cartesian 2-D HIT generation artifact.
The primary comparison is the generated test population against its paired test
target in the physically nondimensionalized conservative-variable space.

For each conservative channel the benchmark reports pooled mean, standard
deviation, configurable quantiles, mean bias, standard-deviation ratio, and
empirical one-dimensional Wasserstein-1 distance. It also reports per-sample
spatial mean and standard deviation, nearest-neighbour correlation,
first-difference RMS, and generated/reference channel-correlation matrices.

The 2-D spectrum is an orthonormal FFT radial shell-power spectrum. By default
the spatial mean of each sample and channel is removed before the FFT. Only
modes with

```text
0 < |k| <= k_Nyquist,min
```

are included, where `k_Nyquist,min` is the smaller axis Nyquist wavenumber. This
avoids treating diagonal corner modes beyond the minimum axis Nyquist as part of
the common resolved radial range.

The default compact spectral bands are defined in normalized wavenumber:

```text
low:  0.00 <= k/k_Nyquist < 0.25
mid:  0.25 <= k/k_Nyquist < 0.50
high: 0.50 <= k/k_Nyquist <= 1.00
```

For each band the reported ratio is

```text
sum(P_generated) / sum(P_target)
```

so 1 means the correct integrated spectral power, values below 1 indicate a
deficit, and values above 1 indicate excess power. The full spectrum remains
the authoritative diagnostic; the band ratios are only compact summaries.

The physical sanity table derives nondimensional velocity components, kinetic
energy density, and specific internal energy from `[rho, rhou, rhov, rhow,
rhoE]`. This is consistent with the repository's convective
nondimensionalization, where momentum scales with `rho_ref * U_ref` and total
energy with `rho_ref * U_ref^2`.

## V1 non-goals

The first version does not add MMD, variograms, boundary statistics, generic
compressible-flow divergence metrics, 3-D FFTs, or unstructured-mesh spectra.
Those should be introduced only when a later scientific comparison requires
them.
