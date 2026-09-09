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
    ├── nearest_reference.csv
    ├── physical_metrics.csv
    └── plots/
        ├── marginals/
        ├── spectra/
        └── nearest_reference_fields/
```

Existing benchmark directories are never replaced unless `overwrite=true` is
set explicitly.

## V2 scientific scope

V2 is deliberately limited to the fixed Cartesian 2-D HIT generation artifact.
The primary comparison is an **unpaired population comparison** between
unconditional generated snapshots and the held-out test population in the
physically nondimensionalized conservative-variable space.

The `sample_ids` attached to generated samples are used by the flow-matching
sampler only as deterministic random-number keys. They do not define a generated
snapshot to test-snapshot target pairing. Population statistics, spectra,
Wasserstein distances, and nearest-reference diagnostics therefore never assume
that generated sample `i` should reproduce test sample `i`.

For each conservative channel the benchmark reports pooled mean, standard
deviation, configurable quantiles, mean bias, mean bias normalized by the test
standard deviation, standard-deviation ratio, and empirical one-dimensional
Wasserstein-1 distance. It also reports per-sample spatial mean and standard
deviation, nearest-neighbour correlation, first-difference RMS, and
population-level channel-correlation matrices.

## Spectrum convention

The 2-D spectrum is an orthonormal FFT radial shell-power spectrum. By default
the spatial mean of each sample and channel is removed before the FFT. Only
modes with

```text
0 < |k| <= k_Nyquist,min
```

are included, where `k_Nyquist,min` is the smaller axis Nyquist wavenumber. This
keeps complete circular radial shells common to both Cartesian directions and
does not interpret unresolved wavelengths shorter than the grid Nyquist
wavelength as physical content.

Raw `k` is the primary physical spectrum coordinate and is written directly to
`spectra.csv` and used on the main plot axis. `k/k_Nyquist` is retained as a
secondary coordinate only to define mesh-resolution-independent summary bands:

```text
low:  0.00 <= k/k_Nyquist < 0.25
mid:  0.25 <= k/k_Nyquist < 0.50
high: 0.50 <= k/k_Nyquist <= 1.00
```

For each band the reported ratio is

```text
sum(P_generated) / sum(P_reference)
```

so 1 means the correct integrated spectral power, values below 1 indicate a
deficit, and values above 1 indicate excess power. The full spectrum remains
the authoritative diagnostic; the band ratios are only compact summaries.

## Nearest real snapshot diagnostic

A generated snapshot has no unique physical target. For visualization and a
compact fidelity/coverage diagnostic, V2 instead finds the closest real test
snapshot in an interpretable descriptor space.

Each snapshot descriptor contains, for every channel:

- spatial mean;
- spatial standard deviation;
- nearest-neighbour correlation;
- first-difference RMS;
- integrated low-, mid-, and high-wavenumber spectral powers.

By default it also contains the unique within-snapshot cross-channel
correlations. Every descriptor feature is standardized using **test-population
statistics only**. Features that are non-finite or do not vary across the test
population are omitted. The distance between two snapshots is the RMS Euclidean
distance across the remaining standardized descriptor features.

The benchmark reports three nearest-neighbour distributions:

```text
generated -> test
```

as a fidelity diagnostic,

```text
test -> generated
```

as a coverage diagnostic, and

```text
test -> test, leave one out
```

as the natural real-data nearest-neighbour calibration scale. The summary also
reports the generated-to-test and test-to-generated mean distances divided by
the leave-one-out test-to-test mean distance.

Field examples show each selected generated snapshot beside its
descriptor-nearest real test snapshot. No generated-minus-reference difference
field is plotted because the snapshots are not pointwise paired.

## Physical sanity convention

The physical sanity table derives nondimensional velocity components, kinetic
energy density, and specific internal energy from `[rho, rhou, rhov, rhow,
rhoE]`. This is consistent with the repository's convective
nondimensionalization, where momentum scales with `rho_ref * U_ref` and total
energy with `rho_ref * U_ref^2`.

Ratios are retained for standard deviations and positive bulk quantities. For
velocity means, whose test expectation is close to zero in HIT, the benchmark
does **not** divide generated mean by test mean. It reports the absolute mean
bias and the bias normalized by the corresponding test velocity standard
deviation instead.

## V2 non-goals

The current version does not add MMD, variograms, boundary statistics, generic
compressible-flow divergence metrics, 3-D FFTs, or unstructured-mesh spectra.
Those should be introduced only when a later scientific comparison requires
them.
