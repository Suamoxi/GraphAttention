# M10 HIT Slice M8-vs-M9 Learning Ablation

## Status

Implementation is present for the first controlled learning comparison between the M8 topology-only sparse Transformer and the M9 relative-displacement sparse Transformer on the precomputed fixed-grid 2-D HIT slice artifacts produced by `diffusion4avbp`.

Software and target-training validation are pending. No learning-quality claim is made until the experiment has been run with the frozen split and training configuration.

## 1. Purpose

M8 and M9 have already been compared for implementation correctness and target GPU cost. M10 asks the first learning question:

$$
\boxed{
\text{Does relative displacement improve prediction over topology-only attention?}
}
$$

The controlled task is

$$
[\rho u,\rho v,\rho w,\rho E]\rightarrow \rho.
$$

This task is intentionally simple. It reuses the existing deterministic node-regression, physical nondimensionalization, train-only statistical scaling, and equal-sample MSE machinery. It does not introduce diffusion noise, masking, temporal pairing, super-resolution, or a second new architectural mechanism.

## 2. Source artifact convention

The source artifacts are produced by `diffusion4avbp/scripts/precompute_cartesian_slice_pt_dataset.py`.

The dataset layout is:

```text
HIT_LES_FORCED_SLICED/
├── meshes/
│   └── slice_mesh.pt
└── samples/
    ├── slice_<source_suffix>_<axis><slice_index>_b<base_slice_index>.pt
    └── ...
```

`slice_mesh.pt` contains the shared 2-D coordinate tensor plus mesh metadata. It does not contain graph connectivity.

Each sample file contains:

- `x [N_slice, 5]`;
- the same shared `coords [N_slice, 2]`;
- metadata including `source_stem`, `axis`, `axis_id`, `slice_index`,
  `slice_coordinate`, `base_slice_index`, channel names, and grid shape.

The `bXXX` suffix is the configured **base slice index** used by the source precomputation. It is not a train/validation/test label. The actual slice index is shifted per source snapshot by the source precomputation policy.

File names are retained as sample identifiers only. Scientific field semantics, orientation, and split grouping are read from metadata rather than parsed from filename tokens.

## 3. Grouped split

The split reproduces the `diffusion4avbp` grouped-split convention.

Each sample exposes

```text
metadata.source_stem
```

as its split group. Unique source groups are collected in first-occurrence order, deterministically shuffled with `random.Random(seed)`, then partitioned by integer-truncated group counts.

With the reference configuration:

```text
seed             = 42
train_ratio      = 0.70
validation_ratio = 0.15
test_ratio       = remainder
```

all slices originating from the same 3-D solution snapshot remain in exactly one split.

This is required to avoid train/validation/test leakage between strongly correlated slices of the same source snapshot.

## 4. Physical coordinate reconstruction

The source slice artifacts deliberately use one shared 2-D reference coordinate mesh for x-, y-, and z-normal slices. The source precomputation validates that the original Cartesian cube has identical coordinate arrays along x, y, and z.

For M9, the shared 2-D coordinates are embedded back into the original global Cartesian frame using the slice metadata.

For an x-normal slice:

$$
r=(x_{\mathrm{slice}},y,z).
$$

For a y-normal slice:

$$
r=(x,y_{\mathrm{slice}},z).
$$

For a z-normal slice:

$$
r=(x,y,z_{\mathrm{slice}}).
$$

This keeps the geometric displacement components in the same global frame as the conservative momentum components `rhou`, `rhov`, and `rhow`.

No learned model sees the filename or raw slice index.

## 5. Graph topology

The source artifact contains no connectivity. M10 therefore defines one deterministic geometry transform for the fixed 2-D Cartesian grid: non-periodic bidirectional 4-neighbour connectivity.

For logical grid node `(a,b)`, edges connect it to valid immediate neighbours

$$
(a\pm1,b),\qquad(a,b\pm1).
$$

Every physical neighbour pair is represented in both message-passing directions.

No diagonal edges, self-loops, k-hop edges, random edges, or periodic wrap edges are added in the first ablation.

Periodic wrap edges are scientifically plausible for the periodic HIT cube but are deliberately deferred so that the first M8-vs-M9 learning comparison changes no additional topology mechanism.

## 6. Preprocessing and objective

Every sample uses the existing `HIT_LES_FORCED` case definition.

The preprocessing order remains:

$$
\text{raw conservative fields}
\rightarrow
\text{physical nondimensionalization}
\rightarrow
\text{train-only sample-balanced standardization}.
$$

The task inputs are ordered as:

```text
rhou.x
rhov.y
rhow.z
rhoE.value
```

and the target is:

```text
rho.value
```

The loss is the existing M6 sample-reduced MSE. All slices have the same node count in this dataset, but the sample-balanced convention is retained rather than replaced with a fixed-grid special case.

## 7. Reference architecture comparison

The intended first comparison uses:

```text
hidden_dim = 128
num_heads  = 8
num_layers = 4
mlp_ratio  = 4
dtype      = FP32
optimizer  = AdamW
lr         = 1e-4
weight_decay = 0
```

M8 and M9 use:

- the same files;
- the same grouped split;
- the same task;
- the same physical preprocessing;
- the same statistical scalers;
- the same batch size;
- the same data-order seed;
- the same optimizer hyperparameters;
- the same epoch budget.

The only architectural scientific difference is the M9 additive relative-displacement score bias.

A single seed is an initial controlled ablation, not a robustness claim. Multiple seeds are required before attributing a small learning difference confidently to the architecture.

## 8. Training runner and artifacts

The dedicated runner is:

```text
scripts/train_slice_ablation.py
```

It intentionally reuses the M6 optimizer-step implementation rather than creating a second loss/gradient path.

Every run writes:

- `resolved_config.yaml`;
- `dataset_split_manifest.json`;
- `standardizers.pt`;
- `history.csv`;
- `best.pt`;
- `last.pt`;
- `summary.json`.

The output directory must not already exist, preventing silent overwrite of experiment evidence.

## 9. Reference commands

M8:

```bash
python scripts/train_slice_ablation.py \
  data=hit_slice_pt \
  task=hit_density_regression \
  model=sparse_transformer \
  +ablation=hit_slice \
  model.hidden_dim=128 \
  model.num_heads=8 \
  model.num_layers=4 \
  model.mlp_ratio=4 \
  optimizer.lr=1e-4 \
  optimizer.weight_decay=0.0 \
  run_name=m8_hit_slice
```

M9:

```bash
python scripts/train_slice_ablation.py \
  data=hit_slice_pt \
  task=hit_density_regression \
  model=geometric_sparse_transformer \
  +ablation=hit_slice \
  model.hidden_dim=128 \
  model.num_heads=8 \
  model.num_layers=4 \
  model.mlp_ratio=4 \
  optimizer.lr=1e-4 \
  optimizer.weight_decay=0.0 \
  run_name=m9_hit_slice
```

These commands define the intended comparison only after the software gate confirms that the artifact paths and exact source metadata match the assumptions below.

## 10. Assumptions

Introduced or inherited assumptions:

- the artifacts were produced by the current `diffusion4avbp` fixed 2-D slice precomputation convention;
- `slice_mesh.pt` declares one shared `[N,2]` coordinate mesh and `grid_shape_2d`;
- each sample stores exactly matching shared 2-D coordinates;
- sample metadata, not filenames, is authoritative for source grouping and orientation;
- `source_stem` identifies the original 3-D solution snapshot;
- the original source cube passed the equal-axis Cartesian validation used during precomputation;
- the five conservative columns are identified by explicit `channel_names` metadata;
- the existing HIT case reference scales are valid for all slices;
- the first topology is non-periodic 4-neighbour Cartesian connectivity;
- the first runner is single-process/single-GPU and FP32.

## 11. Handled edge cases

The implementation handles:

- x-, y-, and z-normal slices;
- arbitrary positive rectangular `grid_shape_2d`;
- one-node Cartesian grids with zero graph edges;
- grouped splits with unequal numbers of samples per source group;
- explicit channel-column reordering through metadata names;
- invalid/missing/non-finite fields or coordinates;
- sample/shared-mesh coordinate mismatch;
- inconsistent axis and axis-id metadata;
- duplicate sample IDs;
- output-directory collision.

## 12. Deferred or unsupported

Deliberately deferred:

- periodic wrap connectivity;
- diagonal/8-neighbour connectivity;
- arbitrary unstructured 2-D slice meshes;
- slice artifacts lacking authoritative orientation/group metadata;
- DDP training in the dedicated first-ablation runner;
- BF16/FP16 learning validation;
- multiple-seed statistical conclusions;
- temporal prediction, masking, denoising, diffusion, or super-resolution targets;
- physical quadrature weights for this fixed Cartesian slice experiment.

## 13. Failure behavior

The adapter fails rather than guessing when:

- required files are missing;
- shared mesh metadata is inconsistent;
- sample coordinates differ from the declared shared mesh;
- requested field names are absent from `channel_names`;
- group/orientation metadata is missing;
- the case identifier does not match the case definition;
- a grouped split produces no validation or test samples;
- CUDA is requested but unavailable;
- an output run directory already exists.

## 14. Remaining validation gates

Before making a learning claim:

1. run the repository software suite and Ruff checks;
2. inspect at least several real slice artifacts through the new adapter;
3. confirm grouped split counts and zero `source_stem` overlap;
4. run one-epoch M8 and M9 smoke trainings on Calypso;
5. run the frozen full M8 and M9 training budget;
6. compare validation and held-out test MSE;
7. if the difference is small, repeat with multiple seeds before attribution.
