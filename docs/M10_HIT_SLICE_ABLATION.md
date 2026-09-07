# M10 HIT Slice M8-vs-M9 Learning Ablation

## Status

M10 implements the first controlled learning comparison between the M8 topology-only sparse Transformer and the M9 relative-displacement sparse Transformer on the precomputed fixed-grid 2-D HIT slice artifacts produced by `diffusion4avbp`.

The adapter now preserves the source artifact's canonical 2-D model geometry exactly. Slicing-axis metadata remains provenance and is not embedded into model coordinates. Software and target-training validation remain pending; no learning-quality claim is made until the experiment has been run with the frozen split and training configuration.

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

`slice_mesh.pt` contains one shared canonical 2-D coordinate tensor plus mesh metadata. It does not contain graph connectivity. The expected coordinate contract is:

```text
coords            [N_slice, 2]
coord_dim         2
grid_shape_2d     [n0, n1]
coordinate_policy shared_physical_reference_2d_mesh
all_samples_share_coords true
```

Each sample file contains:

- `x [N_slice, 5]`;
- the same shared `coords [N_slice, 2]`;
- metadata including `source_stem`, `axis`, `axis_id`, `slice_index`, `slice_coordinate`, `base_slice_index`, channel names, coordinate policy, and grid shape.

The `bXXX` suffix is the configured **base slice index** used by source precomputation. It is not a train/validation/test label. The actual slice index is shifted per source snapshot by the source preprocessing policy.

File names are retained as sample identifiers only. Field semantics and split grouping are read from payload metadata rather than inferred from filename tokens.

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

all slices originating from the same 3-D solution snapshot remain in exactly one split. This prevents train/validation/test leakage between strongly correlated slices from the same source snapshot.

## 4. Canonical 2-D model geometry

The source preprocessing deliberately creates one shared 2-D reference coordinate mesh and stores that same `[N_slice, 2]` tensor in every sample. M10 uses those stored coordinates **without reconstructing a third coordinate**.

For every sample the model geometry is therefore

$$
\boxed{r_i=(x_i,y_i)\in\mathbb R^2.}
$$

For M9, directed edge `j -> i` uses

$$
\boxed{
\Delta r_{ij}=
\begin{bmatrix}
x_j-x_i\\
y_j-y_i
\end{bmatrix}
}
$$

and the M9 geometry MLP therefore has `spatial_dim = 2` for this experiment.

The metadata fields

```text
axis
axis_id
slice_index
slice_coordinate
base_slice_index
```

describe how the 2-D field was extracted from the original 3-D HIT snapshot. They are retained for provenance but are **not model inputs** in the first M8-vs-M9 ablation.

This distinction is important. Reconstructing an x-, y-, or z-normal slice into a 3-D coordinate vector would reveal slice orientation to M9 through the identically zero displacement component, while M8 receives no corresponding orientation information. That would contaminate the intended ablation. The canonical 2-D policy avoids that hidden architectural side channel.

The conservative state still contains the three global momentum components `rhou`, `rhov`, and `rhow`. M10 therefore represents a 2-D spatial slice of a 3-D state, not a two-dimensional CFD simulation. The canonical 2-D geometry is not claimed to be a global 3-D coordinate frame. For this first HIT experiment, slice orientation is deliberately excluded from model conditioning. If orientation is later introduced, it must be a separate documented ablation, for example through explicit conditioning or a slice-local transformation of vector components.

## 5. Graph topology

The source artifact contains no connectivity. M10 defines one deterministic geometry transform for the fixed 2-D Cartesian grid: non-periodic bidirectional 4-neighbour connectivity.

For logical grid node `(a,b)`, edges connect it to valid immediate neighbours

$$
(a\pm1,b),\qquad(a,b\pm1).
$$

Every physical neighbour pair is represented in both message-passing directions.

No diagonal edges, self-loops, k-hop edges, random edges, or periodic wrap edges are added in the first ablation.

Periodic wrap edges are scientifically plausible for the periodic HIT cube but are deliberately deferred so the first M8-vs-M9 learning comparison changes no additional topology mechanism.

## 6. Preprocessing, objective, and reported metrics

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

The optimizer objective is the existing M6 sample-reduced MSE in standardized target space. Validation and test reporting includes both that standardized MSE and the corresponding MSE after inverting only the statistical target scaling, i.e. in physically nondimensionalized `rho/rho_ref` space.

All slices have the same node count in this dataset, but the sample-balanced convention is retained rather than replaced with a fixed-grid special case.

## 7. Reference architecture comparison and matched initialization

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

M8 and M9 use the same files, grouped split, task, physical preprocessing, statistical scalers, batch size, data-order seed, optimizer hyperparameters, epoch budget, and shared parameter initialization.

For seed `s`, the runner first constructs the frozen M8 model with seed `s`. An M8 run uses that state directly. An M9 run constructs its additional geometry parameters with deterministic seed `s+1`, then copies every parameter shared with M8 from the M8 reference state. The common input projection, q/k/v projections, output projections, LayerNorms, Transformer MLPs, and final projection therefore start identically; only M9's geometry MLP parameters have no M8 counterpart.

The only architectural scientific difference is therefore the M9 additive bias from canonical 2-D relative displacement.

A single seed is an initial controlled ablation, not a robustness claim. CUDA sparse reductions are not bitwise deterministic, and multiple independent seeds are required before attributing a small learning difference confidently to the architecture.

## 8. Training runner and artifacts

The dedicated runner is:

```text
scripts/train_slice_ablation.py
```

It reuses the M6 optimizer-step implementation rather than creating a second loss/gradient path. The first runner is deliberately single-process and FP32.

The fixed slice dataset has one common node/edge count per sample, so `batch_size` is an exact proxy for the node/edge load in this dedicated experiment; the general variable-mesh budget machinery remains the repository convention outside this fixed-grid ablation.

Every run writes:

- `resolved_config.yaml`;
- `dataset_split_manifest.json`;
- `standardizers.pt`;
- `history.csv`;
- `best.pt`;
- `last.pt`;
- `summary.json`.

The output directory is created only after dataset, grouped split, and train-only standardizer preflight validation. An existing output directory is never overwritten.

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

- artifacts follow the current `diffusion4avbp` fixed 2-D slice convention;
- `slice_mesh.pt` declares one shared `[N,2]` coordinate mesh, `coord_dim=2`, `grid_shape_2d`, and `coordinate_policy=shared_physical_reference_2d_mesh`;
- each sample stores exactly matching shared 2-D coordinates and the same coordinate policy;
- `source_stem` identifies the original 3-D solution snapshot for leakage-safe grouping;
- slice axis and slice coordinate are provenance only for this experiment and are not model conditioning;
- the source cube passed the equal-axis Cartesian validation used during precomputation;
- the five conservative columns are identified by explicit `channel_names` metadata;
- the existing HIT case reference scales are valid for all slices;
- the first topology is non-periodic 4-neighbour Cartesian connectivity;
- the first runner is single-process/single-GPU and FP32;
- shared M8/M9 parameters are initialized from one M8 reference state for each experiment seed.

## 11. Handled edge cases

The implementation handles:

- x-, y-, and z-extraction provenance while retaining one common 2-D model geometry;
- arbitrary positive rectangular `grid_shape_2d`;
- one-node Cartesian grids with zero graph edges;
- grouped splits with unequal numbers of samples per source group;
- explicit channel-column reordering through metadata names;
- invalid/missing/non-finite fields or coordinates;
- sample/shared-mesh coordinate mismatch;
- inconsistent coordinate policy or coordinate dimension;
- inconsistent axis and axis-id provenance metadata;
- duplicate sample IDs;
- output-directory collision;
- explicit verification that M9's unmatched initialization keys are geometry-MLP parameters only.

## 12. Deferred or unsupported

Deliberately deferred:

- periodic wrap connectivity;
- diagonal/8-neighbour connectivity;
- arbitrary unstructured 2-D slice meshes;
- slice-orientation conditioning;
- transforming global momentum components into a slice-local vector basis;
- artifacts lacking the shared 2-D coordinate-policy metadata;
- DDP training in the dedicated first-ablation runner;
- BF16/FP16 learning validation;
- multiple-seed statistical conclusions;
- temporal prediction, masking, denoising, diffusion, or super-resolution targets;
- physical quadrature weights for this fixed Cartesian slice experiment.

## 13. Failure behavior

The adapter/runner fails rather than guessing when:

- required files are missing;
- shared mesh metadata does not declare the expected 2-D coordinate policy;
- sample coordinates differ from the declared shared mesh;
- sample coordinate dimension/policy differs from the shared mesh;
- requested field names are absent from `channel_names`;
- source grouping or extraction provenance metadata is malformed;
- the case identifier does not match the case definition;
- a grouped split produces no validation or test samples;
- CUDA is requested but unavailable;
- a model other than the frozen M8/M9 classes is requested;
- the common M8/M9 state dictionaries do not match except for M9 geometry parameters;
- an output run directory already exists.

No third coordinate is inferred from `axis`, `slice_index`, or `slice_coordinate`.

## 14. Remaining validation gates

Before making a learning claim:

1. run the repository software suite and Ruff checks;
2. load several real slice artifacts and verify exact `[N,2]` coordinate preservation;
3. confirm grouped split counts and zero `source_stem` overlap;
4. confirm the M9 probe/model uses `spatial_dim=2`;
5. run one-epoch M8 and M9 smoke trainings on Calypso;
6. run the frozen full M8 and M9 training budget;
7. compare validation and held-out test standardized and nondimensional MSE;
8. if the difference is small, repeat with multiple seeds before attribution.
