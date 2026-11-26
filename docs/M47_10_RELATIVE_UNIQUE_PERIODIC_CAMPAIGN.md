# M47.10: relative-only / unique-node / periodic-graph campaign

## Scope

Three fresh, *matched* M47 six-regime (`mach_re`) Flow Matching runs:

| Arm | Absolute PE | Nodes | Local graph |
| --- | --- | --- | --- |
| A1 | OFF | Original `65^3/129^3` (duplicated periodic endpoints) | Open |
| A2 | OFF | Unique `64^3/128^3` (M49 `drop_max` reduction) | Open |
| A3 | OFF | Unique `64^3/128^3` | Six-neighbour periodic |

The legacy A0 (absolute-PE baseline) is **not** retrained or modified.
All arms retain the same six M47-qualified snapshot populations, per-case
physical references, frozen chronological split, `mach_re` conditioning,
**identical train-only sample-balanced channel moments fitted on unique physical nodes for all arms**, original conservative channels,
straight Gaussian Flow Matching loss, independent standard Gaussian source,
AdamW learning rate `1e-4`, seed 42, 10-layer h128/h4 DiNAT, equal-case
six-sample gradient accumulation, and 100 epochs. The sparse attention
backend is `torch_sparse_triton` for all three new arms.

A1 versus A2 changes the number of physical DOFs and the independently
sampled Gaussian dimensions. This is intrinsic to eliminating stored duplicate
nodes, **not** a correlated-noise prior. A3 has the same source distribution
and physical DOFs as A2; only the one-hop/two-hop graph and correct
minimum-image displacements change.

## Implemented mechanism

- Reuse M49 `reduce_cartesian_endpoints` and strict `reduce_sample` equality
  validation. A2/A3 remove max-coordinate planes and reject any original
  field whose periodic duplicates disagree.
- A3 maps a 3-D periodic Cartesian six-neighbour graph back to the original
  **arbitrary exporter node order**, with validation of complete lattice
  support. Graph construction is cached per case mesh. Exact-two-hop
  attention is rebuilt by the existing `GraphTaskCollator` from the periodic
  one-hop edges.
- A3 stores the **physical domain length divided by that case's `L_ref`**
  in mesh metadata. Both M43/M47 training forwards and the Flow Matching
  sampler use it to compute minimum-image relative edge displacements.
- A1 and A2 retain the historical no-periodic-length forward, unchanged.
- M43/M47 historical runs retain all previous defaults and checkpoints.
- Each new run's strict signature captures variant, effective model settings,
  checkpoint interval and source-distribution policy. Code revision is locked
  on resume; do not update the repository mid-run.
- Independent Gaussian source is unchanged. No prior tying, extra loss,
  projection, state masking or noise-model experiment is introduced.

## Launch / resume

Run on CERFACS Grace in the production Singularity container:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --ff-only origin main
git status --short
sbatch scripts/slurm/m47_10_ablation_train.slurm
```

Array indices: `0=A1`, `1=A2`, `2=A3`. By default tasks run sequentially
(`%1`) to avoid GPU contention. Submitting one arm separately:

```bash
sbatch --array=0 scripts/slurm/m47_10_ablation_train.slurm
sbatch --array=1 scripts/slurm/m47_10_ablation_train.slurm
sbatch --array=2 scripts/slurm/m47_10_ablation_train.slurm
```

If a 12-hour allocation ends before 100 epochs, **resubmit the same index**
at the same Git revision. The launcher detects a completed `last.pt` and
uses strict checkpoint/optimizer/Flow Matching RNG restoration. An interrupted
incomplete epoch is replayed. Concurrent writers to the same arm are blocked.

Outputs:

```
/scratch/coop/theret/GraphAttention_runs/m47_10_relative_periodic/
    a1/train_s42/
    a2/train_s42/
    a3/train_s42/
```

Each run saves `last.pt` and `best.pt` every epoch, plus immutable
`epoch_0010.pt`, `epoch_0020.pt`, ..., `epoch_0100.pt`, and original
`history.csv`, `history.json`, `run_signature.json`, `split.json`,
`standardizers.pt`, and `summary.json`. Test fields are not evaluated by the new campaign
trainer (the legacy M43/M47 behavior is unchanged). Evaluate covariance separately
on the frozen validation split with the correct per-case scale and this
checkpoint's recorded data/topology policy. No test samples are used
for checkpoint selection.

## Important interpretation constraints

- Re54 is always `65^3` stored nodes, and Re94 always `129^3` stored nodes;
  **resolution and Reynolds number remain confounded**.
- All six cases are in the training population. No claim of unseen-regime
  generalization follows from within-regime validation alone.
- A1 has absolute PE disabled but still includes duplicate physical DOFs;
  its endpoint mismatches can persist despite improved interior physics.
- The A3 periodic graph is specific to verified *Cartesian, cubic, periodic*
  HIT meshes and must not be applied to unstructured arbitrary geometry.
- To isolate optimization from representation, compare fixed checkpoints
  every 10 epochs; use source-consistent metrics, normalized spectral bands,
  and the same validation sample IDs and generation seed across arms.
