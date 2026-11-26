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

## Real-mesh and full-size GPU preflight (mandatory before training)

The launcher now refuses A1/A2/A3 until two same-revision gates have completed:

1. `scripts/slurm/m47_10_mesh_preflight.slurm` (Grace CPU): validates the
   immutable M46 qualified sample selection and M47 reference hashes, then
   qualifies all six real AVBP meshes as complete Cartesian periodic hexahedral
   lattices using **M49 `periodic_quotient`**. It compares the unique-node
   coordinates to A2/A3, checks every directed A3 one-hop edge for correct
   signed minimum-image displacement, six different signed neighbors per node,
   and exactly `6*nside^2` wrap edges. It also checks duplicate conservative
   field values on **three train and one validation snapshot per case** (no
   test fields are opened). During actual A2/A3 training every loaded sample
   is independently checked for exact endpoint agreement.
2. `scripts/slurm/m47_10_gpu_preflight.slurm` (Grace GPU): uses the real
   Re94/Ma0.2 `128^3` unique mesh and the production collator to construct
   `6*N` one-hop and `18*N` exact-two-hop edges. It performs a 10-layer
   h128/h4 relative-only `torch_sparse_triton` forward, backward, and one
   AdamW optimizer update with the real mesh/condition and a **synthetic
   standardized Gaussian Flow Matching state**. The throwaway preflight model
   temporarily activates adaLN-Zero residual gates to exercise all sparse
   attention backward kernels; this never changes scientific training weights.
   It checks finite loss/gradients/parameters, nonzero gradients through all
   geometric attention blocks, and logs peak GPU memory and timings.

Both stages emit immutable-identity reports under `$M47_10_ROOT/preflight/`:
`mesh.json`, `gpu.json`, `ready.json`. The training launcher refuses missing,
failed, stale-Git or changed-reference gates. Successful real-mesh/GPU execution
is **not implied by the committed source**: inspect the actual Slurm logs and
reports before interpreting training.

### Launch A1, A2 and A3 as separate Slurm jobs

Each step is a distinct `sbatch` submission, not a Slurm job array.
All three training jobs depend **only** on the successful GPU preflight.
Failure of A1 does not block A2 or A3. The cluster schedules their GPU
resources independently; they are not artificially serialized.

From a **clean and stable repository revision** on the Calypso login node:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --ff-only origin main
git status --short

# Each command creates a new Slurm job and returns a distinct job ID.
MESH=$(sbatch --parsable scripts/slurm/m47_10_mesh_preflight.slurm)
MESH=${MESH%%;*}
GPU=$(sbatch --parsable --dependency=afterok:$MESH \
    scripts/slurm/m47_10_gpu_preflight.slurm)
GPU=${GPU%%;*}

sbatch --dependency=afterok:$GPU scripts/slurm/m47_10_train_a1.slurm
sbatch --dependency=afterok:$GPU scripts/slurm/m47_10_train_a2.slurm
sbatch --dependency=afterok:$GPU scripts/slurm/m47_10_train_a3.slurm
```

Alternatively, `bash scripts/slurm/m47_10_submit_all.sh` performs these
same five **individual** `sbatch` submissions and prints their IDs. It
does not use `--array`.

The training scripts call `m47_10_train_common.sh` with their fixed
variant. The shared worker retains the actual six-case `mach_re`
training configuration, `torch_sparse_triton`, 100 epochs, checkpoints
every 10 epochs and strict resume plus preflight verification. A job that
fails can be independently inspected or resubmitted.

If preflights already passed at the **same exact Git revision** and the
input hashes have not changed, submit only the desired arm:

```bash
sbatch scripts/slurm/m47_10_train_a1.slurm
sbatch scripts/slurm/m47_10_train_a2.slurm
sbatch scripts/slurm/m47_10_train_a3.slurm
```

The shared worker still checks `preflight/ready.json`; omitting the
Slurm dependency does not bypass scientific qualification. If a 12-hour
allocation ends before epoch 100, resubmit only its corresponding arm
script, which restores `last.pt` with strict optimizer and RNG state
checks. The same rule applies to a failed arm after the failure is fixed.

Check each job independently with `squeue` or `sacct` and inspect
`m47_10_<arm>_<jobid>.out/.err` or the `m47_10_mesh` and
`m47_10_gpu` preflight logs under
`/scratch/coop/theret/GraphAttention_runs/`.

**Important:** Because the scientific manifests and preflight reports
are tied to the Git revision, do not update `main` in this checkout
during the campaign. Run it from a dedicated clean checkout if other
development must continue concurrently.

Outputs:

```
/scratch/coop/theret/GraphAttention_runs/m47_10_relative_periodic/
    preflight/{mesh.json,gpu.json,ready.json}
    a1/train_s42/
    a2/train_s42/
    a3/train_s42/
```

Each run saves `last.pt` and `best.pt` every epoch, plus immutable
`epoch_0010.pt`, `epoch_0020.pt`, ..., `epoch_0100.pt`, and original
`history.csv`, `history.json`, `run_signature.json`, `split.json`,
`standardizers.pt`, and `summary.json`. Test fields are not evaluated by the
new campaign trainer (the legacy M43/M47 behavior is unchanged). Evaluate
covariance separately on the frozen validation split with the correct
per-case scale and each checkpoint's recorded node/topology policy. No test
samples are used for checkpoint selection.

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
