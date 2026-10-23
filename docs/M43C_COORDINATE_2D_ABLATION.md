# M43C — Controlled absolute versus relative-only 2-D DiNAT-DiT ablation

**Status:** the two training runs have been reported complete by the user;
generation, distribution benchmarking and matched quality/speed comparison
are the next gates. This document does not imply that those inference jobs
have already run. The 3-D M43-B checkpoint and its runner remain unchanged.

## Scope

Compare two otherwise-identical M33-like Flow Matching models on the
pre-existing HIT_LES_FORCED_SLICED 2-D dataset.

**absolute**: original centered-bbox coordinate MLP is added to the node
embedding. **relative**: skip global bounding-box normalization and absolute
coordinate MLP in forward entirely, retaining the original relative
edge-displacement attention bias. Both use the same one-hop and exact-two-hop
connectivity, no self-attention, scatter backend, 10 DiT blocks, h128/H4,
Flow Matching loss, seed 42, batch size 128, and identical split and
fitted standardizers.

For matched initialization, the relative-only model retains a dormant
coordinate projection module *in the state dict* although it is never
evaluated and receives no gradients. Shared weights are initialized
identically. The historical default is absolute=true, preserving all
existing checkpoint behavior. Do not conflate this functional ablation
with a future parameter-count reduction.

Training comparison: 100 completed epochs, separate NEW run directories.
Generation comparison: held-out test population, Heun 50 steps = 100
model evaluations, sampling seed 5678, identical graph order.
Existing benchmarking: per-variable marginals, normalized Wasserstein,
physical-k energy spectrum, 10-90% between-snapshot variability bands,
and 2-D slice visualizations. This is **unconditional 2-D generation**,
not a paired LES/DNS super-resolution test.

## Running on Grace without modifying ongoing 3-D training

~~~bash
cd /scratch/coop/theret/GraphAttention
git fetch origin
git worktree add --detach ../GraphAttention_m43c_coords \
  origin/feature/m43c-relative-only-2d-ablation
cd ../GraphAttention_m43c_coords
~~~

Test only (no datasets loaded):

~~~bash
sbatch --export=ALL,M43C_REPO="$PWD",M43C_STAGE=tests \
  scripts/slurm/m43c_coordinate_2d_ablation.slurm
~~~

Train the matched pair from scratch, using separate checkpoints:

~~~bash
sbatch --export=ALL,M43C_REPO="$PWD",M43C_STAGE=train,M43C_VARIANT=absolute \
  scripts/slurm/m43c_coordinate_2d_ablation.slurm
sbatch --export=ALL,M43C_REPO="$PWD",M43C_STAGE=train,M43C_VARIANT=relative \
  scripts/slurm/m43c_coordinate_2d_ablation.slurm
~~~

## Evaluate the two COMPLETED training runs

Update the existing detached experiment worktree. This does **not** alter
the independently saved training checkpoints:

~~~bash
cd /scratch/coop/theret/GraphAttention_m43c_coords
git fetch origin
git checkout --detach origin/feature/m43c-relative-only-2d-ablation
ls -lh /scratch/coop/theret/GraphAttention_runs/m43c_coordinate_2d/m43c_{absolute,relative}_s42_e100/best.pt
~~~

Submit the two evaluation jobs, then submit a **CPU-only comparison Slurm job**
that starts only if both evaluations succeed:

~~~bash
ABS_JOB=$(sbatch --parsable --export=ALL,M43C_REPO="$PWD",M43C_STAGE=evaluate,M43C_VARIANT=absolute \
  scripts/slurm/m43c_coordinate_2d_ablation.slurm)
REL_JOB=$(sbatch --parsable --export=ALL,M43C_REPO="$PWD",M43C_STAGE=evaluate,M43C_VARIANT=relative \
  scripts/slurm/m43c_coordinate_2d_ablation.slurm)
COMPARE_JOB=$(sbatch --parsable --dependency=afterok:"$ABS_JOB":"$REL_JOB" \
  --export=ALL,M43C_REPO="$PWD" \
  scripts/slurm/m43c_coordinate_2d_compare.slurm)
echo "Absolute: $ABS_JOB | Relative-only: $REL_JOB | Comparison: $COMPARE_JOB"
~~~

The combined evaluation stage first generates the test population from
best.pt with 50-step Heun (100 model evaluations), then benchmarks it.
Each evaluation also supports separate M43C_STAGE=generate and
M43C_STAGE=benchmark reruns without overwriting completed outputs.

The comparison job requests **four CPUs, 24 GiB of RAM, and zero GPUs**
on the existing ARM64 `grace` partition, matching the repository's
CPU-only plotting job conventions (`--gres=gpu:0`).

SLURM `afterok` dependencies enforce that both evaluation jobs return
successfully before the comparison begins. If your two evaluation jobs
have **already completed**, submit only the comparison job:

~~~bash
sbatch --export=ALL,M43C_REPO="$PWD" \
  scripts/slurm/m43c_coordinate_2d_compare.slurm
~~~

The comparison is read-only toward both trained models and both generated
test populations. Its results are written to a new comparison directory,
and it refuses to overwrite an existing result. It compares scientific
splits, scalers, generated-sample keys, sampler settings, actual reference
tensor values, and then produces a JSON report and PDF plots.

The comparison refuses runs with different splits, standardizers,
configurations beyond the position flag, reference populations, or
spectral k bins. It produces comparison.json and two PDF plots
(validation MSE; physical-k population energy spectra).

## Output layout

Root: /scratch/coop/theret/GraphAttention_runs/m43c_coordinate_2d

- m43c_absolute_s42_e100/ and m43c_relative_s42_e100/: distinct
  checkpoints, histories, TensorBoard, train-only normalizers and held-out
  generations.
- benchmark_absolute/ and benchmark_relative/: standard CFD evaluation,
  including per-field PDFs and 2-D slices from the existing benchmark.
- coordinate_comparison_e100/: strict comparison JSON, validation-loss
  PDF, physical-k mean/10-90% velocity energy-spectrum PDF, direct
  normalized-space Wasserstein comparison PDF, and one direct
  absolute/relative/reference marginal PDF per conservative channel.

## Interpreting the result

This is a clean architectural test of translation/position priors on
homogeneous HIT slices, **not** evidence that absolute position is
universally useless. Near-wall flows, geometric boundaries, or moving
meshes may need physically meaningful positional context.

The speed gain from skipping the coordinate MLP may be small. The 129^3
profile's dominant indexed backward remains an independent adaLN
broadcast problem. Keep this ablation separate from the proposed
single-graph expansion fast path to avoid confounding effects.

All runs and checkpoint products remain separate from existing 3-D
M43-B training. No HPC workload is started automatically on commit.

## What the evaluation can and cannot prove

The primary comparison includes best/final Flow Matching validation MSE,
generation time (including setup and sample I/O), frozen train-standardized
per-variable Wasserstein-1 distances, and the reference-relative velocity
energy spectrum on actual physical k. Compare spectral variance bands
carefully: their quantiles are across independent generated snapshots.

Each arm's existing benchmark also produces conservative-field PDFs and
2-D field plots. Do not interpret generated/nearest-reference side-by-side
fields as paired prediction errors; generation is an **unpaired distribution
comparison**, and the paired ablation only shares the initial Gaussian noise
stream between its two network architectures.

**Physical admissibility gate:** the velocity energy-spectrum code needs
positive finite generated density. If a model produces rho <= 0 at any
node, the energy/physics benchmark may fail. This is a scientific failure,
not a license to clip or hide bad nodes. Preserve the generated tensor
and report the invalid-density fraction; only re-run after deciding how
to handle the resulting invalid physics.

**No training restart:** evaluating these runs uses checkpoint best.pt
read-only; source history/scalers/splits remain unchanged.
