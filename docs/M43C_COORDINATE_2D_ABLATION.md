# M43C — Controlled absolute versus relative-only 2-D DiNAT-DiT ablation

**Status:** isolated code, tests, Slurm and postprocessing prepared. No
Calypso 2-D training/inference results exist yet. The 3-D M43-B checkpoint
and original scientific training runner remain unchanged.

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

After BOTH training jobs finish, run matched test-population generation
and existing physical-statistics benchmarks:

~~~bash
sbatch --export=ALL,M43C_REPO="$PWD",M43C_STAGE=evaluate,M43C_VARIANT=absolute \
  scripts/slurm/m43c_coordinate_2d_ablation.slurm
sbatch --export=ALL,M43C_REPO="$PWD",M43C_STAGE=evaluate,M43C_VARIANT=relative \
  scripts/slurm/m43c_coordinate_2d_ablation.slurm
~~~

Then strictly compare both science/performance results:

~~~bash
singularity exec --nv -B /home -B /scratch \
  /softs/local_arm/singularity/images/pyg25.03.sif \
  /scratch/coop/theret/avbpML_env/bin/python3 \
  -m scripts.compare_m43c_coordinate_2d \
  --root /scratch/coop/theret/GraphAttention_runs/m43c_coordinate_2d \
  --epochs 100
~~~

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
  PDF, and mean/10-90% velocity energy-spectrum PDF.

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
