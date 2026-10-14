# M41.4 — controlled computational scalability experiment (GH200)

This experiment **does not evaluate physical/generative quality**. It benchmarks
exact owned-query one-hop and exact-two-hop M37 DiNAT-DiT computation while
varying partition count and the CPU metadata loading policy.

## Provenance and no duplicate global graph

- Input: previously verified 16,974,593-node Re150/Ma0.3 mesh.
- Immutable source:
  /scratch/coop/theret/GraphAttention_runs/m41_4_257_generic_graph
  Its global disk-backed one-hop CSR is **reused** in each layout. All
  new coords.npy and csr entries point to this source via symlinks;
  no global exact-two-hop graph and no Cartesian assumption.
- Spatial ownership layouts, chosen manually as in earlier M41:

| Partitions | Bounding-box spatial bins | Maximum owned guard |
|---|---|---:|
| 16 | 4 x 2 x 2 | 1,200,000 |
| 32 | 4 x 4 x 2 | 650,000 |
| 64 | 4 x 4 x 4 | original, no regeneration |
| 128 | 8 x 4 x 4 | 180,000 |

All layouts must preserve 101,451,264 directed one-hop and 303,167,994
directed exact-two-hop edges, except configured self loops (none for the M37
checkpoint). The prepare script asserts exact equality to the original graph.

## Step 1 — build only alternative partition plans

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m41_4_257_scaling_prepare.slurm
~~~

The job runs new unit tests (and original numerical M37 tests on
irregular hex/tet graphs) before computing 16/32/128 partition plans.
The old 64-partition plan and its CSR remain untouched. A preparation
failure leaves a directory without prepare.json; manually inspect it
rather than silently overwriting or treating it as complete.

## Step 2 — control for run-to-run variation at 64 partitions

~~~bash
sbatch scripts/slurm/m41_4_257_scaling_bench.slurm
~~~

By default, both lazy and preload are measured **three times each**.
One Grace allocation contains six separate forward trials, but **every
trial is a fresh Python process**, so CPU peak RSS, CUDA memory peaks,
checkpoint loading and model state are not contaminated by previous
trials. The order alternates: lazy/preload, preload/lazy, lazy/preload.
Fixed random seed 5678 and the same M37 checkpoint. Timings are full
model-forward wall clock, separate from partition-plan loading.
The first trial of each strategy saves the full five-channel output.

## Step 3 — compare partition count in lazy mode

~~~bash
M41_4_SCALING_SERIES=sweep \
    sbatch scripts/slurm/m41_4_257_scaling_bench.slurm
~~~

By default, compare **16, 32, 64 and 128 partitions**, two fresh-process
forwards each (lazy metadata). The 64-partition run reuses the original
preparation. Checkpoint, seed and full physical graph are fixed.
The first trial per layout saves the full output (approximately
324 MiB for 16,974,593 x 5 float32) for a chunked **elementwise**
numerical comparison against the 64-partition lazy reference.

## Output artifacts

Default root:
 /scratch/coop/theret/GraphAttention_runs/m41_4_257_scaling

- layouts/layout_016, layout_032, layout_128:
  partition files, read/halo counts, ownership, preparation timing, symlinks.
- benchmarks/repeated/layout_064:
  six individual forward.json reports and subprocess logs.
- benchmarks/sweep/layout_NNN:
  two individual forward reports per requested partition count.
- scaling_summary.json, scaling_summary.csv, scaling_summary.md:
  medians and variance, plan setup time, process peak RSS, GPU allocated
  and reserved memory, halo amplification, largest partition read/owned,
  **full-output max_abs, RMS, relative L2, and tolerance violations**.

To rebuild the summary independently after all experiments:

~~~bash
cd /scratch/coop/theret/GraphAttention
singularity exec -B /home -B /scratch \
    /softs/local_arm/singularity/images/pyg25.03.sif \
    /scratch/coop/theret/avbpML_env/bin/python3 \
    -m scripts.summarize_m41_4_scalability \
    --root /scratch/coop/theret/GraphAttention_runs/m41_4_257_scaling \
    --require-repeated --require-layouts 16 32 64 128
~~~

The numerical consistency tolerance is atol=rtol=2e-4 by default. If
the comparison fails, summary files are still written, but the job
exits nonzero. This check compares two partitioned implementations,
not the full unpartitioned M37 model at 257^3.

## Interpretation and limitations

This is a **fixed 257^3 graph / partition-count sweep**, not a
node-count strong/weak-scaling experiment. Each run evaluates a single
model forward, not Heun or generative quality. The full-output
comparison tests consistency between partition layouts; previously
validated small-graph M37 parity remains the numerical gate.

CPU memory is Linux process RSS, **not total-node RAM**. GPU memory
uses PyTorch allocated and reserved peaks; filesystem cache, CPU
frequency scaling and Grace-node differences can add variability.
Interpret medians and ranges rather than single-run improvements.
The paired 64-partition comparison shares one Grace allocation to
reduce cross-host effects.

All jobs request grace, one GPU, four CPUs and four hours with **no
explicit --mem directive**. As usual, the user launches sbatch after
pulling the commit; no asynchronous jobs are submitted by ChatGPT.

## Actual partition-preparation outcome — Grace job 464950

After fixing the Python syntax error in the initial preparation attempt,
the rerun on commit 1afe6909 passed 11/11 regression tests and built
all three layouts without touching the immutable base CSR.

| Layout | Max owned | Max owned+halo read | Halo reads across partitions | Repartition time |
|---|---:|---:|---:|---:|
| 16 (4x2x2) | 1,081,665 | 1,165,192 | 1,328,176 | 57.289 s |
| 32 (4x4x2) | 545,025 | 603,396 | 1,864,792 | 55.084 s |
| 128 (8x4x4) | 139,425 | 164,480 | 3,486,976 | 49.905 s |

Each layout conserved exactly 101,451,264 directed local edges
and 303,167,994 exact-two-hop directed edges, including edges
crossing ownership boundaries. These totals are a useful structural
gate but do not establish full-output equivalence. The next stage
is the paired repeated 64-partition lazy/preload benchmark followed
by the partition-count sweep. No new timing or memory results for
these forward benchmarks have yet been reported.
