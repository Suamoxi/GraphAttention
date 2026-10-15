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


## Completed 64-partition repeated experiment — 2026-10-15

Uploaded scaling_summary.json reports **6/6 completed full forwards**
on the same fixed 16,974,593-node mesh, unchanged checkpoint and
input seed 5678. Layout 4x4x4 was used for both loading policies.

| Metric | Lazy (n=3) | Preloaded (n=3) |
|---|---:|---:|
| Median forward (s) | 109.9363 | 160.5281 |
| Mean forward (s) | 110.9415 | 158.8194 |
| Standard deviation forward (s) | 2.2118 | 14.7195 |
| Forward range (s) | 109.4109–113.4773 | 143.3201–172.6099 |
| Median CPU peak process RSS (GiB) | 19.9460 | 31.6921 |
| Median metadata setup (s) | 0.1488 | 8.0998 |
| GPU peak allocated (GiB) | 2.2552 | 2.2552 |
| GPU peak reserved (GiB) | 7.6797 | 7.6797 |

Relative to preloading, lazy mode reduced median forward time
by **31.52%**, median peak process RSS by **37.06%**, and
metadata setup time by **98.16%**. The runs were alternated
within the same allocated Grace job. Results are consistent with
the earlier single-run comparison but still have only three
repetitions and may depend on host memory locality/caching.

First full saved lazy/preloaded outputs agree at
max_abs=3.09944e-6, RMS=2.94397e-7, relative_L2=2.91160e-7,
zero violations at atol=rtol=2e-4 across 84,872,965 elements.
All three forward outputs within each strategy had distinct
SHA-256 digests: **not bitwise reproducible**. No other repeat
was compared elementwise by the uploaded summary.

The 16/32/64/128 partition-count sweep is the outstanding next
experiment. All requisite partition plans were previously
prepared (job 464950). Run without rebuilding one-hop CSR:

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
M41_4_SCALING_SERIES=sweep \
    sbatch scripts/slurm/m41_4_257_scaling_bench.slurm
~~~

The sweep defaults to **lazy** loading, two fresh-process forwards
per layout, and elementwise first-trial comparisons to the saved
64-partition lazy baseline. Results and scaling curves remain
unknown until that job completes.


## Full 16/32/64/128 partition sweep completed — 2026-10-15

The uploaded scaling summary reports **status=complete and 14
successful forward runs overall**, consisting of six repeated
64-partition lazy/preloaded trials and eight additional lazy
partition-count trials (two per layout). All used the same 257^3
node graph, unchanged M37 checkpoint and deterministic seed 5678.

### Fixed-graph lazy benchmark

| Partitions | Layout | Forward median (s) | Forward min–max (s) | Peak CPU RSS median (GiB) | GPU peak allocated (GiB) | GPU peak reserved (GiB) | Halo read factor |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | 4x2x2 | 108.2767 | 107.5813–108.9720 | 23.1401 | 8.6786 | 31.7637 | 1.078245 |
| **32** | **4x4x2** | **106.2875** | **105.1422–107.4327** | **20.9924** | **4.4139** | **18.8223** | **1.109858** |
| 64 | 4x4x4 | 110.6150 | 108.4702–112.7597 | 19.9520 | 2.2552 | 7.6797 | 1.141713 |
| 128 | 8x4x4 | 119.0989 | 116.1027–122.0952 | 19.4791 | 1.1759 | 4.4434 | 1.205423 |

**Interpretation:** Among these four tested layouts, 32 lazy
had the shortest observed median wall time (106.29 s).
It was approximately 4% faster than 64 lazy; 64 used roughly
half of its GPU-allocated peak memory. 16 consumed more CPU
and GPU memory than 32 despite being slightly slower in this
small sample. At 128 partitions memory is lower but halo
amplification reaches ~1.205 and per-forward latency rises
to ~119 s. This is a fixed-node-count partition sweep, not
a graph-size scaling curve or proof of an asymptotic law.

GPU-allocated peak shrinks strongly when doubling partitions:
8.679 -> 4.414 -> 2.255 -> 1.176 GiB. CPU RSS is relatively
flat near 20 GiB above 32 partitions. For context the two
full global CPU FP32 hidden-state buffers at H=128 require
approximately 16.188 GiB before accounting for graph/topology,
coordinates, temporary tensors and Python overhead. Peak RSS
captures the complete process, not a tensor-by-tensor breakdown.

### Numerical check

Six saved full outputs (first trial of each displayed experiment,
including the reference itself) were compared in bounded chunks,
84,872,965 values each. Maximum cross-configuration absolute
difference was 3.337860107421875e-6; relative L2 errors were
about 2.91e-7. **No element exceeded atol+rtol*abs(reference)
with atol=rtol=2e-4**. Repeated outputs had different SHA-256
digests, so they are not bitwise reproducible. Not all 14
complete outputs were saved for elementwise comparisons.
This is partition-to-partition numerical consistency, not
same-size unpartitioned M37 reference equivalence.

### Implications and next experiments

- Use **32 lazy partitions** as the current lowest-latency
  M41.4 setting for the 257^3 graph.
- Use **64 lazy partitions** when GPU headroom or bounded
  per-partition allocations are higher priorities; the measured
  forward penalty relative to 32 was only ~4.1% in this sweep.
- Repeat the 32-vs-64 comparison more times under one matched
  allocation before declaring a statistically stable optimum.
  The present sweep has just two samples per partition count.
- Profile partition metadata loading, CPU->GPU staging, GPU
  sparse attention/block execution, CPU RAM allocation and
  layerwise barriers separately. This is the next route to
  improving forward throughput without changing model math.
- No further 16/32/64/128 partition preparation is needed.
  All 14 recorded forwards completed and all saved-output
  tolerance comparisons passed. Physical-field quality,
  Heun sampling quality and spectra remain outside scope.
