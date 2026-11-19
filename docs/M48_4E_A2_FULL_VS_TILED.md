# M48.4e A2 — Full-field versus complete exact-halo tiling

**Status (2026-11-19): active-path parity and bounded 80^3 extension implemented; Calypso qualification pending.**

A1 established that deep dynamic TwoScaleDiNAT can require nearly the entire
48^3 periodic fine grid to compute a single owned patch. A2 measures whether
performing one global training update is computationally preferable to applying
independent exact-halo patch computations over the same owned field.

## Scientific and numerical boundary

The measurement uses the **existing** `TwoScaleDiNAT` static/dynamic networks,
exact one-hop / exact-two-hop attention, M45 trilinear cross-scale support,
relative geometry, and M48 targeted patch builder. It does **not** train on
DNS, execute CEDAR, change the frozen 44-teacher statistics, or evaluate
physical generation quality. The 44 teacher manifest, case selection and
frozen-normalization metadata are authenticated before GPU work. Only one
representative physical period is used within each grid family; all full and
tiled strategies for that grid use the same value. The source fields are not
opened.

GPU inputs are synthetic float32 five-component fields, with deterministic
seed and **one shared global FM time** per complete field. A2 intentionally
differs from the production independent-per-patch FM sampling, to isolate
computational equivalence rather than compare distinct stochastic objectives.

## Equivalent mathematical objective

For a full fine grid with N=n^3 nodes and C=5 components, let G be the
full periodic two-scale graph, and T_i be non-overlapping owned patch cores
covering all N indices exactly once. For identical network weights and
identical input fields, compare:

    L_full = mean_{i in G, c} (v_theta(x)_ic - y_ic)^2
    L_tiled = sum_tiles (|T_i| / N) *
              mean_{j in T_i, c} (v_theta^exact-halo(x)_jc - y_jc)^2

Each tiled loss is backpropagated individually, accumulating a **single**
gradient, followed by **one** gradient clipping operation and **one** AdamW
update. Both strategies therefore supervise exactly N fine-grid nodes per
optimizer update. Numerical equivalence is checked **at the first two optimizer steps**
(pre-update predictions and gradients, including the second step *after* one
AdamW update) using an explicitly activated DiT model, not the original
zero-output-head initialization. For every attention block in the fine encoder,
coarse encoder, fine decoder and optional dynamic coarse encoder, the
zero-initialized adaLN modulation gate is seeded to small nonzero weights/biases;
likewise the fine decoder output projection receives a nonzero weight. This is
**benchmark-only** and does not alter production model initialization.
The numerical gate requires a nonzero, spatially varying prediction and
nonzero gradients in the fine/coarse/final attention blocks, cross-scale up,
and (dynamic only) cross-scale down and dynamic coarse attention. Norms are
reconstructed from saved named parameter gradients; zeroed or forged metrics
cannot qualify. Tolerances: prediction allclose rtol=5e-4, atol=5e-4;
gradient allclose rtol=1e-2, atol=1e-4, with independent relative-L2
thresholds 1e-3 (prediction) and 2e-3 (gradient). Any unmet tolerance
produces `PARITY_FAILED`, never an accepted speed result.

The full reference graph uses the unchanged
`build_targeted_two_scale_patch` with all fine indices owned and **zero
closure layers**, which constructs complete fine/coarse supports, regardless
of production model depth. The actual network depth is independently passed
to `TwoScaleDiNAT` unchanged. Complete graph cardinality and each tile's
ownership are verified before any CUDA calls. This avoids the generic
full two-hop builder's higher memory complexity and avoids inventing a new
periodic adjacency convention.

## Timing and memory contract

Each `(mode, grid, depth, core, strategy)` runs in an **isolated CUDA process**.
The reference and tiled processes instantiate identically seeded networks,
the same global random inputs/targets, and the same optimizer. A worker
constructs the complete geometry on CPU, preloads its graph(s) on the GPU,
and then runs `warmup=2`, `measure=2` by default. Synchronized wall update
seconds and CUDA event seconds include forward, backward, global GPU gathers
in the tiled path, gradient clipping and optimizer. They exclude CPU geometry
construction, graph transfers, CEDAR/DNS I/O, and initial random input
loading, whose geometry setup costs are reported **separately**.

Peak allocated and reserved GPU memory cover geometry residency and updates.
A measured configuration exceeding the initially available GPU 85% reserved
budget is disqualified. A per-worker timeout, explicit maximal tile count,
and cumulative host graph-bytes cap prevent uncontrolled deep tiling. A CUDA
OOM or timeout remains an unqualified result (not a speed measurement). An
unexpected worker error and prediction/gradient mismatch are reported
separately and fail the launcher acceptance gate.

For a complete tiled update, the number of required tiles is `(n/p)^3`,
**not** `ceil(n^3/p^3)` with partial coverage. Only positive divisors of n
are permitted. A2 defaults to n=48, p=24, eight non-overlapping owned cubes
with equal-size loss contributions. For n=80, p=20 requires **64** tiles and
must be enabled explicitly with a higher max-tile cap and adequate resources.

### Why the GPU comparison is not the A1 structural work ratio

A1 edge/node ratios are topology-only proxies. They do not account for
nonlinear GPU occupancy, launch overhead, stage caching, optimizer state,
serially accumulated gradients, or peak activation memory. The A2 measured
ratio `t_tiled/t_full` (>1 favors global execution) answers whether the
current **naive complete-tile recomputation** is slower on one GPU under the
same loss coverage. The benchmark does **not** test layerwise partitioning,
shared coarse caching across tiles, alternative architectures, the real
independent projected-noise sampling schedule, or domain-size generalization.

## Launch on Grace

Run short depth-2 qualification first:

    cd /scratch/coop/theret/GraphAttention
    git pull --rebase
    sbatch --export=ALL,M48_MODE=static,M48_A2_DEPTHS=2 \
      scripts/slurm/m48_4e_a2_full_vs_tiled.slurm
    sbatch --export=ALL,M48_MODE=dynamic,M48_A2_DEPTHS=2 \
      scripts/slurm/m48_4e_a2_full_vs_tiled.slurm

Then run the default 48^3 / p24 / depths 2,4,8 comparison:

    sbatch --export=ALL,M48_MODE=static \
      scripts/slurm/m48_4e_a2_full_vs_tiled.slurm
    sbatch --export=ALL,M48_MODE=dynamic \
      scripts/slurm/m48_4e_a2_full_vs_tiled.slurm

Optional 80^3 / p20, initially depth 2 only:

    sbatch --export=ALL,M48_MODE=static,M48_A2_FINE_SIDES=80,\
M48_A2_CORE_SIDES=20,M48_A2_DEPTHS=2,M48_A2_MAX_TILES=64 \
      scripts/slurm/m48_4e_a2_full_vs_tiled.slurm

Other tunables: `M48_A2_WIDTH`, `M48_A2_HEADS`, `M48_A2_MEASURE`,
`M48_A2_WARMUP`, `M48_A2_HOST_GIB`, `M48_A2_GPU_FRACTION`,
`M48_A2_WORKER_TIMEOUT`, `M48_A2_MAX_TILES`, `M48_A2_OUT_ROOT`.

Output directory:

    /scratch/coop/theret/GraphAttention_runs/m48_4e_a2_full_vs_tiled/
      job_<SLURM_JOB_ID>_<MODE>_n48_p24/
        summary.json
        trials.csv
        comparisons.csv
        trials/*.json
        trials/*.log
        verification/*.pt

The verification tensors are synthetic prediction/gradient fingerprints,
not data fields or trained checkpoints. Files are not overwritten, and
no scientific training starts from A2 results.

## Decision gate

A2 is qualified for a configuration only when both strategies complete
within the safety budget, every fine node is supervised exactly once,
and the pre-update predictions and gradients agree. Compare update time,
peak memory, and geometry setup costs. If the full graph fits and is faster
for deep dynamic 48^3, favor global execution for that regime; if it does
not fit at 80^3, keep exact-halo or investigate stage-wise partitioning,
activation checkpointing and shared-context computation. No production
model execution-mode change should occur on the basis of a structural
proxy alone.


## A2.1 (next qualification): nontrivial active-path numerical parity

Earlier 48^3 CUDA receipts with all-zero predicted velocities were a useful
execution benchmark but were not sufficient to validate attention-dependent
output equivalence. The revised A2 now forces the DiT gates and final head
nonzero in a deterministic, synthetic-only initialization and verifies two
consecutive updates (step 0 and step 1 after one AdamW step). Every required
attention/cross-scale gradient group must be nonzero. The model definition,
FM loss, exact halo builder and scientific training code remain unchanged.

Run the 48^3 tests **before** enabling the 80^3 tests:

    cd /scratch/coop/theret/GraphAttention
    git pull --rebase
    sbatch --export=ALL,M48_MODE=static,M48_A2_FINE_SIDES=48,M48_A2_DEPTHS=2,4,8 \
      scripts/slurm/m48_4e_a2_full_vs_tiled.slurm
    sbatch --export=ALL,M48_MODE=dynamic,M48_A2_FINE_SIDES=48,M48_A2_DEPTHS=2,4,8 \
      scripts/slurm/m48_4e_a2_full_vs_tiled.slurm

Requirement: both summaries show PASS and each depth has two parity steps
with nonzero spatial prediction variance and nonzero attention-path gradients.
There is no equivalence claim for long training trajectories; two-step parity
validates the execution kernel and immediate optimizer consistency only.

## A2.2 (after A2.1 PASS): bounded 80^3 capacity and throughput

The fine grid is 80^3 = 512000 owned nodes. With p=20, 64 non-overlapping
cubic owned cores cover it exactly once; the coarse grid is 40^3.
The 80^3 job **requires** `M48_A2_PARITY_GATE` pointing to a summary from
A2.1 with the same commit SHA, architecture, width, heads, teacher provenance,
and all requested depths numerically qualified. The full field and tiled
strategies remain independent CUDA workers with a GPU memory headroom gate.

In the 80^3 tiled path, `auto` geometry residency holds the 64 exact-halo
graphs on CPU and transfers each graph to GPU during its owned-core update,
instead of permanently preloading all 64 graphs in device memory. This is an
honest out-of-core benchmark: **tiled update timing includes each host-to-GPU
graph transfer**. The 48^3 path continues to preload 8 graphs, as in A2.
The CPU aggregate geometry has an incremental 12 GiB default budget; worker
timeouts and CUDA memory exclusions are recorded as unqualified, not assigned
an imaginary throughput or speedup. Caching/prefetching improvements remain a
separate experiment. This path does not read source CFD payloads or use CEDAR.

First compare the depth-2, width-64 models (one static and one dynamic job):

    # Change JOBID to the corresponding PASSED A2.1 job ID for each architecture.
    STATIC_A21=/scratch/coop/theret/GraphAttention_runs/m48_4e_a2_full_vs_tiled/job_<STATIC_JOBID>_static_n48_p24/summary.json
    DYNAMIC_A21=/scratch/coop/theret/GraphAttention_runs/m48_4e_a2_full_vs_tiled/job_<DYNAMIC_JOBID>_dynamic_n48_p24/summary.json

    sbatch --export=ALL,M48_MODE=static,M48_A2_FINE_SIDES=80,M48_A2_CORE_SIDES=20,\
M48_A2_MAX_TILES=64,M48_A2_DEPTHS=2,M48_A2_PARITY_GATE="$STATIC_A21" \
      scripts/slurm/m48_4e_a2_full_vs_tiled.slurm
    sbatch --export=ALL,M48_MODE=dynamic,M48_A2_FINE_SIDES=80,M48_A2_CORE_SIDES=20,\
M48_A2_MAX_TILES=64,M48_A2_DEPTHS=2,M48_A2_PARITY_GATE="$DYNAMIC_A21" \
      scripts/slurm/m48_4e_a2_full_vs_tiled.slurm

Only after inspecting capacity, memory and parity at depth 2 should depth 4,
then depth 8 be attempted with the same gate. For the 80^3 run, all
comparisons may legitimately be UNQUALIFIED if either method exceeds the
bounded memory/time limits; this is a capacity finding, not a passed speed
comparison. Numerical mismatches or unexpected worker errors still fail the
Slurm postflight.

The user's separate reference capacity (at least 129^3 nodes, 10 layers,
width 128 on a GH200) is useful hardware context but cannot predict the
A2 memory footprint without matching the *number and type of subnetworks*,
cross-scale attention, optimizer, precision and activation handling. A2 depth
8 here means 24 static or 32 dynamic DiT blocks across the full model, not
an isolated 8- or 10-layer transformer.

## Scientific training-comparison boundary

A2 only compares matched **computational updates**, not convergence or CFD
statistics. A later real-data full-vs-patch study must match teacher case
splits, frozen normalization, physical conditioning, FM source projection,
optimizer schedule and effective node exposure. It must compare held-out
field statistics and spectra (including errors near patch joins) as well as
training throughput and peak memory, before claiming patch-wise training
is scientifically equivalent or superior.
