# M48.4e A2 — Full-field versus complete exact-halo tiling

**Status (2026-11-19): implemented; isolated CUDA qualification pending Calypso.**

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
optimizer update. Numerical equivalence is checked **before the first
optimizer update**, by comparing assembled global predictions and all
parameter gradients. Tolerances: prediction allclose rtol=5e-4, atol=5e-4;
gradient allclose rtol=1e-2, atol=1e-4. Absolute and relative L2 errors are
also stored; an unmet tolerance produces `PARITY_FAILED`, never an accepted
speed result.

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
and then runs `warmup=1`, `measure=2` by default. Synchronized wall update
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
