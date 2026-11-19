# M48.4e A1 — Exact periodic halo growth versus network depth

**Status:** CPU-only code and local unit tests prepared. The 48^3 and 80^3
real-grid geometry matrix is **not qualified** until Calypso receipts pass.
This audit does not run a GPU benchmark or train a network.

## Scientific objective

Measure whether the two-scale DiNAT **exact backward attention closure** becomes
computationally inefficient as the network becomes deeper. Use the unchanged
`build_targeted_two_scale_patch` builder used for M48.4c/e; do not replace it
with artificial periodic patch boundaries or an approximate halo width.

The tested depth `d` applies **separately** to the fine pre-encoder, fine
post-decoder, and coarse static encoder, plus the dynamic coarse encoder when
`mode=dynamic`. Thus static has `3*d` DiT blocks; dynamic has `4*d`.
Cross-scale attention layers are in addition to those counts.

The default audit is:

| Factor | Values |
|---|---|
| Mode | static and dynamic, separate Slurm jobs |
| Fine mesh | 48^3, 80^3; corresponding coarse 24^3, 40^3 |
| Depth per subnetwork | 2, 4, 6, 8 |
| Owned fine cube side | 12, 16, 20, 24 |
| Periodic patch origin | `seam` and `interior` (different parity) |
| Source data | Authenticated **metadata only** for 44 teacher snapshots; no CFD fields |

Per mode, this yields `2 grids × 4 depths × 4 core sizes × 2 origins = 64`
independently checked CPU worker configurations. Two modes give 128 receipts.
The seam origin matches the previously validated M48.4e Phase A GPU geometry:
`(N-3, N//3+1, N-2)`; the interior origin is `(N//4,)*3`.

## Numerical and compute contract

For each owned fine cube, the builder walks the exact alternating one-hop /
exact-two-hop dependencies backward through all fine and coarse stages,
including the optional fine->coarse dynamic feedback. It creates the actual
induced fine/coarse graph and nonzero trilinear cross-attention supports,
using modulo indexing only on the true global periodic lattice.

Per trial, record:

- Fine and coarse nodes read; fine halo nodes = fine-read minus owned;
  fine-read / owned amplification and fine/coarse domain coverage fractions.
- Fine/coarse one-hop and exact-two-hop edge counts; nonzero upsample
  cross-attention edges, graph payload bytes and geometry construction time.
- Idealized full-coverage node and edge-work *proxies* to compare relative
  duplication across depths and patch sizes without claiming measured GPU speed.
- A fail-closed trial status and complete case/depth/mesh/origin identity.

For a core side `p` and fine side `N`, the primary geometry metric is
`N_fine_read / p^3`. This is both the per-patch read amplification and the
idealized full-coverage fine-read amplification if equivalent same-size patches
were to cover the complete domain. It is not a measured GPU FLOP multiplier.
The idealized coarse amplification is
`N_coarse_read * N^3 / (p^3 * (N/2)^3)`.

The `node_work_proxy` weights read nodes by the number of fine/coarse DiT
blocks (2d fine; d coarse static; +d coarse dynamic when enabled). The
`edge_work_proxy` weights each exact induced local/dilated edge set by its
number of alternating DiT blocks and counts one up cross-attention pass,
plus one down pass in dynamic mode. The full-domain reference uses periodic
Cartesian degree 6 for one-hop and 18 for exactly-two-hop, plus `27/8`
nonzero trilinear upsample cross edges per fine node for r=2.
The *idealized full-coverage work ratio* is:

```
  (per-patch work / p^3) / (full-field work / N^3)
```

These are topology/read-work proxies. They do **not** account for
activation checkpointing, framework and kernel overheads, actual physical
sampling/overlap, or the CPU cost of independent CEDAR projections. They cannot
establish that partitioned training is faster/slower than global training; A2
will benchmark full-field execution at matched supervision and model settings.

The CPU-only geometry is topology-dependent; periodic domain lengths vary
between physical teacher cases, and are authenticated per record. The audit
uses one authenticated representative physical length per grid for computing
coordinates. Its node/edge counts depend on topology and origin, not physical
length. This does **not** authorize mixing different physical spacings in the
same production GPU batch.

## Execution and safety

Each geometry configuration runs in a **separate Python subprocess** with an
address-space cap (48 GiB by default), a 300 s timeout and one receipt per
configuration. The Grace Slurm launcher requests 64 GiB memory, 8 CPUs and
**zero GPUs**, with a six-hour job limit. Failed, timed-out or missing worker
receipts cannot silently qualify A1; the top-level status becomes `ERROR`
and the job fails its postcheck. Completed trials are saved atomically,
including when later trials fail. The output folder must not already exist.

Run from a checkout updated to the A1 commit:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
bash -n scripts/slurm/m48_4e_a1_halo_geometry.slurm
sbatch --export=ALL,M48_MODE=static scripts/slurm/m48_4e_a1_halo_geometry.slurm
sbatch --export=ALL,M48_MODE=dynamic scripts/slurm/m48_4e_a1_halo_geometry.slurm
```

Optional bounded smoke test, with one depth, one patch size and one origin,
on both fine grids (2 trials per mode):

```bash
sbatch --export=ALL,M48_MODE=static,M48_A1_DEPTHS=2,M48_A1_CORE_SIDES=12,M48_A1_ORIGINS=seam scripts/slurm/m48_4e_a1_halo_geometry.slurm
sbatch --export=ALL,M48_MODE=dynamic,M48_A1_DEPTHS=2,M48_A1_CORE_SIDES=12,M48_A1_ORIGINS=seam scripts/slurm/m48_4e_a1_halo_geometry.slurm
```

The default sweep controls are `M48_A1_DEPTHS`, `M48_A1_CORE_SIDES`,
`M48_A1_FINE_SIDES`, `M48_A1_ORIGINS`, `M48_A1_WORKER_TIMEOUT`,
`M48_A1_WORKER_VIRTUAL_GIB`, and `M48_A1_OUT_ROOT`.

Each run writes:

```
/scratch/coop/theret/GraphAttention_runs/m48_4e_a1_halo_geometry/
  job_<JOBID>_<MODE>/
    summary.json
    trials.csv
    trials/<MODE>_n<N>_d<DEPTH>_p<CORE>_<ORIGIN>.json
    trials/<MODE>_n<N>_d<DEPTH>_p<CORE>_<ORIGIN>.log
```

Share both `summary.json` and `trials.csv` per mode. The `summary.json`
contains min/max read and work metrics across both origin profiles, not a
premature performance recommendation. The prior M48.4e GPU benchmark at
`d=2,p=12,seam` is a regression anchor: static 9,232 fine / 2,506 coarse
and dynamic 34,751 fine / 5,208 coarse nodes, for both grids.

## Acceptance / next investigation

The structural sweep passes only when all requested cases have exact completed
receipts, proper 44-teacher manifest authentication, and no CUDA/data access.
Use A1 to identify which deeper patch models have extremely large closure
coverage before spending GPU time on them.

Proceed to **A2**, a controlled full-field versus partitioned execution
comparison, followed by **A3**, a targeted GPU batch-capacity sweep on
structurally viable deeper models. **Do not** infer an optimal scientific
patch size or optimizer batch from A1 alone.
