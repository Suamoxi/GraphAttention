# M48.4c — configuration-C exact-halo sampling and GPU batching benchmark

Status: implementation and regression tests prepared on GitHub;
**target-cluster benchmark unvalidated until Grace job results are provided**.

## Scientific and numerical contract

The 44 independent MS_SR teacher sources from 22 physical cases, the
published q8 -> q4 filter `w1p25_l1`, and the immutable
`configs/m48/teacher_44_w1p25_l1_q4_q8.json` statistical scales
are exactly those of M48.4b. The selection and manifest hashes are
verified before any source is loaded. No new physical nondimensionalization
or per-case residual scaling is introduced, and no validation or
locked-test fields are opened.

Per source:

- `Uc = C Uf` is the published CEDAR restriction;
- `B = P Uc` is full-domain CEDAR lifting;
- `R = Uf - B` has `C R = 0`;
- the coarse input is `(Uc - mu_c) / sigma_c`;
- the fine baseline input is `(B - mu_c) / sigma_c`;
- the target endpoint is `X1 = R / sigma_R` with no residual centering;
- each patch samples **independently** `t` and a full-lattice
  standard normal `eps`, then projects `X0 = (I - P C) eps`;
- `Xt = (1 - t) X0 + t X1`, `v_target = X1 - X0`.

Every CEDAR operation is on the full original periodic lattice in
float64. The fine and coarse network features are float32. CEDAR is
NOT used for attention geometry or applied to a cropped patch.

## Patch support, step by step

1. Choose a periodic owned fine-grid cube with side 12,
   `12^3 = 1728` **loss nodes**, using 3 seeded uniform periodic
   coordinate origins; real global index `(i*nf+j)*nf+k`.
2. Trace backwards through the **post-upsample fine decoder** using
   alternating one-hop and exact-two-hop incoming edges, to produce
   `fine_post`.
3. For every `fine_post` query, find its exact nonzero 3-D trilinear
   interpolation keys on the coarse grid (M45 attention priors,
   not the CEDAR observation). This gives `coarse_up`.
4. For dynamic mode only, close `coarse_up` backwards through
   `coarse_dynamic_layers`. Find all fine keys of the resulting
   fine->coarse cross-attention. Static mode omits this branch.
5. Close required coarse keys through the static coarse encoder,
   and close `fine_post ∪ fine_down` through the fine encoder.
6. Build induced local and exact-two-hop edges on these sorted
   **global-ID-derived read sets**, with periodic wraparound on the
   actual full lattice. Missing incoming edges for halo-only queries
   are allowed ONLY when they cannot affect owned outputs.
7. Remap global indices to patch-local IDs. Slice `Xt`, `B`,
   `Uc` using the appropriate read IDs. Loss is evaluated
   **only** at the original owned fine query IDs.

This is exact dependency closure at the specified finite model depth.
It is *not* a fixed-width 12+2h cuboid; the read sets may be irregular,
especially for the dynamic fine->coarse->fine dependency path.
The builder rejects read sets beyond its explicit fine/coarse caps.

Two or four patch graphs may be packed as a **disjoint graph union**:
all fine/coarse edge lists and cross-scale bipartite edges are offset,
and independent fine/coarse batch indices are supplied to DiNAT
conditioning. Graph-specific FM times remain distinct. No attention
edge may cross between packed samples, even if the physical patches
overlap on the original periodic lattice.

## Configuration C, statistical weighting and optimization

    core_side = 12
    patches_per_teacher_source_per_epoch = 16
    effective_batch = 4
    gpu_microbatch_size in {1, 2, 4}

For one effective optimizer batch of four patches, all from the same
teacher source, each independently samples FM time, projected noise,
and origin. For each patch `b`, compute

    L_b = mean_{owned fine nodes i, 5 channels}
          ||v_theta(Xt_b, t_b, Uc) - (X1_b - X0_b)||^2

Then use

    L_update = (L_1 + L_2 + L_3 + L_4) / 4

With GPU microbatch `m`, pack `m` patch graphs into a single forward
and backward pass, accumulate the scaled gradients from all `4/m`
microbatches, clip **once**, and call AdamW **once**.

The optimizer objective and sample seeds are independent of `m`.
Floating-point reduction/activation scheduling can cause small
numerical differences; this is checked with CPU packed-vs-separate
gradient and update reference tests for both network modes.

One full configuration-C epoch has 44 teacher sources times
16 patches per source = 704 independently sampled patches, in
44 * (16/4) = 176 AdamW updates. The proposed eight-epoch campaign
therefore has 5,632 patch samples and 1,408 optimizer updates per
architecture. Each teacher source receives equal *patch* importance,
regardless of its fine-grid node count. No claim about statistical
independence of spatial or temporal turbulent structures follows.

## Bounded two-grid benchmark (not a training epoch)

The initial Grace benchmark uses the full 44-source **teacher-only
authentication**, but performs optimizer steps for only two
representative sources: one on 48^3 and one on 80^3. They are selected
deterministically from the authenticated teacher manifest. The
`--probe-grids` switch changes the schedule signature, making
these checkpoints **incompatible with the future full-epoch run**.

Each mode and microbatch size is run as a separate SLURM job, for
a fair isolated GPU peak-memory measurement.

    cd /scratch/coop/theret/GraphAttention
    git pull --rebase
    bash -n scripts/slurm/m48_4c_sampling_benchmark.slurm
    sbatch --export=ALL,M48_MODE=static,M48_MICROBATCH_SIZE=2 scripts/slurm/m48_4c_sampling_benchmark.slurm
    sbatch --export=ALL,M48_MODE=dynamic,M48_MICROBATCH_SIZE=2 scripts/slurm/m48_4c_sampling_benchmark.slurm

Then compare `M48_MICROBATCH_SIZE=1` and `4` for each mode.
Do **not** set automatic batch-size backoff: CUDA OOM must fail
visibly, preserving identical scientific and optimizer conditions.

Per run, the launcher performs pinned CEDAR checks, Python syntax
qualification and existing M48 tests plus the new M48.4c tests.
It then records `config.json`, `normalization.json`,
`halo_preflight.json`, `summary.json`, `checkpoint_latest.pt`
inside

    /scratch/coop/theret/GraphAttention_runs/m48_4c_sampling_benchmark/
        job_<JOBID>_<MODE>_micro<M>/

All paths refer to the container-visible `/scratch/...` mount, **not
host-only `/gpfs-calypso/scratch/...`**.

For every optimizer update, log owned samples, unique owned nodes,
fine/coarse read-node sum, global CEDAR defects, CPU projection and
patch-geometry time, GPU forward/backward time, peak allocated and
reserved bytes, and end-to-end owned-node throughput. The
`halo_preflight.json` also counts edges, read nodes and halo ratios
at non-seam and seam-crossing cubes for each grid size, separately
for static and dynamic.

## Resume and failure behavior

An optimizer batch is atomic with respect to checkpoints. On normal
exit or every specified checkpoint interval, save model, AdamW,
CPU/CUDA RNG states, completed optimizer updates and logged history.
On interruption, uncheckpointed optimizer steps are replayed from
their start. Resume rejects changed git revision, normalization,
manifest/source seal, core, effective batch, GPU microbatch size,
model configuration, seed, or benchmark/full-epoch schedule.

The default M48.4b one-patch path remains intact, with
`effective_batch=1` and `gpu_microbatch_size=1`.

Known restrictions:

- This phase uses a **single Grace GPU**. DDP and inter-rank balancing
  are not implemented. Packed graph batching eliminates independent
  per-patch GPU launches but still duplicates halos across patches.
- GPU microsize 4 may or may not improve end-to-end throughput.
  Graph construction and full CEDAR projections are CPU-heavy and
  are repeated for each independently drawn patch.
- One-step timings include warmup and are not steady-state estimates.
  Compare both grid sizes and more warmup/repeats before changing the
  full-training GPU microbatch.
- Exactness tests at small lattices check both static/dynamic
  outputs and parameter gradients against the separate full-graph
  or patch computation; the new 12^3 production-size network still
  needs cluster qualification.
- Generated CFD field quality, physical spectra, 3-D Heun sampling,
  and validation by regime are outside this bounded benchmark.
