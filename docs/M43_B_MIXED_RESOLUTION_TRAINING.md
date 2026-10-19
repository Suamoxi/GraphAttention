# M43 Experiment B — Real six-case mixed-resolution Flow Matching

## Objective

Train ONE unconditional M37 AlternatingDilatedGeometricDiT (Flow Matching)
on all six verified Re54 / Re94 compressible HIT simulations at 65^3 and
129^3. This is a resolution-AND-turbulence-regime generalization baseline,
NOT coarse-to-fine conditioning, super-resolution or SGS closure training.
M42/dnstoles and inference-only M41 streaming remain out of scope.

This implementation follows two real Grace results:

- M43-B1 (2026-10-19, job 471112): 65^3 succeeded with full
  FP32 forward/backward/AdamW; 129^3 first forward CUDA OOM.
- M43-B1b (job 471143): non-reentrant per-DiT-block activation
  checkpointing enabled full 129^3 training (2 steps) with
  53.166 GiB peak allocated / 58.297 GiB reserved on 95 GiB HBM;
  second step 21.27 s; 65^3 checkpointed peak 6.878 GiB.
  Exact 65^3 diagnostic losses matched baseline and small-graph
  tests checked output and gradient parity.

## Verified six-case archive

Archive root: /archivenfs/phlusim/HIT_DNS_COMP

| Physical case | Grid | Snapshots |
|---|---|---:|
| Re54/Ma0.05 | 65^3 | 100 |
| Re94/Ma0.05 | 129^3 | 100 |
| Re54/Ma0.2 | 65^3 | 100 |
| Re94/Ma0.2 | 129^3 | 100 |
| Re54/Ma0.3 | 65^3 | 100 |
| Re94/Ma0.3 | 129^3 | 100 |
| TOTAL | 300 per resolution | 600 |

Each case has a fixed MESH/mesh.mesh.h5 and
one unambiguous SOLUT or RUN/SOLUT directory.
Snapshots are ordered by their numerical timestep suffix
(also safe for non-padded filenames), and ambiguous prefixes
or duplicate counters fail closed. Filename order does not
prove physical temporal independence. Mesh and
three representative snapshot HDF5 schemas are
checked without loading complete arrays. Per-case
mesh indices and all snapshot filenames remain
explicit in a saved immutable run signature.

The case data document identifies case_info.json and
RUN/run.params as sources of provenance. It does
NOT establish the six numerical rho_ref/U_ref/L_ref
values, and **no such values are fabricated**.

## IMPORTANT reference-scale gate

Scientific Flow Matching runs REQUIRE six real M3.3
case YAML files. Default directory on Calypso:

 /scratch/coop/theret/HIT_DNS_COMP_case_references/

Expected exact filenames:

 Re54_Ma0.05.yaml
 Re94_Ma0.05.yaml
 Re54_Ma0.2.yaml
 Re94_Ma0.2.yaml
 Re54_Ma0.3.yaml
 Re94_Ma0.3.yaml

Each file must follow cases/README.md schema:
- case_id: the EXACT relative ID, for example Re54/Ma0.3
- reference_scheme: SAME semantically coherent convention
  across all six cases (e.g. HIT target one-component rms).
- references.rho_ref: physical density with original units,
  explicit numerical value and origin/provenance.
- references.U_ref: defined characteristic velocity and
  documented provenance, not an arbitrary RMS sampled from
  the model training target.
- references.L_ref: original case's characteristic physical
  length, e.g. periodic box length if defined by setup.
- optional T_ref and regime declarations Ma_target,
  Re_lambda_target, etc., with explicit definitions and
  original provenance.
- All three mandatory scales must be positive, have
  units and provenance, and be inference_available=true.

Do NOT copy the HIT_LES_FORCED numeric values to these
different simulations. Do NOT silently infer L_ref from
node count or velocity from Mach-label text. Review
the real source case_info.json and RUN/run.params to
produce authoritative numeric YAML values. The new
loader checks presence, case identity, coherent scheme,
and required scales. Its read-only inspect mode instead
lists missing references and source provenance paths.

### Phase 0 — Archive inspection (safe, NO training)

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m43_b_multires_inspect_smoke.slurm
~~~

The launcher defaults to M43_B_MODE=inspect. It runs small
synthetic regression tests, then inventories the real
six cases, expected reference YAML paths and provenance
files, writing inventory.json to:

 /scratch/coop/theret/GraphAttention_runs/m43_b_multires/
     inspect_JOBID/inventory.json

No GPU model work, scaler fitting, or training is done by
the inspect command. Its Slurm reservation is for
consistency with later smoke steps, not a GPU requirement.

Inspect individual physics metadata on Calypso, e.g.:

~~~bash
cat /archivenfs/phlusim/HIT_DNS_COMP/Re54/Ma0.3/case_info.json
cat /archivenfs/phlusim/HIT_DNS_COMP/Re54/Ma0.3/RUN/run.params
~~~

Use equivalent paths from the inventory if provenance files
are organized differently. If reference values remain
ambiguous, **stop** and obtain the original simulation
setup rather than silently inventing them.

## Temporal splits and leakage protection

For EACH 100-snapshot case, sort by padded snapshot name
and allocate nominal train first 70, val next 15, test
last 15; remove two snapshots on BOTH sides of each
of the two split boundaries:

- train: positions 0..67 = 68 snapshots / case
- excluded near train/validation: positions 68..71
- validation: positions 72..82 = 11 / case
- excluded near validation/test: positions 83..86
- test: positions 87..99 = 13 / case

Across six physical cases, 408 train, 66 validation,
78 test and 48 withheld. All 600 snapshots are
accounted for without overlap. Filenames and split
ranges are recorded in split.json, and the run signature
binds every source snapshot to its mesh and case.

The index-space embargo reduces immediate neighbor
leakage but does NOT by itself establish statistical
independence. The physical snapshot time interval and
large-eddy turnover time must be audited before claiming
independent generalization. A later blocked/randomized
trajectory split may be justified by the actual cadence.

## Training mechanics

- One shared M37 model, hidden dimension 128, 10 layers,
  4 heads, alternating one-hop/exact-two-hop sparse
  geometric attention, torch_sparse FP32, no extra self.
- Checkpointing ON via model.activation_checkpointing.
  Inference and validation automatically bypass it.
- Five conservative channels: rho, rhou, rhov, rhow, rhoE.
- Unconditional straight-path Flow Matching with the
  existing time-conditioned adaLN and time scale 1000.
- M3.3 per-case convective nondimensionalization
  BEFORE statistical normalization.
- M6 sample-balanced shared mean/std fitted ONLY from
  the declared physical TRAINING snapshots (full
  run: 408 samples). Validation and test are never
  sampled during fit. For Flow Matching inputs and
  targets the same standardized state is used.
- Microbatch size exactly ONE physical full 3D graph.
  The six-case sampler shuffles within each case
  deterministically per epoch then round-robins in
  order 65^3,129^3,65^3,129^3,65^3,129^3.
- Accumulate gradients from SIX distinct physical
  samples, one from each case, before each AdamW
  update; divide accumulated gradient by SIX.
  The final shorter accumulation group (if any) is
  normalized by the number of actual samples.
  Neither GPU memory nor loss weighting is
  proportional to mesh node count.
- Stream physical samples through the collator and
  GPU in order (do NOT cache six large host batches
  awaiting the optimizer step). Graph topology derived
  from each fixed mesh is cached by the GraphTaskCollator;
  subsequent physical snapshots reuse the cached CPU
  derived topology in the same training process.
- Shared train/validation/test losses reduce nodes
  within each physical sample first, then average
  by number of physical samples. The runner reports
  global MSE, BY CASE MSE (all six) and BY RESOLUTION
  MSE (65^3, 129^3). Epoch history is JSON+CSV.
- Deterministic validation noise from FlowMatchingTask
  is seeded by sample ID and held constant across
  epochs; test evaluated only at best validation.
- AdamW lr=1e-4, weight decay 0, seed=42.
- best.pt and last.pt are replaced atomically at
  each completed epoch; resume reloads model,
  optimizer and Flow Matching RNG state. Manifest
  SHA includes snapshot/mesh mapping, physical case
  reference file hashes, splits, seed, code git
  revision, numerical preprocessing and model recipe.
  Resume rejects mismatched data, references and code.
- Peak CUDA allocated/reserved, separate synchronized
  forward/backward timings by resolution, optimizer-step
  wall time (including input loading), processed samples/s,
  and per-resolution validation metrics saved to summary.json.
  Timing synchronization is diagnostic and contributes overhead.
- Existing M37 attention math, loss and generation
  algorithm remain unchanged.

### Phase 1 — Six-case one-step smoke test

Only after creating and reviewing ALL SIX reference YAMLs:

~~~bash
sbatch --export=ALL,M43_B_MODE=smoke \
  scripts/slurm/m43_b_multires_inspect_smoke.slurm
~~~

Each smoke run gets a fresh smoke_JOBID output folder.
It fits the M6 scaler on ONE TRAINING sample per
case (6 total, never validation/test), trains ONE
six-case optimizer update with real 65^3/129^3
full-graph checkpointed gradients, then evaluates
one held-out validation/test snapshot per case.
This is a software/numerical/throughput gate, not
a scientific training run: its scaler is deliberately
not the full training-split scaler.
Inspect smoke_JOBID/summary.json, per-case metrics,
GPU peak, and saved model checkpoints. The smoke command also reloads
the last checkpoint through the scientific resume path,
checking restored model and training RNG state against the
pre-reload state; an independent synthetic test checks AdamW
optimizer restoration and signature mismatches.

### Phase 2 — Actual training (explicit opt-in only)

After the smoke gate and reference-scale audit:

~~~bash
sbatch --export=ALL,M43_B_CONFIRM_LONG_TRAIN=YES \
  scripts/slurm/m43_b_multires_train.slurm
~~~

The separate trainer requires:
- M43_B_CONFIRM_LONG_TRAIN=YES
- six authoritative reference YAML files
- clean Git checkout
- previously validated smoke run (human gate,
  not yet an automatic cross-run source check)
- unique train output dir, or explicit resume

Training launcher default: two epochs, one GH200,
four CPUs, 16 hours, no explicit --mem.
The full 408-snapshot epoch has 68 six-sample
optimizer updates, plus 66 validation cases.
Checkpointing is mandatory for 129^3.

Default full run directory:

 /scratch/coop/theret/GraphAttention_runs/m43_b_multires/train_s42/

Outputs:
  run_signature.json
  inventory.json
  split.json
  standardizers.pt
  ref_Re*_Ma*.yaml (frozen copies)
  last.pt
  best.pt
  history.csv
  history.json
  summary.json

To resume more epochs, use the SAME Git revision
and reference YAMLs. For instance:

~~~bash
sbatch --export=ALL,M43_B_CONFIRM_LONG_TRAIN=YES,M43_B_RESUME=1,M43_B_EPOCHS=4 \
  scripts/slurm/m43_b_multires_train.slurm
~~~

On interruption mid-epoch, the last COMPLETE epoch
checkpoint remains authoritative. The interrupted
epoch is replayed from the last checkpoint using
deterministic per-epoch sample order and saved
noise generator state; PyTorch RNG state, in-memory
sparse geometry caches and I/O timing need not be
bitwise identical after resume.

## Controlled scientific interpretation

This is NOT a resolution-only study: 65^3 data
are Re54 and 129^3 data are Re94, with three Mach
regimes at each resolution. Training one model on
them jointly is useful as an architectural test,
but transfer metrics mix resolution and Reynolds
differences. For a controlled scale study, construct
matched filtered resolutions from the SAME DNS
realization in subsequent experiments.

The present cell-edge graph does NOT yet implement
AVBP SOLUTBOUND periodic mapping. The physical
coordinate normalization and exact-two-hop topology
remain the original M37 convention. We do not
claim recovered fully periodic spatial attention.

**Status: source, tests and Grace inspect/smoke/train
launchers prepared. No audited reference values,
real six-case smoke, or long training result has
been provided yet.**
