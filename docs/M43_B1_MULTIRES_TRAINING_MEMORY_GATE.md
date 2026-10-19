# M43-B1 — full-graph mixed-resolution training feasibility

## Scientific scope

Assess whether the UNMODIFIED M37 10-layer local/exact-two-hop
geometric DiNAT-DiT can perform actual single-GPU Flow Matching
forward/backward/AdamW updates on two different large HDF5 meshes.

This is the first systems gate for Experiment B (one unconditional
model trained on multiple resolutions). No actual multi-case training
starts and no M42, SGS or conditional super-resolution code changes.

## Data

HIT_DNS_COMP archive inventory provided by the user:

| Resolution | Cases | Snapshots |
|---|---:|---:|
| 65^3 | 3 (Re54, Ma0.05/0.2/0.3) | 300 |
| 129^3 | 3 (Re94, Ma0.05/0.2/0.3) | 300 |
| 193^3 | 2 (Re070, Ma0.15/0.35) | 242 |
| 257^3 | 7 (Re080, Re090, Re150) | 784 |
| 513^3 | 1 (Re160 Ma0.20) | 21 |
| Total | 16 | 1,647 |

Pilot cases: Re54/Ma0.3 (65^3 nodes) and Re94/Ma0.3
(129^3 nodes). The two cases share Mach number but NOT
Reynolds number, physical box dimensions or realization.
Do not interpret this pilot as a pure resolution-generalization
experiment. The dataset README describes the case generator
but does not itself enumerate completed mesh resolutions.

Data root:
 /archivenfs/phlusim/HIT_DNS_COMP

The gate inspects HDF5 metadata for node/cell count and
the five conservative fields (rho, rhou, rhov, rhow, rhoE).
It chooses the middle sorted snapshot from one nonempty
SOLUT or RUN/SOLUT. If both contain snapshots, fail
closed to avoid ambiguous sampling. No source data modified.

## Measurements and numerical path

- Reuse AVBPHDF5Dataset + GraphTaskCollator: construct
  actual cell-edge one-hop graph and exact-two-hop graph.
- FlowMatchingTask with model time embedding scale 1000.
- M37 hidden_dim=128, num_heads=4, num_layers=10,
  mlp_ratio=4, torch_sparse backend, no extra self loops.
- Batch size one, FP32, two REAL AdamW updates.
  First step allocates optimizer states and CUDA caches;
  second checks reuse. No gradient accumulation necessary
  for the probe (already available for later training).
- Measure separate HDF5 read, CPU topology assembly,
  H2D/scaling, model initialization, forward,
  backward and optimizer stage times.
- Record CPU process peak RSS, GPU PyTorch allocated/
  reserved peaks, HBM capacity/free, model parameters,
  edge counts, host/Slurm ID, Git revision, and finite loss.

**This is a memory/time benchmark, NOT scientific training.**
Raw CFD fields are standardized PER PROBE SNAPSHOT, and
coordinates scaled by the box extent to avoid numerical
amplitude pathologies during the short gradient test.
Neither the loss nor these statistics can be used for M6
training or validation. The actual multi-case experiment
must use case-specific physical nondimensionalization,
train-only shared statistical fitting and grouped splits.
The original M37 model and attention mathematical
operations are unchanged.

M41.6 host-streamed partition execution is deliberately
NOT used: it is inference-only and runs under no_grad().
The real PyTorch autograd path is required here. No
activation checkpointing is enabled; an OOM is meaningful.

## Grace launch

From the GraphAttention checkout:

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m43_b1_mixed_resolution_memory_probe.slurm
~~~

Resources: grace, one GPU, four CPUs, four hours,
no explicit memory cap. Stage 1: clean checkout,
syntax compilation and focused small CPU regression tests.
Stage 2: two full-HDF5 independent subprocess training
probes on the same allocation, 65^3 then 129^3.

Results by default:
 /scratch/coop/theret/GraphAttention_runs/m43_b1_training_memory_gate/
   signature.json
   preflight.json
   progress.json
   summary.json
   runs/Re54_Ma0.3_65/report.json
   runs/Re54_Ma0.3_65/process.log
   runs/Re94_Ma0.3_129/report.json
   runs/Re94_Ma0.3_129/process.log

Reports are incremental. CUDA OOM produces status cuda_oom;
a subprocess killed by CPU OOM or SIGKILL may be marked
process_failed by the surviving orchestrator. Both results
are retained, even if one fails. Overall status all_passed
requires BOTH cases to finish two optimizer steps; otherwise
the job exits nonzero after recording summary.json.
Do not delete failed results; send them for diagnosis.

An exclusive lock prevents two jobs from mixing outputs.
Completed results may be reused only if signature,
selected snapshot and Git revision still match.

## After results

If 129^3 fits, proceed with a separate multi-case
Experiment B on six Re54/Re94 simulations and 600 total
snapshots. Implementation must include a case-specific
physical reference contract, train-only sample-balanced
standardization, trajectory-aware splits, resolution-aware
microbatch budgets and per-resolution loss reporting.

If 129^3 fails, distinguish CPU graph assembly OOM
from H2D transfer, forward, backward or AdamW memory.
Investigate patch-with-halo training and/or activation
checkpointing before attempting differentiable out-of-core
partitioning. Periodic boundary equivalence remains a
known limitation of the current M37 cell-edge graph.

**Status update (2026-10-19): GH200 full-graph feasibility gate executed. The 65³ probe passed, but 129³ hit CUDA OOM during its first forward. Complete measurements follow.**


## M43-B1 Grace measurements — 2026-10-19

Source: user-uploaded summary(20261019-091244).json.
Summary status: not_all_passed.
Both cases executed on calypso-grace03.calypso.cerfacs.fr,
Slurm job 471112, same source revision
6558dc3a1ac686e2957ef5625fa67889f0376589.
GPU: NVIDIA GH200 480GB (CUDA device HBM capacity = 95.0 GiB).
Model: M37 alternating local/exact-two-hop DiNAT-DiT, h128,
10 layers, 4 heads, 3,051,845 trainable parameters,
torch_sparse backend, 5 conservative channels, FP32,
batch size 1, two scheduled AdamW steps.

| Metric | Re54/Ma0.3: 65³ | Re94/Ma0.3: 129³ |
|---|---:|---:|
| Nodes | 274,625 | 2,146,689 |
| Hexahedral cells | 262,144 | 2,097,152 |
| Directed one-hop edges | 1,622,400 | 12,780,288 |
| Directed exact-two-hop edges | 4,791,930 | 38,042,874 |
| Snapshot HDF5 load (s) | 0.37973 | 2.52681 |
| CPU topology and collator construction (s) | 4.86329 | 39.12749 |
| GPU transfer and probe-only scale (s) | 0.21459 | 0.14035 |
| Model initialization (s) | 1.32988 | 0.82343 |
| Process peak CPU RSS (GiB) | 1.88965 | 7.79498 |
| PyTorch GPU allocation after transfer (GiB) | 0.10667 | 0.83826 |
| Peak PyTorch allocated (GiB) | 41.41574 | 89.32964 before OOM |
| Peak PyTorch reserved (GiB) | 43.03125 | 91.93164 before OOM |
| Completed AdamW steps | **2 / 2** | **0 / 2** |
| Status | **ok** | **cuda_oom in forward_1** |

### Completed 65³ optimizer iterations

| Metric | Step 1 | Step 2 |
|---|---:|---:|
| Forward (s) | 0.51918 | 0.14857 |
| Backward (s) | 2.51208 | 2.42901 |
| AdamW step (s) | 0.05046 | 0.00145 |
| Whole training step (s) | 3.08286 | 2.57963 |
| Peak PyTorch GPU allocated (GiB) | 41.39614 | 41.41574 |
| Diagnostic Flow Matching loss | 2.00077 | 1.99521 |

The second step is the warmed observation; its backward
takes ~94% of the 2.58 s step wall time. The first step
includes lazy sparse-kernel and optimizer initialization.
Diagnostic losses use a SINGLE-snapshot z-score and are not
valid for physical comparisons or generalization claims.

### 129³ OOM

The 129³ probe successfully read the real snapshot,
constructed the complete 12.78M one-hop and 38.04M
exact-two-hop edges, transferred tensors to the GPU,
and instantiated M37.

The FIRST forward pass did not finish. CUDA raised OOM while
attempting a further 4.10 GiB allocation. Error text:
95.00 GiB total, 4.68 GiB free, ~90.31 GiB process memory
in use (including non-PyTorch), and ~88.31 GiB currently
allocated by PyTorch with ~1.32 GiB reserved but not
allocated at the attempted allocation.

Thus the reported 89.33 GiB GPU peak is a FAILED-run
measurement, not the actual memory requirement for
completing forward+backward. The unknown complete-step
requirement may be substantially larger.
The source does not provide block-level GPU memory
snapshots, kernel workspace details, per-stage forward
peaks, or an isolated activation/allocator breakdown.
Do not claim a specific kernel is the culprit.

Only a 95 GiB CUDA HBM allocation is visible to PyTorch,
irrespective of the GPU device name including "480GB"
(which does not imply 480 GB of CUDA-allocated HBM).

### Interpretation for Experiment B

- Mixed-resolution conventional FULL-GRAPH training with
  this fixed M37 FP32 configuration is **NOT feasible at
  129³ on one 95 GiB GH200 under the tested execution**.
- 65³ works at batch size 1, with ~41.42 GiB PyTorch
  peak allocated; no long multi-case training was run.
- Gradient accumulation cannot help when the smallest
  microbatch of one physical volume already fails.
- The current experimental torch_sparse backend requires
  FP32 query/key/value. Blind AMP/bfloat16 substitution is
  not an established numerical-preserving workaround.
- M41.6 CPU-streamed halo-partitioned executor is
  @torch.no_grad() and cannot perform the required
  training backward.
- Memory growth, not HDF5 read or graph-building time,
  is the hard block for 129³. The 39.13 s topology
  preparation also warrants future caching once the
  training OOM is resolved.
- These two cases differ in Reynolds number and may
  differ in domain size. They are not a controlled
  identical-flow resolution comparison.

### Recommended next gate (not implemented in this measurement)

1. Instrument peak/live GPU allocation around input
   embedding and each of the ten M37 blocks during
   forward; sample backward via hooks or memory snapshots.
   Preserve M37 equations and tensor dtypes.
2. Add opt-in, non-reentrant activation checkpointing
   for DiT blocks; assert matching uncheckpointed
   forward values AND gradients on small 3D graphs,
   including geometric attention and trainable adaLN.
   Then rerun 65³/129³ in fresh processes with two
   actual optimizer steps, measured stage memory.
   Checkpointing only reduces retained activations:
   it cannot fix a per-layer workspace that alone
   exceeds available GPU memory.
3. If 129³ still cannot complete, consider exact
   10-layer full-receptive-field halo patch training
   with an owned-node loss, and make the cost of
   overlapping halos explicit. For ten alternating
   one-hop/exact-two-hop layers, the upper bound of
   15 one-hop graph traversals is relevant to a
   single patch's dependency closure.
4. If full-volume training is a hard requirement,
   only then evaluate differentiable, layerwise
   out-of-core halo partitioning and distributed
   training. This is substantially more involved
   than M41.6 inference.

**Current recommendation: do not submit a 600-snapshot
multi-resolution training job. First determine whether
activation checkpointing can make an entire 129³
training step fit.**
