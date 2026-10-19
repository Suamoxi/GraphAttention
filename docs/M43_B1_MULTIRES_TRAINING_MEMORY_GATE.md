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

**Status: implemented but GH200 results pending.**
