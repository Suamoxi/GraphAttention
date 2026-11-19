# Experimental GPU backend - CSR Triton backward

## Goal and status

**Opt-in experimental backend; GH200 numerical and synthetic attention benchmark gates completed, but full-model training remains unmeasured.**
This investigation is separate from the frozen M47.1/M47.4 scientific training
continuations. It does not modify their checkpoints, numerical behavior,
input distributions or training recipes.

The new backend named 'torch_sparse_triton' uses the same original FP32 CSR
SDDMM forward, sparse softmax, and memory-bounded SpMM forward as
'torch_sparse'; only the backward reductions are changed. No BF16, periodic
topology modifications, sampling changes or other model changes are introduced.

Files:

- src/graph_attention/models/triton_sparse_backward.py
- src/graph_attention/models/sparse_transformer.py (opt-in dispatch only)
- src/graph_attention/models/dinat_dit.py (CSR cache reuse)
- tests/unit/test_triton_sparse_backward.py
- scripts/benchmark_triton_sparse_backward.py
- scripts/slurm/sparse_backward_triton_benchmark.slurm
- scripts/benchmark_triton_full_training.py (paired real-HDF5 10-layer probe)
- scripts/slurm/sparse_backward_triton_full_training.slurm

## Mathematical contract

For directed edges source-to-target, the attention CSR has row=target and
column=source. The transposed CSR reverses those roles. The per-head
formulas are, for score gradient D and output gradient G:

- dQ = D K / sqrt(head_dim), target CSR
- dK = D-transpose Q / sqrt(head_dim), transposed CSR
- dV = P-transpose G, transposed CSR
- dP_ij = dot(G_i,V_j), bounded edge/feature tiles

The existing PyTorch softmax backward calculates D and propagates gradients
through the learned geometric-bias network. The optimized backward never
allocates a full E-by-H-by-d message tensor and uses no global atomics.

Each Triton program processes a small number of CSR rows and a finite
edge-feature tile. It loops over the necessary number of edge tiles,
supporting arbitrary variable degree, isolated nodes, asymmetric directed
graphs, self-loops, and degrees larger than one tile. The algorithm makes
no Cartesian or regular-mesh assumption.

Initial tile sizes: 4 rows by 32 edges for row reductions and 64 edges for
per-edge dot products. These are correctness-first defaults, not tuned.

## Calypso invocation

From a clean main checkout:

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase origin main
git status --short
# First validate compilation and parity on the smallest graph
sbatch --array=0 scripts/slurm/sparse_backward_triton_benchmark.slurm

# Only if the first job succeeds, run the other topologies/resolutions
sbatch --array=1-3 scripts/slurm/sparse_backward_triton_benchmark.slurm
~~~

Array indices 0-3 are 65-cube one-hop, 65-cube exact-two-hop, 129-cube
one-hop and 129-cube exact-two-hop. Head count=4, head dimension=32, FP32.

The job checks that Triton/CUDA are available inside the deployed Python
environment, runs tests/unit/test_triton_sparse_backward.py and existing
DiNAT-DiT tests with deterministic CUDA reductions, then performs three
comparisons with unrestricted CUDA timing:

- isolated SDDMM backward,
- isolated SpMM backward,
- full geometric sparse-attention forward and backward.

Each comparison benchmarks 'torch_sparse' against 'torch_sparse_triton'.
Graph construction, CSR conversion and initial JIT compilation are outside
the timed region. Profile traces are separate and opt-in using
CSR_TRITON_PROFILE=YES; CSR_TRITON_REPEATS defaults to 3.

Output directory:

~~~text
/scratch/coop/theret/GraphAttention_runs/sparse_backward_triton_v1/
  65cube_one_hop_<jobid>/results.json
  65cube_two_hop_<jobid>/results.json
  129cube_one_hop_<jobid>/results.json
  129cube_two_hop_<jobid>/results.json
~~~

## Numerical acceptance gates

Unit tests compare original/new SDDMM/SpMM gradients on irregular unsorted
directed graphs including high-degree (49-edge) rows and isolated vertices.
They also check full geometric-attention outputs, input and weight gradients,
edge-displacement gradients, and a two-layer DiNAT model with activation
checkpointing.

Gradient thresholds on small examples start at rtol=3e-4..5e-4 and
atol=3e-5..5e-5. Forward thresholds are tighter. Different summation
orders are not bitwise equivalent. Investigate discrepancies instead of
simply loosening tolerances.

Initial GPU gate: all tests pass, memory does not increase unexpectedly,
and 65/129-cube tests complete without OOM. The required speedup is
a reduction in median backward and full-attention-step time, not only a
faster isolated kernel.

## Measured GH200 pilot (November 2026)

Synchronized unprofiled medians of three repetitions, synthetic Cartesian
graphs, FP32, four heads by 32 dimensions. All attention forwards unchanged.

| Graph | Nodes | Directed edges | Geometric-attention backward, original -> Triton (ms) | Backward speedup | Combined forward/backward speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 65^3 one-hop | 274625 | 1622400 | 18.081 -> 8.751 | 2.07x | 1.60x |
| 65^3 two-hop | 274625 | 4791930 | 49.771 -> 14.530 | 3.43x | 2.19x |
| 129^3 one-hop | 2146689 | 12780288 | 141.908 -> 65.848 | 2.16x | 1.62x |
| 129^3 two-hop | 2146689 | 38042874 | 391.253 -> 112.498 | 3.48x | 2.18x |

The final 129^3 two-hop GH200 run was Slurm job 511292 at revision
38abd607. Its deterministic numerical tests reported 27 passed and 1
skipped (DGL dependency unavailable). Maximum allocated geometric-attention
GPU memory: 16.205 GiB for each backend; maximum reserved: 20.611 GiB.
These results do not establish full-model training speedup or scalability for
all arbitrary-degree unstructured graphs. The JSON protocol intentionally
reports "completed_not_accepted" pending full-model evaluation.

## Next gate: real-HDF5 ten-layer M37 optimizer steps

The next probe reuses the existing M43-B1 real HDF5, GraphTaskCollator, and
Flow Matching code. It pairs the original torch_sparse backend against the
opt-in torch_sparse_triton backward with 10 layers, 128 hidden channels,
4 heads, AdamW, and non-reentrant activation checkpointing. Two fresh CUDA
subprocesses use matched HDF5 inputs, graph construction and RNG seeds;
the first optimizer step in each is a warmup, then three timed steps follow.

M43-B1 uses per-snapshot state standardization and box-normalized coordinates
for feasibility. This is NOT the original M47 physical normalization or
a scientific validation run. No checkpoints or trained model weights are
saved and no frozen M47.4 training scripts are changed.

Start from a clean checkout:

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --ff-only origin main
sbatch scripts/slurm/sparse_backward_triton_full_training.slurm
~~~

Default real case: Re54/Ma0.3 at 65^3, in
/archivenfs/phlusim/HIT_DNS_COMP. Output root:
/scratch/coop/theret/GraphAttention_runs/triton_full_training_v1/job_<jobid>/

Required analysis artifacts:

- /scratch/coop/theret/GraphAttention_runs/csr_full_train_<jobid>.out
- /scratch/coop/theret/GraphAttention_runs/csr_full_train_<jobid>.err
- <output>/preflight.json and <output>/summary.json
- <output>/torch_sparse/report.json and <output>/torch_sparse/process.log
- <output>/torch_sparse_triton/report.json and <output>/torch_sparse_triton/process.log

The Slurm script prints full ARTIFACT_* paths before GPU execution.
Use its output for analysis before attempting a 129^3 real graph.

