# Experimental GPU backend - CSR Triton backward

## Goal and status

**Implemented as an opt-in experimental backend; not yet tested on a CUDA GPU.**
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
DiNAT-DiT tests, then performs three separated comparisons:

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

If successful, next benchmark the complete ten-block DiNAT-DiT model on
the original M47 physical graphs, with matched training batches and all
activation checkpointing enabled. M47.4 remains pinned to the original
scientific trainer throughout.

This backend is experimental: no measured speedup or GPU numerical
equivalence is claimed until Calypso output has been inspected.
