# Standalone Physics-Token DiT (Transolver-style global-state experiment)

**Status: code/config/tests and Slurm pilot prepared; Calypso training, GH200
timings, and generated-field quality are NOT yet measured.**

## Objective: isolate learned global communication

Test whether the dynamic learned-physics-state mechanism of **Transolver**
improves global information exchange in HIT Flow Matching, compared with
existing Full DiT (M38) and local/exact-two-hop DiNAT-DiT (M37).

**Do not combine it with DiNAT yet.** This is an independent architecture
and a separately trained checkpoint. It does not change the original
M37, M41.1/2/3, training objectives, existing samplers or M42/dnstoles.

Source: Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General
Geometries*, ICML 2024, arXiv:2402.02366,
<https://arxiv.org/abs/2402.02366>.
Official original irregular-mesh module:
<https://github.com/thuml/Transolver/blob/main/Physics_Attention.py>.

## How the new attention works

For each physical sample with `N` nodes, head dimension `D=H/heads`,
and `G` learned global physics-state tokens (default `G=64`):

1. Each node's assignment projection yields normalized soft weights
   `W[b,h,n,g] = softmax_g(slice_proj(project_x(x)) / tau)`.
   Temperature `tau` is learned per head (initialized at 0.5,
   clamped to [0.1,5] for stable training).
2. Separate projected features `F` are pooled into **dynamic**
   tokens `Z = (W^T F) / (sum_nodes W + 1e-5)`.
3. Token self-attention exchanges information globally among **G**
   tokens per head, via PyTorch SDPA by default.
4. Deslicing: `Y = W attention(Z)` maps updates back onto the
   original `N` nodes. Output projection, DiT residual and MLP retain
   their standard forms.

The tokens are **not fixed learned vectors** (as in a Perceiver
latent-array model); they are data-dependent weighted physical states.
Slicing and deslicing use different weight tensors for different input
fields and model layers. Node-to-token assignments can connect
geometrically distant nodes.

This reproduces the **irregular-mesh Physics-Attention algorithm** in
the original code as an attention module, NOT the complete Transolver
PDE neural operator. We keep the existing DiT's input projection,
absolute normalized coordinate embedding, AdaLN-Zero conditioning,
normalization, MLP, and final output projection. We do **not** copy its
original supervised PDE training recipe or convolution-based structured
2-D/3-D variant.

## Complexity and limitations

| Attention mechanism | Approximate attention interaction count |
|---|---|
| Full DiT | `O(N^2)` |
| M37 DiNAT sparse local/exact-two-hop | `O(N k)` |
| Dynamic Physics-Token attention | `O(N G + G^2)` |

Projections and nodewise MLP add additional costs. For G=64, this is
not guaranteed to outperform sparse DiNAT with 6/18 graph neighbors.

With `token_chunk_size=null` (default), assignments have shape
`[B,heads,N,G]` per layer and their forward activations can be
large at `257^3`.

Positive `token_chunk_size` enables a **two-pass, recomputed** scheme:

- Pass A: accumulate weighted token sums and normalization denominators
  from node chunks (no full N×G assignment tensor).
- Global token attention over G states per head.
- Pass B: recompute each chunk's assignment weights and deslice its
  node outputs.

This is mathematically the same full-graph soft slicing for each
physical sample up to FP reduction order; the unit tests compare
outputs **and gradients** to the unchunked implementation.

**Not yet fully out of core:** the base DiT keeps its complete hidden
states, nodewise projections, and training backward activations on
GPU. Chunking reduces the attention assignment workspace, not the
entire model's activation memory. We do not claim `257^3` fits
on one GPU. GPU memory and throughput are empirical questions.

## Fair experiment protocol

Hold fixed:

- AVBP `HIT_LES_FORCED` 3-D dataset, mesh and existing grouped train /
  validation / test split, using seed 42.
- Five conservative variables `rho`, `rhou`, `rhov`, `rhow`, `rhoE`.
- Original `FlowMatchingTask`, training/validation loss, physical
  non-dimensionalization + training-set normalization, and time
  embedding scale 1000.
- H=128, L=10, four heads, `mlp_ratio=4`, same absolute coordinate
  normalization and time-conditioning AdaLN-Zero body as M37/M38.
- 50-step Heun sampler, 100 model evaluations; same sampling seed for
  comparisons.

The controlled model factory creates a seeded Full DiT reference
and copies all **matching non-attention body** tensors exactly.
Attention-specific physics-token parameters have their own independent
seed `seed+1`. The initialization policy is tracked with the
checkpoint. Parameter counts differ and should be reported; this is a
*matched backbone*, not a parameter-count-matched model.

The first 10-epoch pilot is for integration and performance sanity,
**not a quality comparison against 1000-epoch M37/M38**. For fair
quality comparison train to the same convergence budget / 1000 epochs.

We should compare:

- Train/validation Flow Matching MSE on the exact same split.
- Single-forward and 50-step Heun latency, max allocated GPU memory.
- Generated 3-D physical-field distributions, mean/std, and
  standardized Wasserstein distance.
- Energy spectrum in physical k, emphasizing high-wavenumber content,
  plus optional velocity-gradient/two-point statistics.
- No assumptions that physics-token pooling preserves turbulent
  small-scale details; this is the central empirical hypothesis.

## Files

- `src/graph_attention/models/physics_token_dit.py`:
  `PhysicsTokenMultiheadAttention`,
  `PhysicsTokenDiTGraphTransformer`.
- `configs/model/physics_token_dit.yaml`: separate Hydra target.
- `src/graph_attention/training/model_factory.py`: controlled matched
  DiT-body initialization; new attention-specific weights independent.
- `tests/unit/test_physics_token_dit.py`: exact paper-style formula,
  chunk/dense forward+gradient agreement, global and permutation
  behavior, packed graph separation, seed/initialization contract,
  DiT gradients, 50-step Heun reproducibility, invalid configuration
  rejection and assignment chunk size.
- `scripts/slurm/physics_token_dit_3d_train.slurm`: GH200 pilot
  with focused tests, same real 3D HIT dataset and objective as M37.
- `scripts/slurm/physics_token_dit_3d_generate.slurm`: separate
  checkpoint-based 50-step Heun generation (once trained).

## Start pilot on Calypso

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/physics_token_dit_3d_train.slurm
```

The pilot defaults to:
- `MAX_EPOCHS=10` (pilot only).
- `MICROBATCH=4`, `ACCUM=4` (effective 16 samples per optimizer step).
- `TOKEN_COUNT=64`, `TOKEN_CHUNK_SIZE=null` (original dense slicing).
- Saved experiment directory:
  `/scratch/coop/theret/GraphAttention_runs/physics_token_3d_flow_h128_l10_h4_g64_b4_acc4`.

For a production continuation, after examining pilot reports and memory
usage, set `MAX_EPOCHS=1000` and submit the same Slurm script again
with the **same** training hyperparameters and run name. The existing
generic runner resumes from its last checkpoint.

Example memory-limited ablation (run name must be unique):

```bash
export RUN_NAME=physics_token_3d_flow_h128_l10_h4_g32_chunk4096_b4
export TOKEN_COUNT=32
export TOKEN_CHUNK_SIZE=4096
export MAX_EPOCHS=10
sbatch scripts/slurm/physics_token_dit_3d_train.slurm
```

Changing token count or chunk mode in the same run directory is not
supported. Use separate names for ablations; do not mix checkpoints.

## 50-step Heun after training

```bash
sbatch scripts/slurm/physics_token_dit_3d_generate.slurm
```

The generation script uses the saved `resolved_config.yaml` and the
original generic sampler; it does **not** use M37 checkpoints. The
generated test set can then be compared to M37/M38 with the existing
statistical/spectral tools.

## Follow-up only if successful

If a learned global-state baseline captures the relevant HIT statistics,
consider adding an *optional* Physics-Token attention branch alongside
M37 local/exact-two-hop attention. The hybrid should be a new model,
not an in-place change to existing M37 weights or checkpoints.

For HPC scaling, a distributed implementation could locally accumulate
token numerators/denominators and all-reduce these small global arrays,
but gradients and hidden-state memory still need explicit engineering.
