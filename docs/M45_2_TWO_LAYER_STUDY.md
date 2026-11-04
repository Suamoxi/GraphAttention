# M45.2: Two-layer static and dynamic Two-Scale DiNAT

## Objective

Repeat the synthetic Flow Matching comparison of job 491555 with **two graph
attention blocks per DiNAT backbone**, instead of one. In each backbone the
first block uses one-hop edges and the second block uses the exact-two-hop
(dilated) edges from the existing M37 implementation. The static/dynamic
communication topology and the Q/B synthetic operators are unchanged.

This remains a synthetic *architecture qualification*, not physical LES/DNS
validation. No change is made to `dnstoles`.

## Network configuration

| Stage | Static | Dynamic | Input/conditioning |
|---|---|---|---|
| Coarse encoder | 2 layers | 2 layers | 5 coarse channels, `log2(r)` |
| Fine encoder | 2 layers | 2 layers | 5 FM-state + 5 baseline channels, `[t,log2(r)]` |
| Fine-to-coarse cross-attention | absent | present | Sparse bipartite, geometric bias |
| Dynamic coarse processor | absent | 2 layers | 16 features, `[t,log2(r)]` |
| Coarse-to-fine cross-attention | present | present | Sparse bipartite, geometric bias |
| Fine velocity decoder | 2 layers | 2 layers | 16 features, `[t,log2(r)]`; 5 output channels |

All DiNAT backbones use 16 hidden features, 4 attention heads, and MLP ratio 4.
The original one-layer pilot remains reproducible; the shared runner's layer
flag defaults remain `1` to preserve its command-line contract. M45.2 passes
explicit `2` for each stage using the four CLI flags:

- `--fine-pre-layers`
- `--fine-post-layers`
- `--coarse-static-layers`
- `--coarse-dynamic-layers`

All four are stored in `summary.json` settings and checkpoint model_config.

## Controlled comparison to job 491555

M45.2 defaults preserve the previous job's synthetic block-average operator,
refinement factor 2, coarse side 4 (64 coarse nodes), fine side 8 (512 fine
nodes), core side 4 (64 owned fine nodes), 16 features, 4 heads, 24 training
and 8 held-out realizations, 2,000 updates, LR 1e-3, seed 2026, projected FM,
8 Heun steps and 4 generated samples. **The backbone depths are the intended
experimental change.** Static and dynamic use the same input realizations and
FM stochastic paths, but they have different parameter counts.

The limited four-sample generation diagnostic is intentionally retained for
baseline comparability. This is insufficient to establish recovered global
stochastic-mode statistics; a larger ensemble and independent training seeds
are still required.

## Numerical qualification

`tests/unit/test_two_scale_hit.py` now checks one- and two-layer static and
dynamic **global versus dependency-complete local prediction parity**. For
both two-layer models it also compares owned-output-loss parameter gradients
against the global graph to detect incomplete partition dependencies.

`tests/unit/test_m45_synthetic.py` checks one- and two-layer owned-node FM
optimizer updates with both network modes. The Grace Slurm job runs these
checks before starting training. If any fails, the experiment aborts.

At the tested very small `4^3` coarse resolution, dependency closure may cover
much of the fine mesh, especially for dynamic coupling. The run is not a
production memory-scaling benchmark.

## Grace execution

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m45_2_two_layer_synthetic.slurm
```

To qualify 2 layers without the longer training, set `M45_2_STEPS=32`:

```bash
sbatch --export=ALL,M45_2_STEPS=32 scripts/slurm/m45_2_two_layer_synthetic.slurm
```

To compare a different refinement factor, use
`--export=ALL,M45_2_REFINEMENT=3`; the default fine-side 12 remains divisible
by owned core side 4. Additional overrides include `M45_2_OPERATOR`,
`M45_2_COARSE_SIDE`, `M45_2_CORE_SIDE`, `M45_2_WIDTH`, `M45_2_LAYERS`,
`M45_2_TRAIN_SAMPLES`, `M45_2_VAL_SAMPLES`, `M45_2_GENERATED_SAMPLES`,
`M45_2_HEUN_STEPS`, `M45_2_STEPS`, and `M45_2_SEED`.

## Exact files and output locations

| Purpose | Path |
|---|---|
| Static/dynamic backbone | `src/graph_attention/models/two_scale_dinat.py` |
| Alternating local/exact-two-hop DiNAT | `src/graph_attention/models/dinat_dit.py` |
| Paired synthetic training runner | `scripts/m45_synthetic_pilot.py` |
| One-/two-layer parity and gradient tests | `tests/unit/test_two_scale_hit.py` |
| One-/two-layer FM optimizer tests | `tests/unit/test_m45_synthetic.py` |
| **Grace submission file** | `scripts/slurm/m45_2_two_layer_synthetic.slurm` |
| Slurm stdout/stderr | `/scratch/coop/theret/GraphAttention/m45_2_two_layer_<JOBID>.out` / `.err` |
| Experiment root | `/scratch/coop/theret/GraphAttention_runs/m45_2_two_layer/job_<JOBID>/` |
| Summary and comparison | `<experiment-root>/summary.json`, `<experiment-root>/metrics.csv` |
| Individual metrics | `<experiment-root>/<case>/<operator>/<mode>/metrics.json` |
| Training losses/gradients | `<experiment-root>/<case>/<operator>/<mode>/train_history.csv` |
| Saved weights | `<experiment-root>/<case>/<operator>/<mode>/checkpoint.pt` |

For evaluation, provide **both Slurm logs**, `summary.json`, `metrics.csv`,
and the four `train_history.csv` files. Checkpoints are needed only for
post-training diagnostic or continued training; they remain in the job folder
and can be retrieved from Drive when available.

## Success criteria and scientific limits

1. All one-/two-layer parity and optimizer tests pass on the real M37 backend.
2. The job completes and records all four model/case combinations.
3. Depth-specific configuration is captured by the summary/checkpoints.
4. Compare held-out normalized FM loss and mode-second-moment diagnostic against
   one-layer job 491555, *without declaring generative success from four draws*.
5. Report read/owned ratios and training time; do not interpret small periodic
   volumes as large-mesh scalability evidence.

A longer two-layer depth does not ensure better generation or provide a
capacity-matched architectural comparison. Those are separate experiments.
