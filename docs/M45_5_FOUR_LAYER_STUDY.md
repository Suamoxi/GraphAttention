# M45.5 — Four-layer Two-Scale DiNAT architecture comparison

## Purpose

Increase DiNAT backbone depth from **2 to 4 attention blocks per stage**, as a **separate experiment** from M45.4's controlled data/optimization changes.

Each backbone uses the existing M37 alternating connectivity:

1. One-hop sparse geometric attention.
2. Exact-two-hop sparse geometric attention.
3. One-hop sparse geometric attention.
4. Exact-two-hop sparse geometric attention.

The static architecture has **12 DiNAT blocks** (4 coarse + 4 fine encoder + 4 fine decoder). The dynamic architecture has **16 DiNAT blocks** (add 4 dynamic coarse). Cross-scale attention, hidden width 16, four heads, physical Q/B, and Flow Matching formulation are unchanged.

## Controlled comparison to two-layer job 491707

Defaults match job 491707 in every training setting except network depth:

| Setting | M45.5 |
|---|---|
| Coarse/fine size | `4^3 / 8^3` |
| `r`, `Q/B` | `2`, block average |
| Train/validation samples | `24 / 8` |
| Optimizer updates | `2000` |
| Width/heads | `16 / 4` |
| Training objective | `patch_owned`, 64 owned fine nodes |
| Learning rate | `1e-3` |
| Training seed | `2026` |
| Validation seed | `7002026` (same 8 validation fields as M45.2) |
| Original training-generation diagnostic | 4 samples, 8 Heun steps |
| New fixed-coarse diagnostic (enabled by default) | 128 samples, 25 Heun steps; FM oracle error versus time |

The current `8^3` fine mesh is small enough that the four-layer dependency closure can span the whole fine domain. Do not interpret the read/owned ratio as a large-mesh scalability result.

## Grace execution

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m45_5_four_layer.slurm
```

A short test job can be submitted with:

```bash
sbatch --export=ALL,M45_5_STEPS=32,M45_5_RUN_DIAGNOSTICS=0 \
  scripts/slurm/m45_5_four_layer.slurm
```

For the optional **exact Gaussian Heun positive control** inside this job, set `M45_5_RUN_ORACLE=1`. It is disabled by default because the M45.4 pilot already provides this reference. Other overrides: `M45_5_WIDTH`, `M45_5_DEPTH`, `M45_5_GENERATED_SAMPLES`, `M45_5_VELOCITY_DRAWS`.

## Tests and outputs

The updated tests run depth-1, depth-2 and depth-4 M37 global-versus-complete-halo owned-prediction checks (static and dynamic) and owned-node optimizer steps. The existing depth-2 gradient parity checks remain; the depth-4 test exercises forward parity and gradients through the optimizer step.

| Role | Path |
|---|---|
| Job script | `scripts/slurm/m45_5_four_layer.slurm` |
| Slurm logs | `/scratch/coop/theret/GraphAttention/m45_5_four_layer_<JOBID>.out/.err` |
| Root | `/scratch/coop/theret/GraphAttention_runs/m45_5_four_layer/job_<JOBID>/` |
| Train summary | `<root>/summary.json`, `<root>/metrics.csv` |
| Model files | `<root>/<case>/block_average/<mode>/checkpoint.pt`, `train_history.csv`, `metrics.json` |
| Conditional diagnostics | `<root>/diagnostics/time_summary.csv`, `generation_metrics.csv`, `mode_amplitudes.csv`, `spatial_correlations.csv`, `summary.json` |

**Inference:** Full-graph fine residual Heun generation, as in M45.2/M45.3; production partitioned inference is not implemented.

The four-layer run is **not parameter matched** to the two-layer baseline, and has only one training seed. A gain in FM loss or modal generation should be interpreted as a depth/capacity/optimization result, not a proof that exact-two-hop attention alone caused the gain.
