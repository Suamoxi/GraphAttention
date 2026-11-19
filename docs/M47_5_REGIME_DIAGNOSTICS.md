# M47.5 — Regime dependence and spatial-structure audit (no training)

This experiment evaluates the existing *epoch-10 last.pt* checkpoints from M47.4,
without altering their state, source signatures, model equations, sparse
attention topology, standardizers or the Flow Matching sampler.

## Research design

For each of the three frozen arms (none, mach, mach_re):

1. Run the original M47.2 50-step Heun generator with 4 independent Gaussian
   latents in each of Re54/Ma0.05, Re54/Ma0.2 and Re54/Ma0.3.
2. Use the original M47.1 validation split, exactly 5 DNS snapshots per
   regime. The three Mach cases are **in distribution**: this is not an
   unseen-regime extrapolation or interpolation benchmark.
3. Compute the original M47.2 physical diagnostics: one-point Wasserstein,
   conservative-channel means/spreads, positive density, positive internal
   energy, dimensional velocity spectrum, velocity correlation, and
   solenoidal/dilatational energy fraction. No clipping or corrections.
4. Audit periodicity in three distinct ways:
   - duplicate stored minimum/maximum endpoint mismatch (existing metric),
   - RMS first/last *unique-grid* FFT wrap jump relative to the RMS interior
     nearest-neighbour jump (new metric),
   - count directed one-hop and exact-two-hop graph edges spanning more than
     75% of each physical coordinate extent (new topology audit).
   This DOES NOT train/test any periodic connectivity modification; it
   quantifies existing topology and field defects.
5. For physically conditioned arms, perform a **counterfactual intervention**
   on a fixed Re54/Ma0.2 mesh, nondimensionalization and Gaussian latent.
   The modified condition values come from the audited case-reference YAMLs.
   - mach_low: replace Ma_condition with Re54/Ma0.05 reference
   - mach_high: replace Ma_condition with Re54/Ma0.3 reference
   - mach_re only: replace log_Re_condition with Re94/Ma0.2 reference, holding
     Ma_condition fixed
   Each intervention changes one physical input at a time; all other tensors
   are kept unchanged. Comparisons use the exact same Gaussian keys as the
   native Re54/Ma0.2 generation. Native samples are stored transiently for
   these paired comparisons.

**Interpretation boundary:** Counterfactual samples are *not* independent
physical simulations at the modified regime. In particular, Re94 conditioning
on a 65-cube Re54 mesh is outside the paired training-regime support and
cannot establish generalization across Reynolds number or resolution. All
counterfactual physical units refer to the Re54/Ma0.2 anchor scales.
Native generation, in contrast, uses each regime's actual audited
reference scales and geometry.

The evaluated population is original validation only; no held-out test
snapshots participate. The previous M47.4 comparison files remain unchanged.

## Run on Calypso

Update a clean checkout before submission:

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase origin main
git status --short
unset M47_CONDITION

# First verify the unconditioned native-case evaluation:
sbatch --array=0 scripts/slurm/m47_5_regime_diagnostics.slurm

# If it succeeds, run physically conditioned arms:
sbatch --array=1-2 scripts/slurm/m47_5_regime_diagnostics.slurm
~~~

Array 0=none, 1=mach, 2=mach_re.

The Slurm job executes three relevant CPU-only test suites inside the
production Singularity image before loading the GPU checkpoint:

- tests/unit/test_m47_5_regime_diagnostics.py
- tests/unit/test_m47_4.py
- tests/unit/test_m47_generation.py

Defaults: 50 Heun steps (100 model evaluations per field), 4 samples per
native regime/intervention, shared seed 20261116, 32 k bins, max lag 8,
8192 sampled nodes per generated or DNS field, GPU FP32.

Environment overrides:
- M47_5_STEPS, M47_5_SAMPLES, M47_5_SEED
- M47_5_OUT_ROOT
- M47_5_STORE_FIELDS=YES to additionally retain full .npz native fields
- M47_4_OUT_ROOT, M47_CONDITION_REFS, M47_ARCHIVE_ROOT
- M47_IMAGE, M47_PYTHON, M47_ENV_ROOT, M47_REPO

Keep the defaults when comparing with existing M47.4 evaluation runs.

The paths in the job are **Singularity paths** under /scratch/coop/theret.
Do not pass host-resolved /gpfs-calypso/scratch paths to the container.
M47.4 source provenance uses the exact reviewed host/Singularity alias check.

The output is immutable: the evaluator refuses to overwrite an existing
output folder. To rerun, use a new M47_5_OUT_ROOT instead of removing previous
scientific records.

## Artifacts required for analysis

For each of none, mach, mach_re:

~~~text
/scratch/coop/theret/GraphAttention_runs/m47_5_regime_diagnostics/
  <mode>/epoch10_h50_n4_seed20261116/
    summary.json                         # mandatory: protocol, checkpoint provenance, aggregate results
    case_Re54_Ma0.05.json                 # mandatory: native low-Mach DNS and generation statistics
    case_Re54_Ma0.2.json                  # mandatory: native mid-Mach statistics and baseline
    case_Re54_Ma0.3.json                  # mandatory: native high-Mach statistics
    controlled_conditioning.json         # mandatory: fixed-geometry interventions (not_applicable for none)
    fields/                              # optional, only when M47_5_STORE_FIELDS=YES
~~~

Also upload the corresponding Slurm stdout and stderr, named:

~~~text
/scratch/coop/theret/GraphAttention_runs/m47_5_regime_<jobid>_<task>.out
/scratch/coop/theret/GraphAttention_runs/m47_5_regime_<jobid>_<task>.err
~~~

For the initial scientific review, **do not upload** last.pt, best.pt,
standardizers.pt, raw HDF5 DNS, or full generated .npz. All needed curves
and scalar physical diagnostics are included in the small JSON files.
If a run fails, provide both the .out and .err logs even if no JSON exists.

## Scientific acceptance and next actions

Evaluate three questions separately:

1. Are native distributions at each audited Mach regime thermodynamically
   admissible and close to the five DNS validation references?
2. Does the model use Ma_condition and/or log_Re_condition meaningfully
   when geometry and the Gaussian field remain identical?
3. Are periodic endpoint jumps, FFT-grid wrap jumps, or graph-edge omissions
   consistent with the observed high-k energy and dilatational errors?

Do not equate a conditioning-sensitive response with a physically correct
regime response. A true held-out regime study requires new reference cases
or a train/validation regime split, and disentangling Reynolds number from
mesh resolution requires additional simulation/mesh combinations.

This implementation has not been executed on Calypso; CPU tests and GPU
numerical behavior must be verified from the returned Slurm logs.
