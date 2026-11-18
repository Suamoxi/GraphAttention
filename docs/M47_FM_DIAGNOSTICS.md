# M47.3 — diagnostic audit before further M47 training

## Scientific question

The frozen two-epoch M47.1 models improve validation Flow Matching velocity
MSE, but M47.2 50-Heun-step samples retain nonphysical internal energy,
excess dilatational energy, weak spatial correlations, and periodic seam jumps.
M47.3 distinguishes failure of the learned Gaussian-to-DNS transport from
normalization or solver inconsistencies. **It does not retrain or select
checkpoints using held-out test snapshots.**

## Method

Inputs: the three **unchanged** `m47_conditioned/<mode>/train_s42/best.pt`
checkpoints and their original train-only standardizers, provenance, and
chronological splits. The M47.2 frozen-contract checker verifies signatures,
reference YAML hashes, and source test-file metadata before diagnostics.

For each selected **validation** snapshot (default: one per regime) construct
a fixed standard Gaussian source `x0` using the same seed and sample ID
across the three conditioning modes. At `t = 0, 0.25, 0.5, 0.75, 1`, evaluate

```
x_t = (1-t) x0 + t x1
v_target = x1 - x0
v_pred = v_theta(x_t, t, condition, graph)
```

Report the standardized-space per-channel MSE, `v_pred=0` baseline MSE,
MSE improvement against zero, predicted/target RMS, means, and cosine
similarity. Aggregate **per physical snapshot**, never pool mesh nodes
across 64³ and 128³. Verify exact input-target FM symmetry,
identical input/target scalers, and float32 affine roundtrips before predictions.

A single **independent** 64³ Gaussian-to-sample trajectory per conditioning
mode (default: Re54/Ma0.2) records the M47.2-equivalent Heun state at
`t={0,0.25,0.5,0.75,1}`. At each time invert the training standardizer and
report thermodynamic admissibility, periodic seam jumps, velocity spectra,
Helmholtz dilatational energy and velocity correlations. For comparison,
one validation DNS snapshot is diagnosed separately; it is *not* the generated
sample's reconstruction target. The trajectory implementation shares the
M47.2 Gaussian source and model velocity functions and has a unit parity test
against `FlowMatchingTask.sample_standardized`. Observations are stored only
at selected integration steps and no full-volume output is saved.

Outputs:
```
/scratch/coop/theret/GraphAttention_runs/m47_fm_diagnostics/
  none/v1_e2_val1_h50_seed20261116/{summary.json,details.json}
  mach/v1_e2_val1_h50_seed20261116/{summary.json,details.json}
  mach_re/v1_e2_val1_h50_seed20261116/{summary.json,details.json}
```

No M47.1 checkpoint, training artifact, or M47.2 generation result is
overwritten. Each output directory must be absent before execution.

## Calypso invocation

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase origin main
git status --short
unset M47_CONDITION
sbatch --array=0-2 scripts/slurm/m47_fm_diagnostics.slurm
```

The Slurm wrapper runs 11 focused unit tests (five new M47.3 tests and six
existing M47.2 tests), then samples one 50-step trajectory per mode and
evaluates six validation snapshots per mode. To skip trajectories during
initial model diagnostics, set `M47_AUDIT_TRAJECTORY=NO`. To cover all five
validation snapshots per case, set `M47_AUDIT_VALIDATION_PER_CASE=5`.

The job array is independent per arm. Each task requires one GH200/CUDA GPU;
the Slurm limit is eight hours. If execution is expensive, run only the
validation-error audit first by setting `M47_AUDIT_TRAJECTORY=NO`.

## Interpretation

1. **FM channel/time audit:** does learning improve upon the zero predictor
   across times, or only in the easiest time regions? Is poor overall transport
   dominated by energy or density?
2. **Time trajectory:** when do spatial correlations and solenoidal energy
   appear? Do errors grow slowly, or is the learned vector field unable to
   alter the initial Gaussian spectral character?
3. **Scaling consistency:** float32 roundtrip tolerance is `1e-3` in
   standardized units. Failing input/target scaler symmetry is a hard error.
4. **Possible subsequent M47.4 experiment:** only after reviewing results,
   continue training using copied M47.1 run directories, not the original ones;
   audit checkpoint resume and preserve the matched three-arm recipe.

### Limitations

Only one trajectory and one validation snapshot per regime are sampled by
default. This is a mechanistic diagnostic, not uncertainty estimation or
physical fidelity certification. Re54/Re94 remain confounded with resolution.
The graph still lacks the documented periodic SOLUTBOUND coupling. The
thermodynamic test is the same necessary ideal-gas proxy used in M47.2.
