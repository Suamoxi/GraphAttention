# M44 SR-0 — first matched DNS/coarse numerical pilot

## Status and boundaries

This is a **one-real-DNS-snapshot systems/supervised-loss smoke test**, not a
scientific DNS-to-LES dataset, not an LES solution, and not a validation run.
No new physical coarse-graining convention is defined in GraphAttention. The
existing `dnstoles` study remains authoritative for physical `Q`, `B`, SGS
statistics, and an eventual conservative baseline.

The initial source is one existing HDF5 DNS snapshot and its original native
hexahedral mesh, read through `AVBPHDF5Dataset` with a case-specific M3.3 YAML.
The prototype qualifies a regular Cartesian subvolume of that mesh and uses
**coarse point selection** as a temporary numerical observation operator.

### Concrete default patch

- Start with the existing Re54/Ma0.3 `65^3` DNS snapshot.
- Take a Cartesian `8 x 8 x 8` cell patch (`9^3 = 729` fine nodes).
- Use `s=2`: select coarse nodes at each second fine-grid coordinate
  (`5^3 = 125` coarse nodes).
- Transfer coarse conditions using eight-vertex trilinear shape-function weights
  in sparse CSR form, with exact partition of unity.
- Define the *pilot* baseline `U_base = T U_c` and residual `r = U_DNS - U_base`.
  **In this pilot both SR-0 coarse condition and baseline are identical after
  transferring/standardizing**; the separate 5-channel baseline input is retained
  so the 15-input M37 model interface remains compatible with later scientifically
  derived baseline operators. This duplication is expected in the pilot.
- **No physical LES filtering.** Point selection does not reproduce a
  finite-volume LES restriction. Here `Q_sample T=I` only because we sampled
  coarse nodes from the fine lattice; this does not imply `Q_physical T=I`.
- The fine graph uses the existing M37 canonical hexahedral 12-edge convention
  and exact-two-hop attention builder. Cut edges at patch boundaries are not
  periodic; no wall or periodic boundary physics is inserted.

## Geometry verification

`build_sr0_cartesian_patch` verifies complete bijection from native node IDs to
its inferred Cartesian lattice, positive extent and regular spacing, native
hexahedral coverage of selected cells, eight unique corners, and the exact M37
12-edge topology. Node order may be shuffled, but physical geometry must be
regular Cartesian and hexahedral. Non-Cartesian / malformed meshes fail closed.
This is deliberately NOT the eventual unstructured tetrahedral/hexahedral SR
pipeline.

The interpolation tensor is fixed and reused:

```
T : R^[Nc x 5] -> R^[Nf x 5]
T(1) = 1
U_base = T(U_c)
```

The five conservative variables must preserve the order
`(rho,rhou,rhov,rhow,rhoE)`. Coarse and fine coordinates must use the *same*
case `L_ref` from M3.3. The existing `GraphTaskCollator` and
`FlowMatchingTask(physical_nondimensionalization=True)` apply the established
reference scales. No length or velocity scale is inferred from a grid side.

## Training and inference smoke

The pilot uses the existing `SR0FlowMatching` implementation unchanged:

1. `z = (U_DNS - U_base) / residual_rms` with an uncentered RMS for each channel.
2. Standardized spatial condition `[T((U_c-mean)/std), (U_base-mean)/std]`.
3. Sample independent `t` and Gaussian noise from the existing M11 task.
4. Fine 15-channel input `[z_t, spatial_condition]`, 5-channel velocity target.
5. Existing M37 DiNAT-DiT sparse attention, adaLN-Zero and per-sample MSE.
6. Two AdamW optimizer steps by default, then 4-step Heun *smoke generation*.
7. Save `summary.json`, `pilot_checkpoint.pt`, `pilot_sample.pt` in an
   output folder that cannot already exist. These are diagnostic artifacts.

**One sample is not enough for independent training/validation/test splits.**
The one-snapshot mean/std/RMS and observed losses must never be reported as a
scientific generalization metric; they are used only to establish finite,
reproducible tensor and gradient flow through the pipeline.

There is no hard restriction of predicted residual to the nullspace of the
point-sampling operator. The generated field may therefore differ from the
input at coarse nodes. The report records this mismatch explicitly. It also
does not enforce positive density or pressure.

## Calypso execution (default one-snapshot job)

On the cluster login node:

```bash
cd /scratch/coop/theret/GraphAttention
git switch main
git pull --rebase
sbatch scripts/slurm/m44_sr0_pilot.slurm
```

The job uses existing Grace Singularity/Python paths and checks:

- `tests/unit/test_m44_sr0_cartesian.py`
- `tests/unit/test_m44_sr0_pilot_pair.py`
- `tests/unit/test_m44_sr0.py`
- `tests/unit/test_flow_matching_task.py`

It uses the established archive root
`/archivenfs/phlusim/HIT_DNS_COMP` and audited references in
`/scratch/coop/theret/HIT_DNS_COMP_case_references` by default. The reference
file for `Re54/Ma0.3` must already exist and be valid. The script will not
invent reference values.

Override the paths and diagnostic output if necessary:

```bash
M44_SR0_ROOT=/archivenfs/phlusim/HIT_DNS_COMP \
M44_SR0_CASE=Re54/Ma0.3 \
M44_SR0_REFS=/scratch/coop/theret/HIT_DNS_COMP_case_references \
M44_SR0_OUTPUT=/scratch/coop/theret/GraphAttention_runs/m44_sr0_pilot/my_first_try \
    sbatch scripts/slurm/m44_sr0_pilot.slurm
```

Or invoke the runner directly within an allocated Grace compute job (not the
login node):

```bash
python -m scripts.m44_sr0_pilot \
  --root /archivenfs/phlusim/HIT_DNS_COMP \
  --case Re54/Ma0.3 \
  --reference-dir /scratch/coop/theret/HIT_DNS_COMP_case_references \
  --side 65 --patch-cells 8 --refinement 2 \
  --device cuda --steps 2 --sample-steps 4 \
  --output /scratch/coop/theret/GraphAttention_runs/m44_sr0_pilot/manual_first
```

The runner uses the existing M43 strict snapshot discovery when `--snapshot`
is omitted. Set `--snapshot /path/to/file.h5` to select an exact snapshot.
Outputs include actual source paths, model settings, loss trace, maximum
point-sampling training match error, and generated sample mismatch.

## Acceptance gate for this initial milestone

1. CPU unit tests pass, including permutation and malformed-geometry cases.
2. Source snapshot, mesh, and audited M3.3 references load without fallback.
3. One qualified patch has `Nf=729`, `Nc=125` for the default settings.
4. The original DNS target matches coarse point samples to floating tolerance.
5. Existing M37 `[729,15] -> [729,5]` performs two finite backward/AdamW updates.
6. Four-step Heun returns a finite `[729,5]` diagnostic sample.
7. Git revision, source snapshot, and diagnostic limits are in `summary.json`.

No GPU result is claimed before the user's Slurm outputs are inspected.

## Beyond the pilot

A scientific conditional SR training experiment needs **multiple independent
DNS trajectories/snapshots**, source-aware grouped data splits, train-only
shared scaling (never a single-snapshot fit), and a physically meaningful `Q`
with the associated coarse state and baseline. Those improvements should
reuse the established GraphAttention training framework and the authoritative
physical operators from `dnstoles`.
