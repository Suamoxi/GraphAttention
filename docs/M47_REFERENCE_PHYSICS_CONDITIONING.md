# M47 — Reference Reynolds/Mach conditioned HIT Flow Matching

## Objective and scope

Use only the six legacy Re54/Re94 × Ma0.05/0.2/0.3 compressible HIT
cases, with their previously reviewed **exploratory** stationary-candidate
snapshots 41–100. Learn one shared M37 DiNAT-DiT Flow Matching model with
optional global reference Mach and reference Reynolds conditions.

Reference Reynolds/Mach are **case-level physics**, not mesh-resolution labels:

\[
Re_{\rm ref} = \frac{\rho_{\rm ref} U_{\rm ref} L_{\rm ref}}{\mu_{\rm ref}}
= \frac{U_{\rm ref} L_{\rm ref}}{\nu_{\rm ref}},\qquad
Ma_{\rm ref} = \frac{U_{\rm ref}}{a_{\rm ref}}.
\]

- `rho_ref` [kg/m³], `U_ref` [m/s] and `L_ref` [m] come **unchanged**
  from each frozen M3.3 case YAML, with its declared reference scheme.
- `mu_ref` [kg/(m·s)], **or** `nu_ref` [m²/s], is the molecular
  viscosity at a consistent prescribed reference state (not a turbulent,
  numerical or instantaneous effective viscosity).
- `a_ref` [m/s] is the reference sound speed consistent with the EOS and
  with the meaning of `U_ref` (one-component vs three-component RMS matters).
- The directory labels `Re54`, `Re94` and `Ma0.05`, etc., are **never**
  used to calculate these conditions or fill missing physical references.
- No value is inferred from DNS sample means, target turbulent gradients,
  or the measured Taylor microscale; input references must be available
  before inference. The original M3.3 YAML files are not overwritten.

**Physical reference status (2026-11-12):** The six legacy cases' `RUN/run.params`
select `AIR`, and the user supplied the **generic AVBP mixture database**,
which is not present as a separate file under each archived case. Its AIR
block specifies Sutherland molecular viscosity, evaluated at the frozen
M3.3 reference temperature of 300 K. The shared file and the six-case
physical-input YAML are versioned in this repository. The sound-speed
input explicitly follows the previously frozen M3.3 convention; it is not
an independently verified AVBP effective-gas acoustic speed. Provenance
hashes anchor the inputs but do not themselves verify their physical
interpretation. Refer to the audited legacy AIR mixture section below.

## Phase 1 — Inspect references, without training

From the login node:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase origin main
sbatch scripts/slurm/m47_reference_conditions.slurm
```

Python (including tests) executes inside the Grace Slurm job. The default
mode `inspect` never writes case reference YAMLs. Inspect:

```text
/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/
  reference_audit.json
  physical_inputs_TEMPLATE_NOT_VALIDATED.yaml
```

The template documents each required source field, but `value: null` and
`TODO` placeholders are not training values. To use an independent audited
physical-input file, copy and fill the template. The schema is:

```yaml
version: m47_physics_conditions_v1
cases:
  Re54/Ma0.05:
    a_ref:
      value: null           # physical reference sound speed [m/s]; must be literal
      units: m/s
      definition: reference_sound_speed_from_original_setup
      provenance: 'TODO: EOS, thermodynamic reference state, and source'
      source_path: 'TODO: e.g. /scratch/coop/theret/HIT_DNS_COMP_case_references/Re54_Ma0.05.yaml'
    mu_ref:
      value: null           # molecular dynamic viscosity [kg/(m*s)]
      units: kg/(m*s)
      definition: molecular_dynamic_viscosity_at_reference_state
      provenance: 'TODO: viscosity law, thermodynamic reference state, and source'
      source_path: 'TODO: e.g. repo:data/avbp_reference/mixture_database.dat'
  # Repeat for the remaining five Re54/Re94 cases.
```

An input may use `nu_ref` in `m^2/s` **instead of** `mu_ref`; never
supply both. An existing authoritative `a_ref`, `mu_ref` or `nu_ref`
within the M3.3 YAML is reused and does not require duplication.

`source_path` may be a file relative to that case's archived directory,
an absolute file path, or a `repo:` path relative to the GraphAttention
checkout. Use the last form for a generic source shared by all cases.
Its checksum is recorded. Neither comments nor a filesystem filename
alone verify numerical interpretation; review the physical data source.

## Phase 2 — Prepare immutable model case definitions

When all six physical conditions are audited:

```bash
sbatch --export=ALL,M47_REF_MODE=prepare,M47_PHYSICAL_INPUT="$PWD/configs/m47_physical_conditions_re54_re94_air_sutherland_300K.yaml" \
  scripts/slurm/m47_reference_conditions.slurm
```

On success, new derived YAMLs and hashes are written under:

```text
/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/
  case_references/Re54_Ma0.05.yaml
  ... (six YAMLs)
  reference_audit.json
```

The script refuses to overwrite an existing `case_references` directory.
All six entries must be complete before it writes any output case files.
Existing `regime` values from the M3.3 YAML are preserved, rather than
relabeling `Re_lambda_target` or `Ma_target` as `Re_ref` or `Ma_ref`.
The reference scheme and numerical values of the original three convective
scales are retained.

Four new dimensionless regime fields are attached to each **new** YAML:

- `Re_ref = rho_ref * U_ref * L_ref / mu_ref` (physical parameter).
- `Ma_ref = U_ref / a_ref` (physical parameter).
- `log_Re_condition = ln(Re_ref)` (fixed invertible numerical encoding).
- `Ma_condition = Ma_ref / 0.1` (fixed invertible numerical encoding).

The final two are the optional model inputs. The fixed `0.1` divisor is
**not a physical Mach reference** or a learned dataset statistic; it only
improves the range of the existing Fourier conditioning features.
The raw physical parameters remain in the case definitions.

## Phase 3 — Controlled Flow Matching smoke tests

M47 reuses the existing M43-B M37 DiNAT-DiT, training and Flow Matching
implementations; no new transformer is introduced. The difference is the
**approved source snapshot subset** and optional graph-level conditions
passed through the existing adaLN pathway.

The pre-existing M46 files are required:

```text
/scratch/coop/theret/GraphAttention_runs/m46_hit_similarity/inventory.json
/scratch/coop/theret/GraphAttention_runs/m46_hit_similarity/qualified_manifest.json
configs/m46_stationary_selection_reviewed_legacy.json
```

Before training, M47 recomputes the selection from the full inventory and
policy, verifies both SHA-256 hashes, verifies the exact six-case, 60-per-case
41–100 windows, matches source snapshot and mesh paths, and verifies all
six generated case-YAML hashes against the reference audit. It rejects
incomplete or altered selections, stale input hashes and missing physics.

The default launcher performs one optimizer update and checkpoint-restore
smoke, **not long training**:

```bash
sbatch --export=ALL,M47_CONDITION=mach_re \
  scripts/slurm/m47_conditioned_train.slurm
```

Other smoke ablations:

```bash
sbatch --export=ALL,M47_CONDITION=none scripts/slurm/m47_conditioned_train.slurm
sbatch --export=ALL,M47_CONDITION=mach scripts/slurm/m47_conditioned_train.slurm
```

`none` still uses the **same six physical reference YAMLs** for M3.3
scaling and the same 360 selected physical snapshots, simply without
feeding physical descriptors to the model. Model conditioning channel
counts (including Flow Matching time) are respectively 1, 2 and 3.

With 60 snapshots per case, the existing chronological 70/15/15 split
and two-snapshot embargo on both sides of the two boundaries produce:

| Per case | Train | Validation | Test | Excluded embargo |
|---|---:|---:|---:|---:|
| 60 approved snapshots | 40 | 5 | 7 | 8 |
| Six cases | 240 | 30 | 42 | 48 |

The M6 shared field standardizer is fitted on **train only**, with equal
sample weighting across regimes, not weighted by node count. The time
conditioning (`1000 * t`) and the existing M37 geometric attention are
unchanged. The M43 runner's original default path remains unconditional
and retains its former dataset behavior when these M47 flags are not used.

Smoke output is unique per job, e.g.:

```text
/scratch/coop/theret/GraphAttention_runs/m47_conditioned/mach_re/smoke_JOBID/
  split.json
  run_signature.json
  qualified_manifest.json
  physical_reference_audit.json
  standardizers.pt
  last.pt
  best.pt
  summary.json
```

## Phase 4 — Actual training, only after smoke pass

Explicit opt-in:

```bash
sbatch --export=ALL,M47_MODE=train,M47_CONDITION=mach_re,M47_EPOCHS=20,M47_CONFIRM_LONG_TRAIN=YES \
  scripts/slurm/m47_conditioned_train.slurm
```

Repeat with `M47_CONDITION=mach` and `none` using identical epochs, seed,
training split, checkpointing, optimizer and sampling conventions.
Each variant has an independent run directory and is resumable only with
its original data/code/audit signature. Check generation results and
ensemble statistics before claiming generalization.

## Scientific limits

1. Re54 and Re94 currently coincide with different native resolutions;
   this dataset cannot separate Reynolds effects from resolution effects.
2. Withholding only snapshots **within** each physical regime measures
   trajectory generalization, **not** generalization to an unseen Reynolds/Mach
   combination. A later leave-one-regime-combination-out test requires a
   separate, explicit training/evaluation design and careful conditioning
   extrapolation/interpolation claims.
3. Re54 and Re94 belong to the same known forcing family, but forced
   turbulence may also depend on the forcing algorithm and injection scale.
4. Reference Reynolds alone does not specify the filter scale or grid;
   mesh geometry remains a separate input. A same-Re, different-resolution
   test eventually requires matched physical cases at multiple meshes.
5. The current M37 graph does not yet map periodic SOLUTBOUND edges.
6. The 60-snapshot intervals are exploratory stationary candidates, not a
   mathematical proof of stationarity or temporal independence.
7. No legacy original M3.3 YAML, DNS snapshot, or checkpoint is modified.

## Audited legacy AIR mixture reference (2026-11-12)

The separate physical-input configuration
`configs/m47_physical_conditions_re54_re94_air_sutherland_300K.yaml`
provides reference parameters for the six Re54/Re94 cases. This does
**not** modify the original M3.3 references.

The six archived `RUN/run.params` files select `mixture_name = AIR`,
`real_gas = no`, and `mixture_database = mixture_database.dat`.
In the uploaded common mixture database the active `AIR` block declares
`transport = simplified`, `species_name = AIR`,
`viscosity_law = sutherland`, `mu_ref = 1.716d-5` at
`T_ref = 273.15d0`, `viscosity_law_coeff = 110.6d0`.
The original species file has an `AIR` species viscosity of
`1.80953d-5` at 300 K, but this is **not** the active AIR mixture's
reference viscosity. Molecular dynamic viscosity evaluated using
Sutherland's formula at the M3.3 `T_ref=300 K` is

```text
mu(300 K) = 1.716e-5 * (300/273.15)^1.5 * (273.15+110.6)/(300+110.6)
          = 1.8459791938944405e-5 kg/(m*s)
```

The published common mixture database's SHA256 is
`7fc60ebdd857e66143653b54da6221e6fd446e11908b949ed9d16d2c13f5fe9f`;
the uploaded common species database has SHA256
`4d41e4bad08f277276681fa73d19e7d55bf311b6e60ecd1190598e2866ac82fd`.
**Source-location correction.** The user-supplied `mixture_database.dat`
is the **shared generic AVBP database**. It is not present separately in
each case's archive directory. A byte-for-byte copy is now retained at
`data/avbp_reference/mixture_database.dat` in GraphAttention. The six M47
viscosity declarations use
`source_path: repo:data/avbp_reference/mixture_database.dat`.
The preparation step hashes this shared file and does not require
`RUN/mixture_database.dat` under any case directory.

The six case-specific `RUN/run.params` files select the AIR mixture;
the user's supplied generic database is used as the corresponding model
definition. This attribution follows the supplied setup and the user's
confirmation that the file is shared; the archived per-case setup contains
no separate mixture-database bytes to compare.

With the frozen M3.3 `rho_ref=1.17 kg/m^3`, case-specific `U_ref`
and box-side `L_ref` values, the resulting `Re_ref` is
`623.68945447` for Re54 and `1640.08914250` for Re94, each at
the three target Mach numbers. These are **box-length Reynolds numbers**
and must not be conflated with the case labels or Taylor Reynolds number.

The corresponding `a_ref=347.2189510957 m/s` intentionally preserves
the M3.3 reference convention
`a_ref=sqrt(1.4*287.05*300)`, so `Ma_ref=(0.05, 0.2, 0.3)`.
The `AIR` molecular weight `0.02885066880 kg/mol` corresponds to
`R_AIR ≈ 288.1896 J/(kg*K)` rather than 287.05. The **exact AVBP
thermodynamic sound speed at 300 K is not established** solely by this
reference convention; computing it independently requires the solver's
effective heat-capacity ratio at 300 K. M47 therefore labels this a
reference-scheme Mach parameter, not a verified a posteriori turbulent
Mach number.

Using the repository-bundled generic AVBP mixture source, run:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase origin main
sbatch --export=ALL,M47_REF_MODE=prepare,M47_PHYSICAL_INPUT="$PWD/configs/m47_physical_conditions_re54_re94_air_sutherland_300K.yaml" \
  scripts/slurm/m47_reference_conditions.slurm
```

After the preparation job passes its reference checks, launch the default one-step smoke test:

```bash
sbatch scripts/slurm/m47_conditioned_train.slurm
```

Do not use the earlier provisional `1.80953e-5`-based YAML or
`Re_ref≈636/1673` values. Both were superseded by the active mixture
Sutherland law.

## Matched three-arm conditioning study

The three-arm `none` / `mach` / `mach_re` Slurm array, signature-checked
MSE comparison, and scientific interpretation limits are documented in
[**M47_ABLATION_CAMPAIGN.md**](M47_ABLATION_CAMPAIGN.md). The initial experiment
measures held-out Flow Matching velocity MSE, **not** generated spectral or
one-point distribution fidelity.
