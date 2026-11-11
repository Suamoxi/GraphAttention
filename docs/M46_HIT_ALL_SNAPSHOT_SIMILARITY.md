# M46 — All-snapshot cross-case HIT similarity audit

## Scope and evidence status

This is a **read-only analysis implementation**, not a model/conditioning change.

- Source inventory: six Re54/Re94 cases from `/archive2/cfd-phlusim/HIT_DNS_COMP` and all ReT060/ReT100 cases found in that archive and in `/scratch/ai/HIT_DNS_pilot/data/`.
- All discovered snapshots enter the empirical dataset. **No stationarity or quality mask** is generated or applied.
- Duplicate `(case_id, numeric snapshot counter)` entries are represented once, preferring the archive. This is a provenance convention, **not evidence of byte-identical arrays**. The discarded path is recorded.
- Physical reference scales come exclusively from explicit M3.3 case YAML files. The tool **does not** infer velocity, density or Mach from `ReT...`, `Mt...`, `Xi...`, grid resolution, or a posteriori statistics.
- `inventory` works with no case YAML. `analyze` fails with an explicit list of missing YAML files rather than inventing scales or silently dropping cases. The M46 pilot has missing source references for some newer cases; their input provenance must be supplied before full analysis can run.
- The result is a *time-mixture empirical distribution*. Treating all snapshots as eligible **does not prove** that their marginal distribution is statistically stationary or that samples are temporally independent.

## Usage

From the repository root, submit the **inventory** as a Slurm job (never run
Python on the login node). The wrapper uses the proven project Singularity
image and explicitly selects `avbpML_env/bin/python3`:

```bash
sbatch scripts/slurm/m46_hit_inventory.slurm
```

The job uses default roots `/archive2/cfd-phlusim/HIT_DNS_COMP` and
`/scratch/ai/HIT_DNS_pilot` and writes to
`/scratch/coop/theret/GraphAttention_runs/m46_hit_similarity`. Override
paths via `M46_ARCHIVE_ROOT`, `M46_PILOT_ROOT`, and `M46_OUTPUT_ROOT`
at submission time using `sbatch --export=ALL,...`.

Inspect `inventory.json`: expected case count is 32 when the six legacy and 26 pilot cases are present. The snapshot count is computed from actual filenames, not hard-coded.

Supply one authoritative case definition per case in the specified reference directory, named `Re54_Ma0.05.yaml`, `ReT060_Mt0.59_Xi0.3.yaml`, etc. Each file must be a valid M3.3 `CaseDefinition`, have a matching `case_id`, use the same reference scheme, and declare inference-available `rho_ref`, `U_ref`, and `L_ref`, with the declared `L_ref` equaling the physical cubic box side. A named `a_ref` is optional and required only for a reference-sound-speed turbulent Mach diagnostic. Do **not** use a posteriori `Mt` labels as inference conditioning metadata.

Submit the **analysis** through Slurm after the inventory succeeds and
the complete per-case reference YAML directory exists:

```bash
sbatch scripts/slurm/m46_hit_analyze.slurm
```

For the memory-lighter, **no-spectra** pass, use:

```bash
sbatch --export=ALL,M46_NO_SPECTRA=YES scripts/slurm/m46_hit_analyze.slurm
```

This places results in `m46_hit_similarity_no_spectra` and copies the
inventory manifest from the original output directory. A subsequent
full-spectra run writes into the original `m46_hit_similarity` directory.
Use `M46_REFERENCE_DIR` to override the default reference directory and
`M46_ANALYSIS_OUTPUT` to override the analysis output; each job retains
per-snapshot caches to support repeat submissions. Both wrappers execute
tests and the analysis *inside* the allocated Slurm job, not on login nodes.
The requested `grace` GPU is a cluster-allocation convention copied from
existing job scripts; no GPU kernels are used in this analysis.

All snapshots receive exact full-grid field moments and turbulent kinetic energy. By default they also receive an FFT spectrum and solenoidal/dilatational energy split. FFTs at `320^3` unique points are memory- and CPU-intensive: use a high-memory CPU node, not the login node. For the lower-memory first execution of all-snapshot field statistics and pairwise distribution comparisons, use `--no-spectra` in a **separate output directory**. Results are resumable per snapshot; changing analysis options or reference-file contents invalidates existing caches explicitly.

## Mathematical definitions

For each case the conservative nodal fields are nondimensionalized using the existing `ConvectiveNondimensionalizer` (M3.3):

\[
\rho^*=\rho/\rho_{ref},\quad (\rho u_i)^*=(\rho u_i)/(\rho_{ref}U_{ref}),\quad (\rho E)^*=(\rho E)/(\rho_{ref}U_{ref}^2).
\]

All statistics and spectra exclude the maximum coordinate plane on each axis of the endpoint-inclusive periodic mesh. The Cartesian coordinate mapping is checked across all node IDs and is independent of exporter ordering. Node means are arithmetic means on this uniform grid; no arbitrary resolution weighting is applied.

Per-snapshot (non-density-weighted) velocity fluctuations are calculated from `u_i = (rho u_i)/rho`:

\[
\overline{\mathbf u^*}=\langle\mathbf u/U_{ref}\rangle_V,
\qquad k^*=\frac12\left\langle|\mathbf u^*-\overline{\mathbf u^*}|^2\right\rangle_V.
\]

The optional turbulent Mach diagnostic uses `sqrt(2*k*)*U_ref/a_ref` **only** if an explicit inference-available `a_ref` scale exists; no ideal-gas sound speed is silently inferred.

Spectra use normalized FFT coefficients on the unique periodic grid. The full-cube, mean-excluded Helmholtz projection is

\[
\widehat{\mathbf u}_d(\mathbf k)=\frac{\mathbf k\, (\mathbf k\cdot\widehat{\mathbf u})}{|\mathbf k|^2},
\qquad \widehat{\mathbf u}_s=\widehat{\mathbf u}-\widehat{\mathbf u}_d.
\]

`dilatational_fraction` is the ratio of full-cube Fourier dilatational energy to total fluctuation energy. The one-dimensional shell-integrated spectrum groups modes at nearest integer `|k|L_box/(2*pi)` up to the *inscribed* Nyquist sphere; its x coordinate is `kL_box = 2*pi*m`, and `E_star` is normalized so `sum(E_star * 2*pi)` equals energy contained in those shells. `spectrum_energy_capture` reports the fraction of full-cube kinetic energy inside those spectral shells. Therefore, the plotted spectrum must not be assumed to integrate to the total TKE at high wavenumber.

Full-grid scalar moments (mean, population standard deviation, skew, excess kurtosis) are computed in bounded HDF5 chunks. For approximate distribution comparison, each snapshot contributes the same number of uniformly selected **Cartesian** interior nodes (default 2048), independent of native node IDs. Per-case samples are pooled over all its snapshots, and each case contributes one distribution to each comparison. Reported Wasserstein-1 is the average absolute difference of 1024 equal-probability quantiles; it is **an estimator**, not exact full-field W1. The normalized variant divides by the pooled standard deviation of the compared cases' full-field distributions. This is distinct from M29's training-standardized Wasserstein definition.

## Outputs and reproducibility

- `inventory.json`: source paths, mesh dimensions, all selected snapshots and all deduplicated paths.
- `per_snapshot/<case>/<snapshot>.npz`: cached exact field moments, TKE, optional spectra, sampled distribution values, reference checksum and source file metadata.
- `case_field_statistics.csv`, `case_field_pdfs.csv`, `case_pairwise_wasserstein.csv`, `case_regime_parameters.csv`.
- `snapshot_spectra.csv`: per-snapshot nondimensional kinetic-energy spectra when FFTs are enabled.
- `analysis_summary.json`: reference convention, weighting and estimator descriptions.

The outputs are for comparing candidate regimes and selecting explicit conditioning. They do not modify M43-B training code or any generator, and they do not establish causal effects of `Xi` when controller settings differ. Forcing families must remain distinct in subsequent interpretation; the current cross-case tables include cross-family pairs but do not claim those cases have identical stochastic forcing.

## Tests and validation status

`tests/unit/test_m46_hit_similarity.py` covers streaming moment algebra, Parseval and Helmholtz limits for analytic sine modes, Cartesian coordinate reordering, one synthetic case with M3.3 references, and archive-over-pilot snapshot deduplication. Synthetic tests validate the implementation; they do not substitute for a Calypso full-volume performance/physics run. In particular, 320^3 CPU FFT peak RSS, archived-vs-pilot numerical identity, real reference completeness and actual AVBP results remain **target-unvalidated**.

Thermodynamic temperature and pressure diagnostics are deferred: the common pilot inventory contains the five conservative fields and no authoritative equation-of-state reconstruction is introduced here.