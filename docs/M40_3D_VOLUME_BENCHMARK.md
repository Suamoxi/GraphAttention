# M37 / M40 full-volume 3-D HIT benchmark

## What is compared

M37 is the 3-D DiNAT-DiT baseline: one-hop NAT alternating with exact-two-hop DiNAT, both without self edges. M40 introduces the center token **only** in NAT/local layers; exact-two-hop DiNAT remains without self edges. Both use h128/L10/H4, physical batch 16, Flow Matching, 1000 epochs, seed 42, torch_sparse and 50-step Heun generation with sampling seed 5678.

The scientific metrics and FFT definitions are in [SCIENTIFIC_SPEC.md, sections 23 and 24](SCIENTIFIC_SPEC.md). This document describes the executable benchmark and its true 3-D volume rendering.

For the detailed **physics and exact estimators behind R_LL, R_NN, R_uu, S2, S3, S4 and longitudinal flatness**, see [3D_SPATIAL_STATISTICS_PHYSICS.md](3D_SPATIAL_STATISTICS_PHYSICS.md). This includes the implementation's snapshot normalization, separation conventions, 10-90% bands and limitations of inertial-range scaling laws on the 32^3 grid.

## Scientific benchmark

- Source fields are conservative variables rho, rhou, rhov, rhow, rhoE.
- Source snapshots contain 33 x 33 x 33 nodes; the duplicated periodic maximum planes are dropped for FFT and rendering to obtain 32 x 32 x 32 unique voxels. No training geometry is modified.
- 3-D radial Fourier spectra, velocity-based kinetic energy spectrum, train-standardized Wasserstein-1, distribution PDFs, physical sanity checks, spatial correlations, structure functions, and nearest-reference diagnostics are evaluated on unpaired generated and held-out populations.

Relevant code:
- src/graph_attention/evaluation/generation_benchmark_3d.py
- scripts/benchmark_generation_3d.py
- src/graph_attention/evaluation/spectra_3d.py
- src/graph_attention/evaluation/spatial_statistics_3d.py
- configs/benchmark_generation_3d.yaml

## True 3-D volume ray casting

The plotting module at src/graph_attention/evaluation/volume_rendering_3d.py renders **the entire 3-D scalar field**, not just central slices, three intersecting planes, or an isosurface.

A fixed orthographic camera sends rays through the complete 32^3 box. Each ray samples the field by trilinear interpolation and composites scalar-derived colors and opacities back to front. Every interior voxel can contribute.

- Colormap: **RdBu_r**, identical to the existing 2-D and 3-D slice plots.
- Color scale: shared 1st/99th percentile robust limits per CFD variable across **all** compared populations and all selected examples.
- Opacity: same strictly positive transfer function for every model; a parameter controls optical depth. No threshold excludes the interior.
- View: common azimuth -58 degrees, elevation 26 degrees, with box outline, orientation triad, and one colorbar.

Normalized scalar q = clip((value - vmin) / (vmax - vmin), 0, 1).
For ray increment ds, set density(q) = 0.45 + 1.5*abs(q-0.5) and alpha = 1 - exp(-optical_depth*density(q)*ds). Then C_next = alpha*RdBu_r(q) + (1-alpha)*C_previous.

The output is a **static PNG of a true volume-rendered 3-D field**, not an interactive rotatable viewer. This is a qualitative diagnostic; it does not affect the numerical benchmarks.

IMPORTANT: generation and test snapshots are **unpaired**. Side-by-side DNS, M37, and M40 panels with the same example index do not represent ground-truth reconstruction of an individual snapshot. They display samples from different populations using identical visualization settings.

## Files produced

The controlled comparison saves five variables for three examples:

    /scratch/coop/theret/GraphAttention_runs/
      m37_m40_3d_local_self_comparison/
        plots/volume_3d/
          example_000/rho_value.png
          example_000/rhou_x.png
          example_000/rhov_y.png
          example_000/rhow_z.png
          example_000/rhoE_value.png
          example_001/...
          example_002/...
        generation_quality/relative_comparison.csv
        benchmark_m37/<generation_name>/
        benchmark_m40/<generation_name>/

Each direct-comparison PNG has three volume-render panels: **DNS / M37 / M40** with matched colors, limits and opacity. Each per-model benchmark also exports DNS/generated 2-panel renders at its own plots/volume_3d path. Existing central 2-D slices remain separately under plots/snapshot_slices.

## Launch sequence

Once M40 training finishes, run:

    git pull --rebase
    sbatch scripts/slurm/m40_3d_dinat_local_self_flow_generate.slurm

M37 must already have a best-checkpoint generation as well. Then submit:

    sbatch scripts/slurm/m37_m40_3d_local_self_quality_compare.slurm

For controlled GPU efficiency, independently submit:

    sbatch scripts/slurm/m37_m40_3d_local_self_efficiency_compare.slurm

No model retraining is needed to change volume-rendering settings after generated_test.pt has been saved.

## Settings

The benchmark config configs/benchmark_generation_3d.yaml includes:

    plots.volume_3d: true
    plots.field_cmap: RdBu_r
    plots.field_examples: 3
    plots.volume_resolution: 220
    plots.volume_steps: 84
    plots.volume_optical_depth: 2.0

Larger image resolution or more ray steps improve rendering smoothness but increase CPU post-processing time. The baseline and candidate must use identical view and color/opacity transfer functions.


## M37 (DiNAT-DiT) vs M38 (Full DiT): true 3-D field comparison

The exact same volume-rendering module can compare the existing **M37 and M38**
generations directly, without rerunning training, inference, or distribution
metrics. It produces side-by-side **DNS test reference / M37 DiNAT-DiT /
M38 Full DiT** ray-cast scalar volumes with the same `RdBu_r` colormap,
per-variable shared 1%-99% scalar normalization, opacity and camera view.

Run:

    git pull --rebase
    sbatch scripts/slurm/m37_m38_3d_volume_compare.slurm

The CPU-only launcher locates the latest completed best-checkpoint Heun-50
generation for each model, checks required saved artifacts, runs the volume
rendering regression tests, and exports 3 examples x 5 CFD variables under:

    /scratch/coop/theret/GraphAttention_runs/
      m37_m38_3d_volume_comparison/
        example_000/rho_value.png
        example_000/rhou_x.png
        example_000/rhov_y.png
        example_000/rhow_z.png
        example_000/rhoE_value.png
        example_001/...
        example_002/...

The script `scripts/plot_m37_m38_3d_volumes.py` checks identical held-out
reference IDs and reference fields before rendering. If their saved test
populations do not match, it rejects the comparison instead of silently
pairing unrelated sample arrays. A shared displayed example index is still
**not a generated-to-DNS physical pairing**.

This visualization does not establish a controlled architectural ablation
by itself: M37 and M38 use different training microbatch sizes, although
both use the same full-volume HIT task. It is a fair color-scale-controlled
visual comparison of generated field populations.
