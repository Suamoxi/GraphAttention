#!/bin/bash
#SBATCH --job-name=m37_m38_3d_quality
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
RUN_ROOT=/scratch/coop/theret/GraphAttention_runs

M38_RUN=${M38_RUN:-${RUN_ROOT}/m38_3d_full_dit_flow_1000e_h128_l10_h4_b18}
M37_RUN=${M37_RUN:-${RUN_ROOT}/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16}
CHECKPOINT_KIND=${CHECKPOINT_KIND:-best}

cd "$REPO"

latest_generation () {
    local run_dir="$1"
    local match
    local legacy
    match=$(find "${run_dir}/generations" -maxdepth 1 -type d \
        -name "flow_heun_steps50_seed5678_${CHECKPOINT_KIND}_epoch*" \
        -print 2>/dev/null | sort -V | tail -n 1)
    if [ -n "$match" ]; then
        printf '%s\n' "$match"
        return 0
    fi

    # Compatibility with generations created before epoch-specific output names
    # were introduced. Only accept the legacy directory when its own summary
    # proves that it was generated from the requested checkpoint kind.
    legacy="${run_dir}/generations/flow_heun_steps50_seed5678"
    if [ -s "${legacy}/generated_test.pt" ] && [ -s "${legacy}/summary.json" ]; then
        if singularity exec \
            -B /home \
            -B /scratch \
            "$IMAGE" \
            "$PYTHON" -c "import json, pathlib, sys; p=json.loads(pathlib.Path('${legacy}/summary.json').read_text()); sys.exit(0 if p.get('source_checkpoint') == '${CHECKPOINT_KIND}.pt' else 1)"
        then
            echo "INFO: using legacy completed ${CHECKPOINT_KIND}.pt generation: ${legacy}" >&2
            printf '%s\n' "$legacy"
            return 0
        fi
    fi

    echo "ERROR: no completed ${CHECKPOINT_KIND}.pt generation found under ${run_dir}/generations" >&2
    exit 1
}

M38_GEN=$(latest_generation "$M38_RUN")
M37_GEN=$(latest_generation "$M37_RUN")
M38_GEN_NAME=$(basename "$M38_GEN")
M37_GEN_NAME=$(basename "$M37_GEN")

M38_BENCH_ROOT=${RUN_ROOT}/benchmark_3d_m38_full_dit_flow
M37_BENCH_ROOT=${RUN_ROOT}/benchmark_3d_m37_dinat_dit_flow
M38_BENCH=${M38_BENCH_ROOT}/${M38_GEN_NAME}
M37_BENCH=${M37_BENCH_ROOT}/${M37_GEN_NAME}

echo "============================================================"
echo "M37 / M38 full-volume 3-D generation quality benchmark"
echo "============================================================"
echo "Reference:              held-out test population"
echo "Pairing:                unpaired population comparison"
echo "FFT source mesh:        33^3 endpoint-inclusive nodes"
echo "FFT unique grid:        32^3 periodic nodes (drop max planes)"
echo "Spectra:                3-D radial FFT"
echo "Energy:                 velocity-based E_3D(k), 10-90% bands"
echo "Wasserstein:            nondimensional + train-standardized"
echo "True full-volume ray casts: enabled (RdBu_r, DNS/M37/M38)"
echo "M38 generation:          $M38_GEN_NAME"
echo "M37 generation:          $M37_GEN_NAME"
echo "============================================================"

git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for gen_dir in "$M38_GEN" "$M37_GEN"; do
    for artifact in generated_test.pt summary.json resolved_config.yaml generation_config.yaml dataset_split_manifest.json; do
        test -s "${gen_dir}/${artifact}" || {
            echo "ERROR: missing generation artifact: ${gen_dir}/${artifact}"
            echo "Run the M38 and M37 3-D inference jobs first."
            exit 1
        }
    done
done

echo
echo "===== 3-D benchmark unit gate ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -m pytest \
        tests/unit/test_generation_benchmark.py::test_cartesian_grid_3d_periodic_drop_max_handles_permuted_node_order \
        tests/unit/test_generation_benchmark.py::test_velocity_spatial_statistics_3d_match_periodic_sinusoid \
        tests/unit/test_generation_benchmark.py::test_velocity_energy_spectrum_3d_recovers_periodic_resolved_tke \
        -q

benchmark_generation () {
    local gen_dir="$1"
    local bench_root="$2"
    local bench_dir="$3"
    local label="$4"

    echo
    echo "===== Benchmark ${label} ====="
    singularity exec \
        -B /home \
        -B /scratch \
        "$IMAGE" \
        "$PYTHON" -u -m scripts.benchmark_generation_3d \
            run_dir="$gen_dir" \
            output_root="$bench_root" \
            overwrite=true

    for artifact in \
        summary.json \
        channel_metrics.csv \
        standardized_wasserstein.csv \
        sample_statistics.csv \
        correlation_matrix.csv \
        spectra.csv \
        spectral_bands.csv \
        physical_metrics.csv \
        energy_spectrum_per_sample.csv \
        energy_spectrum_summary.csv \
        energy_spectral_bands.csv \
        nearest_reference.csv \
        spatial_correlation.csv \
        structure_functions.csv
    do
        test -s "${bench_dir}/${artifact}"
    done

    test -s "${bench_dir}/plots/energy_spectrum/energy_spectrum_mean.png"
    test -s "${bench_dir}/plots/energy_spectrum/energy_spectrum_median.png"
    find "${bench_dir}/plots/snapshot_slices/example_000" -maxdepth 1 -name '*.png' -type f | grep -q .
    test -s "${bench_dir}/plots/spatial_correlation/longitudinal_correlation.png"
    test -s "${bench_dir}/plots/structure_functions/longitudinal_s2.png"
}

benchmark_generation "$M38_GEN" "$M38_BENCH_ROOT" "$M38_BENCH" "M38 Full DiT"
benchmark_generation "$M37_GEN" "$M37_BENCH_ROOT" "$M37_BENCH" "M37 NAT-DiT"

echo
echo "===== Matched true-volume DNS / M37 / M38 renderings ====="
singularity exec -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -m pytest tests/unit/test_volume_rendering_3d.py -q

VOLUME_DIR=${RUN_ROOT}/m37_m38_3d_volume_comparison
singularity exec -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -u -m scripts.plot_m37_m38_3d_volumes \
        --m37-generation "$M37_GEN" \
        --m38-generation "$M38_GEN" \
        --output-dir "$VOLUME_DIR" \
        --num-examples 3 \
        --resolution 220 \
        --steps 84 \
        --optical-depth 2.0 \
        --cmap RdBu_r

for name in rho_value rhou_x rhov_y rhow_z rhoE_value; do
    test -s "${VOLUME_DIR}/example_000/${name}.png"
done

echo
echo "===== M38 standardized Wasserstein ====="
cat "${M38_BENCH}/standardized_wasserstein.csv"
echo
echo "===== M37 standardized Wasserstein ====="
cat "${M37_BENCH}/standardized_wasserstein.csv"

echo
echo "===== M38 energy spectrum summary ====="
cat "${M38_BENCH}/energy_spectrum_summary.csv"
echo
echo "===== M37 energy spectrum summary ====="
cat "${M37_BENCH}/energy_spectrum_summary.csv"

echo
echo "============================================================"
echo "3-D QUALITY BENCHMARK COMPLETE"
echo "M38: $M38_BENCH"
echo "M37: $M37_BENCH"
echo "Full-volume DNS/M37/M38 plots: $VOLUME_DIR"
echo "============================================================"
