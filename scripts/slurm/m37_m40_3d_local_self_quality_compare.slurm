#!/bin/bash
#SBATCH --job-name=m37_m40_3d_quality
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

M37_RUN=${M37_RUN:-${RUN_ROOT}/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16}
M40_RUN=${M40_RUN:-${RUN_ROOT}/m40_3d_dinat_local_self_flow_1000e_h128_l10_h4_b16}
CHECKPOINT_KIND=${CHECKPOINT_KIND:-best}
OUTPUT_ROOT=${RUN_ROOT}/m37_m40_3d_local_self_comparison
M37_BENCH_ROOT=${OUTPUT_ROOT}/benchmark_m37
M40_BENCH_ROOT=${OUTPUT_ROOT}/benchmark_m40

cd "$REPO"

latest_generation () {
    local run_dir="$1"
    local match
    match=$(find "${run_dir}/generations" -maxdepth 1 -type d \
        -name "flow_heun_steps50_seed5678_${CHECKPOINT_KIND}_epoch*" \
        -print 2>/dev/null | sort -V | tail -n 1)
    if [ -z "$match" ]; then
        echo "ERROR: no ${CHECKPOINT_KIND}.pt generation in ${run_dir}/generations" >&2
        exit 1
    fi
    printf '%s\n' "$match"
}

M37_GEN=$(latest_generation "$M37_RUN")
M40_GEN=$(latest_generation "$M40_RUN")
M37_BENCH=${M37_BENCH_ROOT}/$(basename "$M37_GEN")
M40_BENCH=${M40_BENCH_ROOT}/$(basename "$M40_GEN")

echo "============================================================"
echo "M37 versus M40 - 3D local-self-only DiNAT ablation"
echo "============================================================"
echo "M37: one-hop / exact-two-hop (no self)"
echo "M40: self+one-hop / exact-two-hop (no dilated self)"
echo "Both: full-volume Flow Matching, 50-step Heun, seed 5678"
echo "Both: periodic 33^3 stored mesh -> 32^3 FFT"
echo "M37 generation: $M37_GEN"
echo "M40 generation: $M40_GEN"
echo "Outputs: $OUTPUT_ROOT"
echo "============================================================"

git rev-parse HEAD
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for gen_dir in "$M37_GEN" "$M40_GEN"; do
    for artifact in generated_test.pt summary.json resolved_config.yaml generation_config.yaml dataset_split_manifest.json; do
        test -s "${gen_dir}/${artifact}" || {
            echo "ERROR: missing generation artifact: ${gen_dir}/${artifact}" >&2
            exit 1
        }
    done
done

singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -m pytest \
        tests/unit/test_generation_benchmark.py::test_cartesian_grid_3d_periodic_drop_max_handles_permuted_node_order \
        tests/unit/test_generation_benchmark.py::test_velocity_spatial_statistics_3d_match_periodic_sinusoid \
        tests/unit/test_generation_benchmark.py::test_velocity_correlations_match_structure_functions_3d \
        tests/unit/test_generation_benchmark.py::test_velocity_energy_spectrum_3d_recovers_periodic_resolved_tke \
        tests/unit/test_relative_benchmark.py::test_relative_comparison_3d_checks_fft_mesh_compatibility \
        tests/unit/test_volume_rendering_3d.py \
        -q

for spec in "M37:$M37_GEN:$M37_BENCH_ROOT:$M37_BENCH" "M40:$M40_GEN:$M40_BENCH_ROOT:$M40_BENCH"; do
    IFS=: read -r label gen_dir bench_root bench_dir <<< "$spec"
    echo "===== Rebenchmark $label from completed generation ====="
    singularity exec \
        -B /home \
        -B /scratch \
        "$IMAGE" \
        "$PYTHON" -u -m scripts.benchmark_generation_3d \
            run_dir="$gen_dir" \
            output_root="$bench_root" \
            overwrite=true
    for artifact in \
        summary.json channel_metrics.csv standardized_wasserstein.csv \
        sample_statistics.csv correlation_matrix.csv spectra.csv spectral_bands.csv \
        physical_metrics.csv energy_spectrum_summary.csv energy_spectral_bands.csv \
        nearest_reference.csv spatial_correlation.csv structure_functions.csv
    do
        test -s "${bench_dir}/${artifact}" || {
            echo "ERROR: missing $label benchmark artifact $artifact" >&2
            exit 1
        }
    done
done

echo "===== Direct dimensionless M37/M40 comparison ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.compare_generation_benchmarks \
        baseline_dir="$M37_BENCH" \
        candidate_dir="$M40_BENCH" \
        output_root="$OUTPUT_ROOT" \
        comparison_name=generation_quality \
        overwrite=true

test -s "${OUTPUT_ROOT}/generation_quality/relative_comparison.csv"

echo "===== Full 3-D volume raycasting: DNS / M37 / M40 ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.plot_m37_m40_3d_volumes \
        --m37-generation "$M37_GEN" \
        --m40-generation "$M40_GEN" \
        --output-dir "${OUTPUT_ROOT}/plots/volume_3d" \
        --num-examples 3 \
        --resolution 220 \
        --steps 84 \
        --optical-depth 2.0 \
        --cmap RdBu_r

for variable in rho_value rhou_x rhov_y rhow_z rhoE_value; do
    test -s "${OUTPUT_ROOT}/plots/volume_3d/example_000/${variable}.png"
done

echo "===== M37 standardized Wasserstein ====="
cat "${M37_BENCH}/standardized_wasserstein.csv"
echo "===== M40 standardized Wasserstein ====="
cat "${M40_BENCH}/standardized_wasserstein.csv"
echo "===== M37/M40 energy spectrum bands ====="
cat "${M37_BENCH}/energy_spectral_bands.csv"
cat "${M40_BENCH}/energy_spectral_bands.csv"
echo "===== Relative comparison ====="
cat "${OUTPUT_ROOT}/generation_quality/relative_comparison.csv"

echo
echo "M37 / M40 3D QUALITY COMPARISON COMPLETE: $OUTPUT_ROOT"
