#!/bin/bash
#SBATCH --job-name=m35_m36_3d_quality
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

M35_RUN=${M35_RUN:-${RUN_ROOT}/m35_3d_full_dit_flow_pilot_10e_h128_l10_h4_b1_acc8}
M36_RUN=${M36_RUN:-${RUN_ROOT}/m36_3d_dinat_dit_flow_pilot_10e_h128_l10_h4_b1_acc8}
GEN_NAME=flow_heun_steps50_seed5678

M35_GEN=${M35_RUN}/generations/${GEN_NAME}
M36_GEN=${M36_RUN}/generations/${GEN_NAME}

M35_BENCH_ROOT=${RUN_ROOT}/benchmark_3d_m35_full_dit_flow
M36_BENCH_ROOT=${RUN_ROOT}/benchmark_3d_m36_dinat_dit_flow
M35_BENCH=${M35_BENCH_ROOT}/${GEN_NAME}
M36_BENCH=${M36_BENCH_ROOT}/${GEN_NAME}

cd "$REPO"

echo "============================================================"
echo "M35 / M36 full-volume 3-D generation quality benchmark"
echo "============================================================"
echo "Reference:              held-out test population"
echo "Pairing:                unpaired population comparison"
echo "FFT source mesh:        33^3 endpoint-inclusive nodes"
echo "FFT unique grid:        32^3 periodic nodes (drop max planes)"
echo "Spectra:                3-D radial FFT"
echo "Energy:                 velocity-based E_3D(k), 10-90% bands"
echo "Wasserstein:            nondimensional + train-standardized"
echo "Full-volume field plots:disabled"
echo "============================================================"

git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for gen_dir in "$M35_GEN" "$M36_GEN"; do
    for artifact in generated_test.pt summary.json resolved_config.yaml generation_config.yaml dataset_split_manifest.json; do
        test -s "${gen_dir}/${artifact}" || {
            echo "ERROR: missing generation artifact: ${gen_dir}/${artifact}"
            echo "Run the M35 and M36 3-D inference jobs first."
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
        nearest_reference.csv
    do
        test -s "${bench_dir}/${artifact}"
    done

    test -s "${bench_dir}/plots/energy_spectrum/energy_spectrum_mean.png"
    test -s "${bench_dir}/plots/energy_spectrum/energy_spectrum_median.png"
}

benchmark_generation "$M35_GEN" "$M35_BENCH_ROOT" "$M35_BENCH" "M35 Full DiT"
benchmark_generation "$M36_GEN" "$M36_BENCH_ROOT" "$M36_BENCH" "M36 NAT-DiT"

echo
echo "===== M35 standardized Wasserstein ====="
cat "${M35_BENCH}/standardized_wasserstein.csv"
echo
echo "===== M36 standardized Wasserstein ====="
cat "${M36_BENCH}/standardized_wasserstein.csv"

echo
echo "===== M35 energy spectrum summary ====="
cat "${M35_BENCH}/energy_spectrum_summary.csv"
echo
echo "===== M36 energy spectrum summary ====="
cat "${M36_BENCH}/energy_spectrum_summary.csv"

echo
echo "============================================================"
echo "3-D QUALITY BENCHMARK COMPLETE"
echo "M35: $M35_BENCH"
echo "M36: $M36_BENCH"
echo "============================================================"
