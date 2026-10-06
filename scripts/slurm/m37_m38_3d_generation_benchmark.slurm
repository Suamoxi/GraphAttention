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
GEN_NAME=flow_heun_steps50_seed5678

M38_GEN=${M38_RUN}/generations/${GEN_NAME}
M37_GEN=${M37_RUN}/generations/${GEN_NAME}

M38_BENCH_ROOT=${RUN_ROOT}/benchmark_3d_m38_full_dit_flow
M37_BENCH_ROOT=${RUN_ROOT}/benchmark_3d_m37_dinat_dit_flow
M38_BENCH=${M38_BENCH_ROOT}/${GEN_NAME}
M37_BENCH=${M37_BENCH_ROOT}/${GEN_NAME}

cd "$REPO"

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
echo "Full-volume field plots:disabled"
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

benchmark_generation "$M38_GEN" "$M38_BENCH_ROOT" "$M38_BENCH" "M38 Full DiT"
benchmark_generation "$M37_GEN" "$M37_BENCH_ROOT" "$M37_BENCH" "M37 NAT-DiT"

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
echo "============================================================"
