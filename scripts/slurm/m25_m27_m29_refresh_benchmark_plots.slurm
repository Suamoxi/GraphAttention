#!/bin/bash
#SBATCH --job-name=refresh_bench_plots
#SBATCH --time=00:45:00
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

cd "$REPO"

echo "============================================================"
echo "Refresh benchmark plots with current plotting directives"
echo "Targets: M25 EDM / M27 DDPM / M29 Flow t-scale=1000"
echo "============================================================"
git rev-parse HEAD
git status --short

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

RUNS=(
    "m25_full_dit_edm_1000e_h128_l10_h4|edm_karras_heun_steps50_sigmin0p004_sigmax160_rho7_seed5678|benchmark_m25_full_dit_edm"
    "m27_full_dit_linear_ddpm_1000e_h128_l10_h4|linear_ddpm_reverse_sde_euler_maruyama_steps50_denoise_eps0p001_seed5678|benchmark_m27_full_dit_linear_ddpm"
    "m29_full_dit_flow_tscale1000_1000e_h128_l10_h4|flow_heun_steps50_seed5678|benchmark_m29_full_dit_flow_tscale1000"
)

for entry in "${RUNS[@]}"; do
    IFS="|" read -r RUN_NAME GEN_NAME BENCH_NAME <<< "$entry"

    GEN_DIR=${RUN_ROOT}/${RUN_NAME}/generations/${GEN_NAME}
    BENCH_ROOT=${RUN_ROOT}/${BENCH_NAME}
    BENCH_DIR=${BENCH_ROOT}/${GEN_NAME}

    echo
    echo "===== Refresh ${RUN_NAME} ====="
    echo "Generation: ${GEN_DIR}"
    echo "Benchmark:  ${BENCH_DIR}"

    for artifact in generated_test.pt summary.json resolved_config.yaml; do
        test -s "${GEN_DIR}/${artifact}" || {
            echo "ERROR: missing required generation artifact: ${GEN_DIR}/${artifact}"
            exit 1
        }
    done

    singularity exec \
        -B /home \
        -B /scratch \
        "$IMAGE" \
        "$PYTHON" -u -m scripts.benchmark_generation \
            run_dir="${GEN_DIR}" \
            output_root="${BENCH_ROOT}" \
            overwrite=true \
            energy_spectrum.enabled=true \
            energy_spectrum.lower_quantile=0.10 \
            energy_spectrum.upper_quantile=0.90 \
            plots.enabled=true \
            plots.marginals=true \
            plots.spectra=true \
            plots.energy_spectrum=true \
            plots.fields=true

    for file in \
        "${BENCH_DIR}/plots/marginals/Density_rho.png" \
        "${BENCH_DIR}/plots/marginals/X-momentum_rhou.png" \
        "${BENCH_DIR}/plots/marginals/Y-momentum_rhov.png" \
        "${BENCH_DIR}/plots/marginals/Z-momentum_rhow.png" \
        "${BENCH_DIR}/plots/marginals/Total_energy_rhoE.png" \
        "${BENCH_DIR}/plots/energy_spectrum/energy_spectrum_mean.png" \
        "${BENCH_DIR}/plots/energy_spectrum/energy_spectrum_median.png"
    do
        test -s "$file" || {
            echo "ERROR: expected refreshed plot was not generated: $file"
            exit 1
        }
    done
done

echo
echo "============================================================"
echo "BENCHMARK PLOT REFRESH COMPLETE"
echo "============================================================"
