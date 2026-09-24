#!/bin/bash
#SBATCH --job-name=full_dit_tscale_plots
#SBATCH --time=00:20:00
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

EDM_GEN_NAME=edm_karras_heun_steps50_sigmin0p004_sigmax160_rho7_seed5678
DDPM_GEN_NAME=linear_ddpm_reverse_sde_euler_maruyama_steps50_denoise_eps0p001_seed5678
FLOW_GEN_NAME=flow_heun_steps50_seed5678

EDM_GEN=${RUN_ROOT}/m25_full_dit_edm_1000e_h128_l10_h4/generations/${EDM_GEN_NAME}
DDPM_GEN=${RUN_ROOT}/m27_full_dit_linear_ddpm_1000e_h128_l10_h4/generations/${DDPM_GEN_NAME}
FLOW_GEN=${RUN_ROOT}/m29_full_dit_flow_tscale1000_1000e_h128_l10_h4/generations/${FLOW_GEN_NAME}

EDM_BENCH=${RUN_ROOT}/benchmark_m25_full_dit_edm/${EDM_GEN_NAME}
DDPM_BENCH=${RUN_ROOT}/benchmark_m27_full_dit_linear_ddpm/${DDPM_GEN_NAME}
FLOW_BENCH=${RUN_ROOT}/benchmark_m29_full_dit_flow_tscale1000/${FLOW_GEN_NAME}

OUTPUT_DIR=${RUN_ROOT}/full_dit_ddpm_edm_flow_tscale1000_plots

cd "$REPO"

echo "============================================================"
echo "Full DiT comparison plots: EDM / DDPM / Flow Matching"
echo "Flow Matching source: M29, t-scale=1000"
echo "============================================================"
git rev-parse HEAD
git status --short

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for file in \
    "${EDM_GEN}/generated_test.pt" \
    "${DDPM_GEN}/generated_test.pt" \
    "${FLOW_GEN}/generated_test.pt" \
    "${EDM_BENCH}/energy_spectrum_summary.csv" \
    "${DDPM_BENCH}/energy_spectrum_summary.csv" \
    "${FLOW_BENCH}/energy_spectrum_summary.csv"
do
    test -s "$file" || {
        echo "ERROR: missing required artifact: $file"
        exit 1
    }
done

if [ -d "$OUTPUT_DIR" ]; then
    echo "Replacing existing comparison plot directory:"
    echo "  $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.plot_full_dit_comparison \
        --edm-generation "${EDM_GEN}/generated_test.pt" \
        --ddpm-generation "${DDPM_GEN}/generated_test.pt" \
        --flow-generation "${FLOW_GEN}/generated_test.pt" \
        --edm-energy-summary "${EDM_BENCH}/energy_spectrum_summary.csv" \
        --ddpm-energy-summary "${DDPM_BENCH}/energy_spectrum_summary.csv" \
        --flow-energy-summary "${FLOW_BENCH}/energy_spectrum_summary.csv" \
        --output-dir "$OUTPUT_DIR" \
        --bins 128 \
        --dpi 220

for file in \
    "${OUTPUT_DIR}/pdf/pdf_Density_rho.png" \
    "${OUTPUT_DIR}/pdf/pdf_X-momentum_rhou.png" \
    "${OUTPUT_DIR}/pdf/pdf_Y-momentum_rhov.png" \
    "${OUTPUT_DIR}/pdf/pdf_Z-momentum_rhow.png" \
    "${OUTPUT_DIR}/pdf/pdf_Total_energy_rhoE.png" \
    "${OUTPUT_DIR}/energy_spectrum_comparison_mean.png" \
    "${OUTPUT_DIR}/energy_spectrum_comparison_median.png"
do
    test -s "$file" || {
        echo "ERROR: expected plot was not generated: $file"
        exit 1
    }
done

echo
echo "PDF comparisons:"
echo "  ${OUTPUT_DIR}/pdf/"
echo "Energy spectrum comparisons:"
echo "  ${OUTPUT_DIR}/energy_spectrum_comparison_mean.png"
echo "  ${OUTPUT_DIR}/energy_spectrum_comparison_median.png"
echo "============================================================"
