#!/bin/bash
#SBATCH --job-name=m21_m23_noise_diag
#SBATCH --time=02:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

RUN_ROOT=/scratch/coop/theret/GraphAttention_runs
M21_RUN=${RUN_ROOT}/m21_m12_diffusion_linear_1000e_h128_l10_h4
M23_RUN=${RUN_ROOT}/m23_m12_vp_sde_1000e_h128_l10_h4
OUTPUT_ROOT=${RUN_ROOT}/noise_level_diagnostics
OUTPUT_NAME=m21_vs_m23_paired_noise_spectrum_20levels
OUTPUT_DIR=${OUTPUT_ROOT}/${OUTPUT_NAME}

cd "$REPO"

printf '%s\n' "============================================================"
printf '%s\n' "M21 vs M23 paired noise-level denoising diagnostic"
printf '%s\n' "============================================================"
printf 'M21:        %s\n' "$M21_RUN"
printf 'M23:        %s\n' "$M23_RUN"
printf 'Checkpoint: best.pt for both\n'
printf 'Levels:     20 shared normalized t values\n'
printf 'Noise:      same deterministic field per sample across runs and levels\n'
printf 'Spectra:    24 radial bins, low/mid/high benchmark bands\n'
printf '%s\n' "============================================================"

echo
echo "===== Git ====="
git rev-parse HEAD
git status --short

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for run_dir in "$M21_RUN" "$M23_RUN"; do
    for artifact in \
        resolved_config.yaml \
        summary.json \
        dataset_split_manifest.json \
        standardizers.pt \
        best.pt
    do
        if [ ! -f "${run_dir}/${artifact}" ]; then
            echo "ERROR: missing source artifact: ${run_dir}/${artifact}"
            exit 1
        fi
    done
done

if [ -e "$OUTPUT_DIR" ]; then
    echo "ERROR: diagnostic output already exists: $OUTPUT_DIR"
    exit 1
fi

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== Running paired diagnostic ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.compare_noise_level_denoising \
        baseline_run_dir="$M21_RUN" \
        candidate_run_dir="$M23_RUN" \
        baseline_checkpoint=best.pt \
        candidate_checkpoint=best.pt \
        output_root="$OUTPUT_ROOT" \
        output_name="$OUTPUT_NAME" \
        device=cuda \
        batch_size=128 \
        num_workers=4 \
        diagnostic_seed=2468

echo
echo "===== Checking outputs ====="
for artifact in \
    summary.json \
    diagnostic_config.yaml \
    baseline_denoising.csv \
    candidate_denoising.csv \
    baseline_spectral_curves.csv \
    candidate_spectral_curves.csv \
    baseline_spectral_bands.csv \
    candidate_spectral_bands.csv \
    noise_schedule_comparison.csv \
    denoising_comparison.csv \
    spectral_band_comparison.csv
do
    test -f "${OUTPUT_DIR}/${artifact}"
done

echo
echo "===== Summary ====="
cat "${OUTPUT_DIR}/summary.json"

echo
echo "===== Noise schedule comparison ====="
cat "${OUTPUT_DIR}/noise_schedule_comparison.csv"

echo
echo "===== Mean-channel denoising comparison ====="
awk -F, 'NR == 1 || $2 == "__mean__"' "${OUTPUT_DIR}/denoising_comparison.csv"

echo
echo "============================================================"
echo "M21 vs M23 NOISE-LEVEL DIAGNOSTIC COMPLETE"
echo "$OUTPUT_DIR"
echo "============================================================"
