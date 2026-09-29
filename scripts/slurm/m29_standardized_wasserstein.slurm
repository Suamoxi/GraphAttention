#!/bin/bash
#SBATCH --job-name=m29_norm_w1
#SBATCH --time=00:10:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
RUN_ROOT=/scratch/coop/theret/GraphAttention_runs

RUN_NAME=m29_full_dit_flow_tscale1000_1000e_h128_l10_h4
GEN_NAME=flow_heun_steps50_seed5678
BENCH_NAME=benchmark_m29_full_dit_flow_tscale1000

RUN_DIR=${RUN_ROOT}/${RUN_NAME}
GEN_DIR=${RUN_DIR}/generations/${GEN_NAME}
BENCH_DIR=${RUN_ROOT}/${BENCH_NAME}/${GEN_NAME}

GENERATION=${GEN_DIR}/generated_test.pt
STANDARDIZERS=${RUN_DIR}/standardizers.pt
OUTPUT=${BENCH_DIR}/normalized_wasserstein.csv

cd "$REPO"

echo "============================================================"
echo "M29 normalized-space Wasserstein-1"
echo "Normalization: frozen training-set standardization"
echo "============================================================"
git rev-parse HEAD
git status --short

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for file in "$GENERATION" "$STANDARDIZERS" "${BENCH_DIR}/summary.json"; do
    test -s "$file" || {
        echo "ERROR: missing required artifact: $file"
        exit 1
    }
done

singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.calculate_standardized_wasserstein \
        --generation "$GENERATION" \
        --standardizers "$STANDARDIZERS" \
        --output "$OUTPUT"

test -s "$OUTPUT"

echo
echo "============================================================"
echo "M29 NORMALIZED WASSERSTEIN COMPLETE"
echo "Output:"
echo "  $OUTPUT"
echo "============================================================"
