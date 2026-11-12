#!/bin/bash
#SBATCH --job-name=m45_8_compare
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail

# Submit after the entire 0..3 GPU array succeeds.
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
ARRAY_ID="${M45_8_PARENT_JOB:?set evaluation array ID}"
OUT_ROOT="${M45_8_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m45_8_local}"
ROOT="${OUT_ROOT}/job_${ARRAY_ID}"
OUT="${ROOT}/aggregate"

cd "$REPO"
test -s "$IMAGE"
test -x "$PYTHON"
test ! -e "$OUT"
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_8_compare --root "$ROOT" --output-dir "$OUT"
test -s "$OUT/summary.json"
test -s "$OUT/scorecard.csv"
test -s "$OUT/conditional_curves.csv"
echo "M45.8.1 comparison: $OUT"
