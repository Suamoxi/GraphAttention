#!/bin/bash
#SBATCH --job-name=m45_7_3_agg
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail

# Submit with --dependency=afterok:<EVAL_ARRAY_ID> and
# --export=ALL,M45_7_3_PARENT_JOB=<EVAL_ARRAY_ID>.
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
ARRAY_ID="${M45_7_3_PARENT_JOB:?set M45_7_3_PARENT_JOB to the evaluation array ID}"
OUT_ROOT="${M45_7_3_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m45_7_3_multi_coarse}"
PARENT="${OUT_ROOT}/job_${ARRAY_ID}"
OUT="${PARENT}/aggregate"
STEPS="${M45_7_3_HEUN_STEPS:-200}"
SAMPLES="${M45_7_3_GENERATED_SAMPLES:-128}"
MAX_FIELDS="${M45_7_3_MAX_FIELDS:-8}"
EVAL_SEED="${M45_7_3_EVAL_SEED:-53177}"

cd "$REPO"
test -s "$IMAGE"
test -x "$PYTHON"
test ! -e "$OUT"
echo "M45.7.3 aggregate Git=$(git rev-parse HEAD) array=$ARRAY_ID"

singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile scripts/m45_7_3_aggregate.py

singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_7_3_aggregate \
    --input-dir "$PARENT" \
    --output-dir "$OUT" \
    --expected-fields "$MAX_FIELDS" \
    --heun-steps "$STEPS" \
    --generated-samples "$SAMPLES" \
    --eval-seed "$EVAL_SEED"

test -s "$OUT/summary.json"
test -s "$OUT/paired_per_field.csv"
test -s "$OUT/per_seed.csv"
test -s "$OUT/per_validation_field.csv"
echo "M45.7.3 aggregate complete: $OUT"
