#!/bin/bash
#SBATCH --job-name=m45_7_3_coarse
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%A_%a.err
set -euo pipefail

# Six existing M45.7.2 checkpoints: 2026/2027/2028 x fixed/resampled.
# No new training; all Python, tests and generation run inside GPU Slurm jobs.
REPO=/scratch/coop/theret/GraphAttention
RUNS=/scratch/coop/theret/GraphAttention_runs/m45_7_2_resampling
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

case "${SLURM_ARRAY_TASK_ID:?SLURM array required}" in
    0) SEED=2026; POLICY=fixed;    TRAIN=job_503042_2_static_oracle_fixed_10000steps ;;
    1) SEED=2026; POLICY=resample; TRAIN=job_503042_3_static_oracle_resample_10000steps ;;
    2) SEED=2027; POLICY=fixed;    TRAIN=job_503043_2_static_oracle_fixed_10000steps ;;
    3) SEED=2027; POLICY=resample; TRAIN=job_503043_3_static_oracle_resample_10000steps ;;
    4) SEED=2028; POLICY=fixed;    TRAIN=job_503044_2_static_oracle_fixed_10000steps ;;
    5) SEED=2028; POLICY=resample; TRAIN=job_503044_3_static_oracle_resample_10000steps ;;
    *) echo "M45.7.3 requires array task IDs 0..5" >&2; exit 2 ;;
esac

STEPS="${M45_7_3_HEUN_STEPS:-200}"
SAMPLES="${M45_7_3_GENERATED_SAMPLES:-128}"
MAX_FIELDS="${M45_7_3_MAX_FIELDS:-8}"
EVAL_SEED="${M45_7_3_EVAL_SEED:-53177}"
OUT_ROOT="${M45_7_3_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m45_7_3_multi_coarse}"
CKPT="${RUNS}/${TRAIN}"
OUT="${OUT_ROOT}/job_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}_${SEED}_${POLICY}"

cd "$REPO"
test -s "$CKPT/summary.json"
test -s "$CKPT/shared_mode/block_average/static/checkpoint.pt"
test -s "$IMAGE"
test -x "$PYTHON"
if [ -e "$OUT" ]; then
    echo "Refusing to overwrite existing output: $OUT" >&2
    exit 2
fi

echo "M45.7.3 Git=$(git rev-parse HEAD) Slurm=${SLURM_JOB_ID}"
echo "checkpoint=$CKPT seed=$SEED policy=$POLICY fields=$MAX_FIELDS samples=$SAMPLES Heun=$STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    scripts/m45_7_3_condition_sweep.py scripts/m45_7_3_aggregate.py \
    tests/unit/test_m45_7_3_condition_sweep.py

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q tests/unit/test_m45_7_3_condition_sweep.py

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_7_3_condition_sweep \
    --checkpoint-root "$CKPT" \
    --output-dir "$OUT" \
    --seed "$SEED" \
    --residual-policy "$POLICY" \
    --heun-steps "$STEPS" \
    --generated-samples "$SAMPLES" \
    --max-fields "$MAX_FIELDS" \
    --eval-seed "$EVAL_SEED" \
    --device cuda

test -s "$OUT/summary.json"
test -s "$OUT/per_field.csv"
test -s "$OUT/mode_amplitudes.csv"
test -s "$OUT/spatial_correlations.csv"
echo "M45.7.3 completed: $OUT"
