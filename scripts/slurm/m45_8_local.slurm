#!/bin/bash
#SBATCH --job-name=m45_8_local
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%A_%a.err
set -euo pipefail

# M45.8.1: four matched jobs (static/dynamic) x (coarse radii 1/2).
# Never execute PyTorch on a login node.
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
TASK="${SLURM_ARRAY_TASK_ID:?submit with --array=0-3}"
case "$TASK" in
    0) MODE=static;  RADIUS=1 ;;
    1) MODE=dynamic; RADIUS=1 ;;
    2) MODE=static;  RADIUS=2 ;;
    3) MODE=dynamic; RADIUS=2 ;;
    *) echo "M45.8 array requires tasks 0..3" >&2; exit 2 ;;
esac

COARSE_SIDE="${M45_8_COARSE_SIDE:-8}"
REFINEMENT="${M45_8_REFINEMENT:-2}"
TRAIN_FIELDS="${M45_8_TRAIN_FIELDS:-2048}"
VAL_FIELDS="${M45_8_VAL_FIELDS:-16}"
NORMAL_FIELDS="${M45_8_NORMAL_FIELDS:-128}"
STEPS="${M45_8_STEPS:-20000}"
WIDTH="${M45_8_WIDTH:-32}"
DEPTH="${M45_8_DEPTH:-4}"
HEADS="${M45_8_HEADS:-4}"
LR="${M45_8_LR:-0.001}"
POLICY="${M45_8_POLICY:-resample}"
DATA_SEED="${M45_8_DATA_SEED:-71001}"
RESIDUAL_SEED="${M45_8_RESIDUAL_SEED:-72001}"
INIT_SEED="${M45_8_INIT_SEED:-73001}"
SCHEDULE_SEED="${M45_8_SCHEDULE_SEED:-74001}"
VALIDATION_SEED="${M45_8_VALIDATION_SEED:-75001}"
DRAWS="${M45_8_DRAWS:-16}"
HEUN="${M45_8_HEUN_STEPS:-50}"
EVAL_SEED="${M45_8_EVAL_SEED:-76001}"
OUT_ROOT="${M45_8_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m45_8_local}"
OUT="${OUT_ROOT}/job_${SLURM_ARRAY_JOB_ID}/task_${TASK}_${MODE}_r${RADIUS}"

cd "$REPO"
test -s "$IMAGE"
test -x "$PYTHON"
if [ -e "$OUT" ]; then
    echo "Refusing existing output: $OUT" >&2
    exit 2
fi

echo "M45.8.1 code=$(git rev-parse HEAD) SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "mode=$MODE context_hops=$RADIUS data=$DATA_SEED residual=$RESIDUAL_SEED init=$INIT_SEED schedule=$SCHEDULE_SEED validation=$VALIDATION_SEED"
echo "coarse=$COARSE_SIDE x refinement=$REFINEMENT, ntrain=$TRAIN_FIELDS, nval=$VAL_FIELDS, steps=$STEPS, width=$WIDTH"
nvidia-smi --query-gpu=name,memory.total --format=csv

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    src/graph_attention/data/conditional_hit_synthetic.py \
    src/graph_attention/models/two_scale_dinat.py \
    scripts/m45_8_local_train.py \
    scripts/m45_8_local_evaluate.py scripts/m45_8_compare.py \
    tests/unit/test_m45_8_local.py

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q tests/unit/test_m45_8_local.py

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_8_local_train \
    --output-dir "$OUT" \
    --mode "$MODE" --device cuda \
    --residual-policy "$POLICY" \
    --coarse-side "$COARSE_SIDE" --refinement "$REFINEMENT" \
    --train-fields "$TRAIN_FIELDS" --val-fields "$VAL_FIELDS" \
    --normalization-fields "$NORMAL_FIELDS" \
    --steps "$STEPS" --depth "$DEPTH" --width "$WIDTH" \
    --heads "$HEADS" --learning-rate "$LR" \
    --context-hops "$RADIUS" \
    --dataset-seed "$DATA_SEED" --residual-seed "$RESIDUAL_SEED" \
    --init-seed "$INIT_SEED" --schedule-seed "$SCHEDULE_SEED" \
    --validation-seed "$VALIDATION_SEED"

test -s "$OUT/summary.json"
test -s "$OUT/local_hit/block_average/$MODE/checkpoint.pt"

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_8_local_evaluate \
    --checkpoint-root "$OUT" --output-dir "$OUT/diagnostics" \
    --draws "$DRAWS" --heun-steps "$HEUN" \
    --val-fields "$VAL_FIELDS" --eval-seed "$EVAL_SEED" \
    --device cuda

test -s "$OUT/diagnostics/summary.json"
test -s "$OUT/diagnostics/conditional_energy.csv"
test -s "$OUT/diagnostics/conditional_summary.csv"
test -s "$OUT/diagnostics/spectra.csv"
test -s "$OUT/diagnostics/spatial_correlations.csv"
echo "M45.8.1 complete: $OUT"
