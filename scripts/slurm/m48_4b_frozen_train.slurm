#!/bin/bash
#SBATCH --job-name=m48_frozen_fm
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail

# One independent architecture per job: identical FM draws, unbiased GPU
# allocation measurements, same frozen all-44-teacher normalization.
REPO=/scratch/coop/theret/GraphAttention
ROOT="${M48_MS_SR_ROOT:-/scratch/ai/MS_SR}"
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON="${M48_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}"
MODE="${M48_MODE:-}"
MAX_STEPS="${M48_MAX_STEPS:-4}"
OUT_ROOT="${M48_TRAIN_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m48_4b_frozen_train}"
OUT="${M48_RESUME_DIR:-$OUT_ROOT/job_${SLURM_JOB_ID}_$MODE}"
CEDAR_SRC="${M48_CEDAR_SRC:-}"
NORMALIZATION="$REPO/configs/m48/teacher_44_w1p25_l1_q4_q8.json"

trap 'rc=$?; echo "M48.4b failed: exit=$rc line=$LINENO command=$BASH_COMMAND" >&2' ERR

if [ "$MODE" != static ] && [ "$MODE" != dynamic ]; then
    echo "Set M48_MODE to static or dynamic" >&2
    exit 2
fi
if ! [[ "$MAX_STEPS" =~ ^[0-9]+$ ]]; then
    echo "M48_MAX_STEPS must be a nonnegative integer" >&2
    exit 2
fi
if [ "$MAX_STEPS" -eq 0 ] || [ "$MAX_STEPS" -gt 4 ]; then
    if [ "${M48_CONFIRM_LONG_TRAIN:-}" != YES ]; then
        echo "Set M48_CONFIRM_LONG_TRAIN=YES for more than 4 steps" >&2
        exit 2
    fi
fi
if [ -n "${M48_RESUME_DIR:-}" ]; then
    test -d "$OUT"
    RESUME_FLAG=--resume
else
    if [ -e "$OUT" ]; then
        echo "Refusing existing run directory $OUT" >&2
        exit 2
    fi
    RESUME_FLAG=
fi

cd "$REPO"
test -r "$ROOT/dataset/manifest.json"
test -r "$ROOT/code/campaign_data.py"
test -r "$ROOT/code/models.py"
test -r "$NORMALIZATION"
test -s "$IMAGE"
test -x "$PYTHON"
if [ -z "$CEDAR_SRC" ] && [ -d "$ROOT/cedar/src" ]; then
    CEDAR_SRC="$ROOT/cedar/src"
fi
if [ -n "$CEDAR_SRC" ]; then
    test -d "$CEDAR_SRC"
    export M48_CEDAR_SRC="$CEDAR_SRC"
    export SINGULARITYENV_M48_CEDAR_SRC="$CEDAR_SRC"
fi
export PYTHONPATH="$REPO/src:$REPO:$ROOT/code${CEDAR_SRC:+:$CEDAR_SRC}:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export OMP_NUM_THREADS=4
export SINGULARITYENV_OMP_NUM_THREADS=4

echo "=== M48.4b pinned CEDAR, frozen normalization and trainer qualification ==="
echo "Commit: $(git rev-parse HEAD)"
echo "Mode: $MODE; max optimizer steps this invocation: $MAX_STEPS"
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -c 'from preprocess import check_cedar; check_cedar(); print("Pinned CEDAR OK")'
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    scripts/m48_4b_frozen_train.py \
    tests/unit/test_m48_4b_frozen_train.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m48_ms_sr.py \
    tests/unit/test_m48_ms_sr_training.py \
    tests/unit/test_m48_4b_frozen_train.py

echo "=== Real MS_SR teacher training: $MODE ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m48_4b_frozen_train \
    --ms-sr-root "$ROOT" \
    --normalization-file "$NORMALIZATION" \
    --output-dir "$OUT" \
    --mode "$MODE" \
    --epochs "${M48_EPOCHS:-8}" \
    --patches-per-source "${M48_PATCHES_PER_SOURCE:-4}" \
    --core-side "${M48_CORE_SIDE:-4}" \
    --width "${M48_WIDTH:-32}" \
    --heads "${M48_HEADS:-4}" \
    --depth "${M48_DEPTH:-1}" \
    --learning-rate "${M48_LEARNING_RATE:-0.0003}" \
    --checkpoint-every "${M48_CHECKPOINT_EVERY:-20}" \
    --max-steps "$MAX_STEPS" \
    --seed "${M48_SEED:-484}" \
    --device cuda \
    $RESUME_FLAG
test -s "$OUT/config.json"
test -s "$OUT/normalization.json"
test -s "$OUT/checkpoint_latest.pt"
test -s "$OUT/summary.json"
echo "PASS: M48.4b $MODE finished bounded requested optimizer steps."
echo "$OUT"
