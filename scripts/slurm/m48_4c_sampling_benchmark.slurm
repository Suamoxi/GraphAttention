#!/bin/bash
#SBATCH --job-name=m48c_batch_bench
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail

# M48.4c C: 12^3 owned, 16 patches/source/epoch, effective B=4.
# Each run probes one representative authenticated teacher source on each
# of the 48^3 and 80^3 grids, NOT a full epoch or a convergence experiment.
REPO=/scratch/coop/theret/GraphAttention
ROOT="${M48_MS_SR_ROOT:-/scratch/ai/MS_SR}"
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON="${M48_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}"
MODE="${M48_MODE:-}"
MICRO="${M48_MICROBATCH_SIZE:-2}"
OUT_ROOT="${M48_TRAIN_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m48_4c_sampling_benchmark}"
OUT="${M48_RESUME_DIR:-$OUT_ROOT/job_${SLURM_JOB_ID}_${MODE}_micro${MICRO}}"
CEDAR_SRC="${M48_CEDAR_SRC:-}"
NORMALIZATION="$REPO/configs/m48/teacher_44_w1p25_l1_q4_q8.json"
RESUME_FLAG=
trap 'rc=$?; echo "M48.4c failed: exit=$rc line=$LINENO command=$BASH_COMMAND" >&2' ERR

case "$MODE" in static|dynamic) ;; *)
    echo "Set M48_MODE=static or M48_MODE=dynamic" >&2; exit 2 ;;
esac
case "$MICRO" in 1|2|4) ;; *)
    echo "M48_MICROBATCH_SIZE must be one of 1, 2, 4" >&2; exit 2 ;;
esac
if [ -n "${M48_RESUME_DIR:-}" ]; then
    test -d "$OUT"
    RESUME_FLAG=--resume
else
    test ! -e "$OUT" || {
        echo "Output directory exists: $OUT" >&2; exit 2;
    }
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

echo "=== M48.4c configuration-C bounded exact-halo, microbatch qualification ==="
echo "Commit: $(git rev-parse HEAD)"
echo "Mode=$MODE microbatch=$MICRO effective_batch=4 core=12"
echo "Fine grid coverage: 48^3 and 80^3 via teacher-only probe"

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -c 'from preprocess import check_cedar; check_cedar(); print("Pinned CEDAR OK")'
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    scripts/m48_4b_frozen_train.py \
    scripts/m48_4c_patch_batch.py \
    tests/unit/test_m48_4b_frozen_train.py \
    tests/unit/test_m48_4c_patch_batch.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m48_ms_sr.py \
    tests/unit/test_m48_ms_sr_training.py \
    tests/unit/test_m48_4b_frozen_train.py \
    tests/unit/test_m48_4c_patch_batch.py

echo "=== M48.4c GPU probe (2 updates / mode / microbatch) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m48_4b_frozen_train \
    --ms-sr-root "$ROOT" \
    --normalization-file "$NORMALIZATION" \
    --output-dir "$OUT" \
    --mode "$MODE" \
    --epochs 1 \
    --patches-per-source 16 \
    --core-side 12 \
    --effective-batch 4 \
    --microbatch-size "$MICRO" \
    --probe-grids \
    --width 32 --heads 4 --depth 1 \
    --learning-rate 0.0003 \
    --checkpoint-every 1 \
    --max-steps 2 \
    --seed 484 --device cuda \
    $RESUME_FLAG
for f in config.json normalization.json checkpoint_latest.pt summary.json halo_preflight.json; do
    test -s "$OUT/$f"
done
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" - "$OUT/summary.json" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1]))
assert p["completed_steps"] == 2, p["completed_steps"]
assert p["schedule"] == "representative_teacher_grid_probe"
assert p["effective_batch"] == 4
assert len({x["case"].split("/")[0] for x in p["history"]}) == 2
assert max(x["projected_source_abs_defect"] for x in p["history"]) < 1e-9
print("PASS: two distinct teacher grids, projected noise and completed batch steps")
PY
echo "PASS: M48.4c $MODE microbatch=$MICRO"
echo "$OUT"
