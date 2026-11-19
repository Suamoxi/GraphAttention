#!/bin/bash
#SBATCH --job-name=m48f_three_mode
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail

# Isolated static/dynamic two-grid M48.4f execution benchmark.
# Does not alter the M48 scientific trainer, checkpoint, or source set.
REPO=/scratch/coop/theret/GraphAttention
ROOT="${M48_MS_SR_ROOT:-/scratch/ai/MS_SR}"
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON="${M48_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}"
MODE="${M48_MODE:-}"
WIDTH="${M48_WIDTH:-64}"
DEPTH="${M48_DEPTH:-2}"
CORE="${M48_CORE_SIDE:-12}"
STEPS="${M48_STEPS_PER_GRID:-4}"
OUT_ROOT="${M48_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m48_4f_three_mode_v1}"
OUT="$OUT_ROOT/job_${SLURM_JOB_ID}_${MODE}"
NORMALIZATION="$REPO/configs/m48/teacher_44_w1p25_l1_q4_q8.json"
CEDAR_SRC="${M48_CEDAR_SRC:-}"
trap 'rc=$?; echo "M48.4f failed: exit=$rc line=$LINENO command=$BASH_COMMAND" >&2' ERR

case "$MODE" in static|dynamic) ;; *)
    echo "Set M48_MODE=static or M48_MODE=dynamic" >&2; exit 2 ;;
esac
cd "$REPO"
test -r "$ROOT/dataset/manifest.json"
test -r "$ROOT/code/campaign_data.py"
test -r "$ROOT/code/models.py"
test -r "$NORMALIZATION"
test -s "$IMAGE"
test -x "$PYTHON"
test ! -e "$OUT"
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
export OMP_NUM_THREADS=6
export SINGULARITYENV_OMP_NUM_THREADS=6

echo "=== M48.4f three-way real-CEDAR exact-halo SR benchmark ==="
echo "Git revision: $(git rev-parse HEAD)"
echo "Mode=$MODE width=$WIDTH depth=$DEPTH core=$CORE effective_batch=4"
echo "Steps/grid=$STEPS"
echo "OUTPUT_DIR=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -c 'from preprocess import check_cedar; check_cedar(); print("Pinned CEDAR OK")'

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    src/graph_attention/models/two_scale_dinat.py \
    scripts/m48_4f_patch_execution.py \
    scripts/m48_4f_compare.py \
    tests/unit/test_m48_4f_patch_execution.py

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m48_4c_patch_batch.py \
    tests/unit/test_m48_4f_patch_execution.py

echo "=== M48.4f three-mode 48^3 and 80^3 real-CEDAR GPU comparison ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m48_4f_compare \
    --ms-sr-root "$ROOT" \
    --normalization-file "$NORMALIZATION" \
    --output-dir "$OUT" --mode "$MODE" \
    --core-side "$CORE" --width "$WIDTH" --heads 4 --depth "$DEPTH" \
    --effective-batch 4 --steps-per-grid "$STEPS" --seed 484

test -s "$OUT/summary.json"
echo "ARTIFACT_DIR=$OUT"
echo "ARTIFACT_MANIFEST=$OUT/manifest.json"
echo "ARTIFACT_SUMMARY=$OUT/summary.json"
echo "=== M48.4f benchmark completed; scientific training is unchanged ==="
