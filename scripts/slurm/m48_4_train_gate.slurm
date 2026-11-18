#!/bin/bash
#SBATCH --job-name=m48_4_train_gate
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail

# Bounded real MS_SR multi-source training qualification, not a campaign.
REPO=/scratch/coop/theret/GraphAttention
ROOT="${M48_MS_SR_ROOT:-/scratch/ai/MS_SR}"
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON="${M48_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}"
OUT_ROOT="${M48_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m48_4_train_gate}"
OUT="$OUT_ROOT/job_$SLURM_JOB_ID"
CEDAR_SRC="${M48_CEDAR_SRC:-}"

trap 'rc=$?; echo "M48.4 FAILED: exit=$rc line=$LINENO command=$BASH_COMMAND" >&2' ERR
cd "$REPO"
test -f "$ROOT/dataset/manifest.json"
test -f "$ROOT/code/campaign_data.py"
test -f "$ROOT/code/offline_loader.py"
test -f "$ROOT/code/models.py"
test -f "$ROOT/code/preprocess.py"
test -s "$IMAGE"
test -x "$PYTHON"
if [ -e "$OUT" ]; then
    echo "M48.4 output already exists: $OUT" >&2
    exit 2
fi

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

echo "=== M48.4 pinned-CEDAR and manifest preflight ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -c 'import cedar; from preprocess import check_cedar; check_cedar(); print(cedar.__file__)'
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -c 'import hashlib,sys; from pathlib import Path; raw=Path(sys.argv[1]).read_bytes(); digest=hashlib.sha256(raw).hexdigest(); assert digest==sys.argv[2], digest; print("Manifest OK:",digest)' \
  "$ROOT/dataset/manifest.json" \
  993f791f88d801e44e5feb0cee927c4c48eaa459bd6171e9e364b002bb01fdca

echo "=== Python syntax and M48 regression tests ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m py_compile \
  src/graph_attention/data/ms_sr_adapter.py \
  src/graph_attention/data/ms_sr_training.py \
  scripts/m48_ms_sr_train.py \
  tests/unit/test_m48_ms_sr_training.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q \
  tests/unit/test_m48_ms_sr.py \
  tests/unit/test_m48_ms_sr_training.py

echo "=== Bounded two-source, two-mode training qualification ==="
echo "Git $(git rev-parse HEAD); MS_SR $ROOT; out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.m48_ms_sr_train \
  --ms-sr-root "$ROOT" --output-dir "$OUT" \
  --case-prefix "${M48_CASE_PREFIX:-ReT060/}" \
  --filter-id "${M48_FILTER_ID:-w1p25_l1}" \
  --max-sources "${M48_MAX_SOURCES:-2}" \
  --steps "${M48_STEPS:-2}" \
  --core-side "${M48_CORE_SIDE:-2}" \
  --depth "${M48_DEPTH:-1}" \
  --width "${M48_WIDTH:-16}" \
  --heads "${M48_HEADS:-4}" \
  --seed "${M48_SEED:-484}" \
  --device cuda
test -s "$OUT/summary.json"
test -s "$OUT/normalization.json"
test -s "$OUT/static.pt"
test -s "$OUT/dynamic.pt"
echo "PASS: M48.4 bounded real-data training qualification."
echo "$OUT"
