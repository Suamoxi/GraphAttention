#!/bin/bash
#SBATCH --job-name=m48_norm_audit
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail

# Teacher-only, CPU statistical audit. GPU allocation follows the proven
# Grace/Singularity runtime for M48; the job does not train a model.
REPO=/scratch/coop/theret/GraphAttention
ROOT="${M48_MS_SR_ROOT:-/scratch/ai/MS_SR}"
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON="${M48_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}"
OUT_ROOT="${M48_AUDIT_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m48_normalization_audit}"
OUT="$OUT_ROOT/job_$SLURM_JOB_ID"
CEDAR_SRC="${M48_CEDAR_SRC:-}"

trap 'rc=$?; echo "M48 normalization audit failed: rc=$rc line=$LINENO command=$BASH_COMMAND" >&2' ERR
cd "$REPO"
test -r "$ROOT/dataset/manifest.json"
test -r "$ROOT/code/campaign_data.py"
test -r "$ROOT/code/models.py"
test -s "$IMAGE"
test -x "$PYTHON"
if [ -e "$OUT" ]; then
    echo "Refusing existing audit output: $OUT" >&2
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
export OMP_NUM_THREADS=4
export SINGULARITYENV_OMP_NUM_THREADS=4

echo "=== M48 teacher normalization audit: preflight ==="
echo "GraphAttention revision: $(git rev-parse HEAD)"
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -c 'from preprocess import check_cedar; check_cedar(); print("Pinned CEDAR OK")'
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    src/graph_attention/data/ms_sr_normalization_audit.py \
    scripts/m48_normalization_audit.py \
    tests/unit/test_m48_ms_sr_normalization_audit.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m48_ms_sr_training.py \
    tests/unit/test_m48_ms_sr_normalization_audit.py

echo "=== Read-only audit: all 44 teacher sources, both ReT groups ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m48_normalization_audit \
    --ms-sr-root "$ROOT" \
    --output-dir "$OUT" \
    --filter-id w1p25_l1 \
    --case-prefix "" \
    --max-sources 0 \
    --expected-sources 44

test -s "$OUT/normalization_audit.json"
test -s "$OUT/case_comparison.csv"
test -s "$OUT/snapshot_moments.csv"
echo "PASS: teacher normalization audit."
echo "$OUT"
