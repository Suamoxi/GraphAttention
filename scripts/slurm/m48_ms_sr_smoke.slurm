#!/bin/bash
#SBATCH --job-name=m48_ms_sr_smoke
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail

# M48.1-M48.3 single real teacher snapshot, full-domain CEDAR and
# target-driven halo static/dynamic core-masked FM gradient qualification.
# NO scientific training campaign; NO writes into MS_SR.
REPO=/scratch/coop/theret/GraphAttention
ROOT="${M48_MS_SR_ROOT:-/scratch/ai/MS_SR}"
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON="${M48_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}"
OUT_ROOT="${M48_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m48_ms_sr_smoke}"
OUT="$OUT_ROOT/job_${SLURM_JOB_ID}"
FILTER="${M48_FILTER_ID:-w1p25_l1}"
CASE="${M48_CASE_PREFIX:-ReT060/}"
DEPTH="${M48_DEPTH:-1}"
CORE="${M48_CORE_SIDE:-2}"
WIDTH="${M48_WIDTH:-16}"
HEADS="${M48_HEADS:-4}"

# Slurm can append its energy receipt even when an early bare `test` fails.
# Print every preflight dependency and leave an actionable diagnostic.
trap 'rc=$?; echo "M48 FAILURE: exit=${rc}, line=${LINENO}, command=${BASH_COMMAND}" >&2' ERR
echo "=== M48 preflight on $(hostname); job=${SLURM_JOB_ID:-unavailable} ==="
echo "Repository: $REPO"
echo "MS_SR root: $ROOT"
echo "Singularity image: $IMAGE"
echo "Python executable: $PYTHON"
echo "Output: $OUT"

require_file() {
    if [ ! -f "$1" ]; then
        echo "M48 PREFLIGHT FAILED: missing file $1" >&2
        return 2
    fi
}
require_dir() {
    if [ ! -d "$1" ]; then
        echo "M48 PREFLIGHT FAILED: missing directory $1" >&2
        return 2
    fi
}
require_executable() {
    if [ ! -x "$1" ]; then
        echo "M48 PREFLIGHT FAILED: missing/non-executable Python $1" >&2
        return 2
    fi
}
require_nonempty_file() {
    if [ ! -s "$1" ]; then
        echo "M48 PREFLIGHT FAILED: missing/empty file $1" >&2
        return 2
    fi
}
require_dir "$REPO"
cd "$REPO"
require_executable "$PYTHON"
require_nonempty_file "$IMAGE"
if [ -e "$OUT" ]; then
    echo "M48 PREFLIGHT FAILED: output already exists: $OUT" >&2
    exit 2
fi
require_file "$ROOT/dataset/manifest.json"
require_file "$ROOT/code/campaign_data.py"
require_file "$ROOT/code/models.py"
CEDAR_SRC="${M48_CEDAR_SRC:-}"
if [ -n "$CEDAR_SRC" ]; then
    require_dir "$CEDAR_SRC"
elif [ -d "$ROOT/cedar/src" ]; then
    CEDAR_SRC="$ROOT/cedar/src"
fi
if [ -n "$CEDAR_SRC" ]; then
    export M48_CEDAR_SRC="$CEDAR_SRC"
    export SINGULARITYENV_M48_CEDAR_SRC="$CEDAR_SRC"
    echo "CEDAR source location: $CEDAR_SRC"
else
    echo "CEDAR source checkout not found; testing the installed cedar-cfd package."
fi

# Import the actual external MS_SR + pinned CEDAR package, not a
# presumed checkout path. Singularity must receive this PYTHONPATH.
export PYTHONPATH="$REPO/src:$REPO:$ROOT/code${CEDAR_SRC:+:$CEDAR_SRC}:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
echo "=== Verify pinned CEDAR runtime (using MS_SR source hashes) ==="
if ! singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -c 'import cedar; from preprocess import check_cedar; check_cedar(); print("Verified CEDAR: " + str(cedar.__file__))'; then
    echo "M48 PREFLIGHT FAILED: CEDAR is missing or does not match MS_SR's pinned files." >&2
    echo "Provide M48_CEDAR_SRC=/path/to/cedar/src or M48_PYTHON=/path/to/compatible/python." >&2
    exit 2
fi
echo "M48 preflight passed."

echo "M48 real-MS_SR smoke; revision=$(git rev-parse HEAD); job=${SLURM_JOB_ID}"
echo "Data (read-only): $ROOT"
echo "Filter=$FILTER case_prefix=$CASE depth=$DEPTH core_side=$CORE width=$WIDTH"
nvidia-smi --query-gpu=name,memory.total --format=csv

# The required GraphAttention, MS_SR and optional CEDAR module paths
# were exported to the container during preflight.

echo "=== Syntax check ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m py_compile \
  src/graph_attention/geometry/two_scale_targeted.py \
  src/graph_attention/data/ms_sr_adapter.py \
  scripts/m48_ms_sr_smoke.py \
  tests/unit/test_m48_ms_sr.py

echo "=== Full-vs-targeted topology and gradient parity tests (no MS_SR data) ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q tests/unit/test_m48_ms_sr.py

echo "=== MS_SR published CEDAR pair, projected FM source, core gradients ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.m48_ms_sr_smoke \
  --ms-sr-root "$ROOT" \
  --output-dir "$OUT" \
  --case-prefix "$CASE" --filter-id "$FILTER" \
  --depth "$DEPTH" --core-side "$CORE" \
  --width "$WIDTH" --heads "$HEADS" \
  --device cuda

test -s "$OUT/summary.json"
echo "PASS: M48 MS_SR read-only operator and static/dynamic one-step qualification."
echo "$OUT"
