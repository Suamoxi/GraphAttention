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
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
OUT_ROOT="${M48_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m48_ms_sr_smoke}"
OUT="$OUT_ROOT/job_${SLURM_JOB_ID}"
FILTER="${M48_FILTER_ID:-w1p25_l1}"
CASE="${M48_CASE_PREFIX:-ReT060/}"
DEPTH="${M48_DEPTH:-1}"
CORE="${M48_CORE_SIDE:-2}"
WIDTH="${M48_WIDTH:-16}"
HEADS="${M48_HEADS:-4}"

cd "$REPO"
test -x "$PYTHON"
test -s "$IMAGE"
test ! -e "$OUT"
test -f "$ROOT/dataset/manifest.json" || {
    echo "Missing MS_SR campaign manifest at $ROOT/dataset/manifest.json" >&2
    echo "MS_SR may be on another filesystem; do NOT copy or synthesize data silently." >&2
    exit 2
}
test -f "$ROOT/code/campaign_data.py"
test -f "$ROOT/code/models.py"
test -d "$ROOT/cedar/src"

echo "M48 real-MS_SR smoke; revision=$(git rev-parse HEAD); job=${SLURM_JOB_ID}"
echo "Data (read-only): $ROOT"
echo "Filter=$FILTER case_prefix=$CASE depth=$DEPTH core_side=$CORE width=$WIDTH"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Ensure project packages are importable even without an editable install.
export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"

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
