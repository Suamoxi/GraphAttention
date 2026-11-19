#!/bin/bash
#SBATCH --job-name=m48e_a2_fulltile
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=96G
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail

# M48.4e A2: matched global-loss coverage, full vs tiled exact graph.
# Synthetic tensors only; no CEDAR, source DNS fields, model checkpoints or training.
REPO=/scratch/coop/theret/GraphAttention
ROOT="${M48_MS_SR_ROOT:-/scratch/ai/MS_SR}"
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON="${M48_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}"
NORMALIZATION="$REPO/configs/m48/teacher_44_w1p25_l1_q4_q8.json"
MODE="${M48_MODE:-}"
WIDTH="${M48_A2_WIDTH:-64}"
HEADS="${M48_A2_HEADS:-4}"
DEPTHS="${M48_A2_DEPTHS:-2,4,8}"
SIDES="${M48_A2_FINE_SIDES:-48}"
CORES="${M48_A2_CORE_SIDES:-24}"
MAX_TILES="${M48_A2_MAX_TILES:-8}"
WARMUP="${M48_A2_WARMUP:-1}"
MEASURE="${M48_A2_MEASURE:-2}"
TIMEOUT="${M48_A2_WORKER_TIMEOUT:-1800}"
FRACTION="${M48_A2_GPU_FRACTION:-0.85}"
HOST_GIB="${M48_A2_HOST_GIB:-12}"
OUT_ROOT="${M48_A2_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m48_4e_a2_full_vs_tiled}"
OUT="$OUT_ROOT/job_${SLURM_JOB_ID}_${MODE}_n${SIDES}_p${CORES}"
trap 'rc=$?; echo "M48.4e A2 failed: rc=$rc line=$LINENO command=$BASH_COMMAND" >&2' ERR

case "$MODE" in static|dynamic) ;; *) echo 'Set M48_MODE=static or dynamic' >&2; exit 2 ;; esac
cd "$REPO"
test -r "$ROOT/dataset/manifest.json"
test -r "$NORMALIZATION"
test -s "$IMAGE"
test -x "$PYTHON"
test ! -e "$OUT" || { echo "Refusing existing output directory: $OUT" >&2; exit 2; }
export PYTHONPATH="$REPO/src:$REPO:$ROOT/code:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export OMP_NUM_THREADS=8
export SINGULARITYENV_OMP_NUM_THREADS=8

printf '%s\n' '=== A2: matched-coverage full field versus exact-halo patch training kernel ==='
echo "Commit: $(git rev-parse HEAD)"
echo "mode=$MODE fine_sides=$SIDES cores=$CORES depths=$DEPTHS width=$WIDTH heads=$HEADS"
echo "GPU headroom fraction=$FRACTION max full-coverage tiles=$MAX_TILES"
echo "Synthetic-only. No CEDAR projection, no DNS payloads, no scientific training."
nvidia-smi --query-gpu=name,memory.total --format=csv

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m py_compile scripts/m48_4e_a2_full_vs_tiled.py \
    tests/unit/test_m48_4e_a2_full_vs_tiled.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q tests/unit/test_m48_4e_a2_full_vs_tiled.py \
    tests/unit/test_m48_ms_sr.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m scripts.m48_4e_a2_full_vs_tiled --metadata-only --mode "$MODE" \
    --ms-sr-root "$ROOT" --normalization-file "$NORMALIZATION"

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.m48_4e_a2_full_vs_tiled \
    --mode "$MODE" --ms-sr-root "$ROOT" --normalization-file "$NORMALIZATION" \
    --output-dir "$OUT" --fine-sides "$SIDES" --core-sides "$CORES" \
    --depths "$DEPTHS" --width "$WIDTH" --heads "$HEADS" \
    --warmup "$WARMUP" --measure "$MEASURE" \
    --max-tiles "$MAX_TILES" --max-gpu-fraction "$FRACTION" \
    --max-host-graph-gib "$HOST_GIB" --worker-timeout-seconds "$TIMEOUT"

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" - "$OUT/summary.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    r = json.load(f)
assert r['schema'] == 'graph_attention.m48_4e_a2_full_vs_tiled.v1'
assert r['provenance']['teacher_snapshots'] == 44
assert r['no_scientific_training'] and r['no_cedar_or_dns_field_evaluation']
assert len(r['trials']) == 2 * len(r['comparisons'])
assert r['qualified_pairs'] >= 1, 'no comparable exact-coverage CUDA strategies'
assert all(c['status'] in ('PASS','UNQUALIFIED') for c in r['comparisons']), \
    'numerical mismatch or coverage failure: inspect summary.json'
print('A2 result:', r['status'], 'qualified pairs:', r['qualified_pairs'])
PY
for f in summary.json trials.csv comparisons.csv; do test -s "$OUT/$f"; done
echo "A2 results: $OUT"
