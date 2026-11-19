#!/bin/bash
#SBATCH --job-name=m48e_capacity
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail

# M48.4e Phase A: real TwoScaleDiNAT FP32 GPU-only forward/backward/AdamW.
# Synthetic normalized tensor features, authenticated teacher grid metadata;
# no actual DNS fields, CEDAR calls, training or validation data are accessed.
REPO=/scratch/coop/theret/GraphAttention
ROOT="${M48_MS_SR_ROOT:-/scratch/ai/MS_SR}"
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON="${M48_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}"
MODE="${M48_MODE:-}"
WIDTH="${M48_WIDTH:-64}"
DEPTH="${M48_DEPTH:-2}"
HEADS="${M48_HEADS:-4}"
FINE_SIDES="${M48_FINE_SIDES:-48,80}"
CORES="${M48_CORE_SIDES:-12,16,20,24}"
BATCHES="${M48_GPU_BATCHES:-1,2,4,8,16,32,64,128,256}"
WARMUP="${M48_WARMUP:-3}"
MEASURE="${M48_MEASURE:-8}"
HEADROOM="${M48_GPU_FRACTION:-0.85}"
HOST_GIB="${M48_HOST_PACK_GIB:-8}"
WORKER_TIMEOUT="${M48_WORKER_TIMEOUT:-300}"
OUT_ROOT="${M48_CAPACITY_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m48_4e_gpu_capacity}"
OUT="$OUT_ROOT/job_${SLURM_JOB_ID}_${MODE}_w${WIDTH}_d${DEPTH}"
NORMALIZATION="$REPO/configs/m48/teacher_44_w1p25_l1_q4_q8.json"
trap 'rc=$?; echo "M48.4e failure: exit=$rc line=$LINENO command=$BASH_COMMAND" >&2' ERR

case "$MODE" in static|dynamic) ;; *)
    echo "Set M48_MODE=static or dynamic" >&2; exit 2 ;;
esac
cd "$REPO"
test ! -e "$OUT"
test -r "$ROOT/dataset/manifest.json"
test -r "$NORMALIZATION"
test -s "$IMAGE"
test -x "$PYTHON"
export PYTHONPATH="$REPO/src:$REPO:$ROOT/code:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export OMP_NUM_THREADS=8
export SINGULARITYENV_OMP_NUM_THREADS=8

echo "=== M48.4e Phase A GPU-only exact-halo capacity qualification ==="
echo "Commit: $(git rev-parse HEAD)"
echo "Architecture=$MODE width=$WIDTH depth=$DEPTH heads=$HEADS"
echo "Fine sides=$FINE_SIDES core sides=$CORES GPU batches=$BATCHES"
echo "Max reserved fraction=$HEADROOM; NO scientific training"
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    scripts/m48_4e_gpu_capacity.py tests/unit/test_m48_4e_gpu_capacity.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m48_4e_gpu_capacity.py tests/unit/test_m48_ms_sr.py

echo "=== Fail-closed 44-teacher metadata preflight (no CUDA trial) ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m scripts.m48_4e_gpu_capacity \
    --ms-sr-root "$ROOT" --normalization-file "$NORMALIZATION" \
    --mode "$MODE" --metadata-only

echo "=== Isolated GPU capacity sweep ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m48_4e_gpu_capacity \
    --ms-sr-root "$ROOT" \
    --normalization-file "$NORMALIZATION" \
    --output-dir "$OUT" --mode "$MODE" \
    --fine-sides "$FINE_SIDES" --core-sides "$CORES" \
    --microbatches "$BATCHES" --width "$WIDTH" --depth "$DEPTH" \
    --heads "$HEADS" --warmup "$WARMUP" --measure "$MEASURE" \
    --max-gpu-fraction "$HEADROOM" --max-host-pack-gib "$HOST_GIB" \
    --worker-timeout-seconds "$WORKER_TIMEOUT"

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" - "$OUT/summary.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
assert p['schema']=='graph_attention.m48_4e_gpu_capacity.v1'
assert p['no_training_performed'] and p['synthetic_feature_inputs']
assert p['provenance']['teacher_snapshots']==44
assert p['status'] in ('FINISHED','INCOMPLETE'), p['status']
assert any(r.get('status')=='ok' for r in p['trials']), 'no qualified CUDA batch'
print('PASS: M48.4e GPU-only capacity receipt; status=',p['status'])
print('Qualified grid-core configurations=',len(p['best_safe_per_grid_core']))
PY
test -s "$OUT/trials.csv"
echo "M48.4e result directory: $OUT"
