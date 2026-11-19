#!/bin/bash
#SBATCH --job-name=m48e_a1_halos
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail

# CPU-only A1: depth x patch size x periodic origin dependency closure.
# No CUDA allocation, CFD field reading, CEDAR projection or training.
REPO=/scratch/coop/theret/GraphAttention
ROOT="${M48_MS_SR_ROOT:-/scratch/ai/MS_SR}"
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON="${M48_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}"
MODE="${M48_MODE:-}"
DEPTHS="${M48_A1_DEPTHS:-2,4,6,8}"
CORES="${M48_A1_CORE_SIDES:-12,16,20,24}"
SIDES="${M48_A1_FINE_SIDES:-48,80}"
ORIGINS="${M48_A1_ORIGINS:-seam,interior}"
TIMEOUT="${M48_A1_WORKER_TIMEOUT:-300}"
VIRTUAL_GIB="${M48_A1_WORKER_VIRTUAL_GIB:-48}"
OUT_ROOT="${M48_A1_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m48_4e_a1_halo_geometry}"
OUT="$OUT_ROOT/job_${SLURM_JOB_ID}_${MODE}"
NORMALIZATION="$REPO/configs/m48/teacher_44_w1p25_l1_q4_q8.json"
trap 'rc=$?; echo "M48.4e A1 FAILED: exit=$rc line=$LINENO command=$BASH_COMMAND" >&2' ERR

case "$MODE" in static|dynamic) ;; *)
    echo "Set M48_MODE=static or M48_MODE=dynamic" >&2; exit 2 ;;
esac
cd "$REPO"
test -r "$ROOT/dataset/manifest.json"
test -r "$NORMALIZATION"
test -s "$IMAGE"
test -x "$PYTHON"
test ! -e "$OUT" || { echo "Refusing existing output directory: $OUT" >&2; exit 2; }

export PYTHONPATH="$REPO/src:$REPO:$ROOT/code:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export OMP_NUM_THREADS=4
export SINGULARITYENV_OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1

printf 'M48.4e A1 CPU geometry audit | commit=%s | mode=%s\n' "$(git rev-parse HEAD)" "$MODE"
printf 'grids=%s depths=%s cores=%s origins=%s\n' "$SIDES" "$DEPTHS" "$CORES" "$ORIGINS"
printf 'worker_timeout=%ss virtual_memory_cap=%sGiB | GPU=NONE\n' "$TIMEOUT" "$VIRTUAL_GIB"

singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    scripts/m48_4e_gpu_capacity.py scripts/m48_4e_a1_halo_geometry.py \
    tests/unit/test_m48_4e_gpu_capacity.py tests/unit/test_m48_4e_a1_halo_geometry.py
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m48_4e_gpu_capacity.py \
    tests/unit/test_m48_4e_a1_halo_geometry.py \
    tests/unit/test_m48_ms_sr.py

singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m48_4e_a1_halo_geometry \
    --ms-sr-root "$ROOT" --normalization-file "$NORMALIZATION" \
    --output-dir "$OUT" --mode "$MODE" \
    --fine-sides "$SIDES" --depths "$DEPTHS" \
    --core-sides "$CORES" --origins "$ORIGINS" \
    --worker-timeout-seconds "$TIMEOUT" \
    --max-worker-virtual-gib "$VIRTUAL_GIB"

test -s "$OUT/summary.json"
test -s "$OUT/trials.csv"
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" - "$OUT/summary.json" "$MODE" <<'PY'
import json, sys
j=json.load(open(sys.argv[1]))
assert j['schema']=='graph_attention.m48_4e_a1_halo_geometry.v1'
assert j['mode']==sys.argv[2] and j['no_gpu_or_physical_fields']
assert j['provenance']['teacher_snapshots']==44
assert not j['provenance']['teacher_field_payloads_read']
assert j['status']=='PASS', (j['status'],j['failed_trials'])
assert j['successful_trials']==j['requested_trials']==len(j['trials'])
assert len(j['aggregates'])*len(j['requested']['origins'])==j['requested_trials']
print('PASS: M48.4e A1 source-authenticated CPU-only exact-halo geometry sweep')
print('Configurations:',len(j['aggregates']),'Trials:',len(j['trials']))
PY
printf 'A1 results: %s\n' "$OUT"
