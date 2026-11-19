#!/bin/bash
#SBATCH --job-name=m48d_order_audit
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail

# Strictly read-only original AVBP and published MS_SR data audit.
# CPU-only: no GPU allocation, no optimizer, no source modifications.
REPO=/scratch/coop/theret/GraphAttention
PILOT="${M48_PILOT_ROOT:-/scratch/ai/HIT_DNS_pilot}"
ROOT="${M48_MS_SR_ROOT:-/scratch/ai/MS_SR}"
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON="${M48_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}"
OUT_ROOT="${M48_ORDER_AUDIT_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m48_4d_ordering_audit}"
OUT="$OUT_ROOT/job_${SLURM_JOB_ID}"
CEDAR_SRC="${M48_CEDAR_SRC:-}"
trap 'rc=$?; echo "M48.4d ordering audit FAILED: exit=$rc line=$LINENO command=$BASH_COMMAND" >&2' ERR

cd "$REPO"
test -r "$PILOT/VERIFIED.json"
test -d "$PILOT/data"
test -r "$ROOT/dataset/manifest.json"
test -r "$ROOT/code/preprocess.py"
test -r "$ROOT/code/campaign_data.py"
test -r "$ROOT/code/models.py"
test -s "$IMAGE"
test -x "$PYTHON"
test ! -e "$OUT" || { echo "Refusing existing output: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
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
export OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1

echo "=== M48.4d independent physical XYZ -> graph ordering qualification ==="
echo "Revision: $(git rev-parse HEAD)"
echo "Native pilot: $PILOT"
echo "Published MS_SR: $ROOT"
echo "Output: $OUT"
echo "GPU allocation: NONE (read-only CPU audit)"

singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -c 'from preprocess import check_cedar; check_cedar(); print("Pinned CEDAR OK")'
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    scripts/m48_4d_ordering_audit.py tests/unit/test_m48_4d_ordering_audit.py
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m48_4d_ordering_audit.py \
    tests/unit/test_m48_ms_sr.py

# 100% original-node coordinate checks, 100% original hex-cell checks,
# 100% conservative qf values, q2/q4/q8 full-domain regenerated fields;
# first eligible teacher source in each Reynolds group.
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m48_4d_ordering_audit \
    --pilot-root "$PILOT" \
    --ms-sr-root "$ROOT" \
    --output "$OUT/ordering_audit.json"

test -s "$OUT/ordering_audit.json"
test -s "$OUT/ordering_audit_ReT060.json"
test -s "$OUT/ordering_audit_ReT100.json"
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" - "$OUT/ordering_audit.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
assert report['status'] == 'PASS'
assert len(report['cases']) == 2
assert {x['case'].split('/')[0] for x in report['cases']} == {'ReT060', 'ReT100'}
for x in report['cases']:
    assert x['status'] == 'PASS'
    n = x['native_shape'][0]
    assert x['original_periodic_cell_connectivity']['all_native_hex_cells_verified'] == n**3
    assert x['original_periodic_cell_connectivity']['periodic_face_seam_cells_xyz'] == [n**2] * 3
    assert x['graphattention_q4_q8_flattening'] == 'bitwise_equal_all_nodes_all_components'
    assert x['file_hashes_reverified']
    assert len(x['field_reconstruction']) == 5
print('PASS: original AVBP ordering, all periodic cells, full source values, CEDAR and M48 loader')
PY

echo "PASS: M48.4d independent ordering audit"
echo "$OUT"
