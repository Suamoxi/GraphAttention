#!/bin/bash
#SBATCH --job-name=m47_10_gpu
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err
set -euo pipefail
REPO=${M47_10_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_10_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M47_10_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
DATA=${M47_10_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REF_ROOT=${M47_10_REFERENCE_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions}
REFS=${M47_10_REFS:-$REF_ROOT/case_references}
REF_AUDIT=${M47_10_REF_AUDIT:-$REF_ROOT/reference_audit.json}
M46_ROOT=${M47_10_M46_ROOT:-/scratch/coop/theret/GraphAttention_runs/m46_hit_similarity}
QUALIFIED=${M47_10_QUALIFIED:-$M46_ROOT/qualified_manifest.json}
INVENTORY=${M47_10_INVENTORY:-$M46_ROOT/inventory.json}
POLICY=${M47_10_POLICY:-$REPO/configs/m46_stationary_selection_reviewed_legacy.json}
ROOT=${M47_10_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_10_relative_periodic}
for path in "$REPO" "$IMAGE" "$PYTHON" "$DATA" "$REFS" "$REF_AUDIT" "$QUALIFIED" "$INVENTORY" "$POLICY"; do
    test -e "$path" || { echo "Missing required input: $path" >&2; exit 2; }
done
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Refusing dirty checkout" >&2; exit 2; }
mkdir -p "$ROOT/preflight"
exec 9>"$ROOT/preflight/gpu.lock"
flock -n 9 || { echo "Concurrent M47.10 GPU preflight" >&2; exit 2; }
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
echo "[M47.10 gpu] job=$SLURM_JOB_ID git=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# Parity tests (periodic equivariance and Triton/reference gradient comparisons)
# are being diagnosed separately and must not block this feasibility gate.
# Keep a lightweight argument-contract test plus the real 128^3 optimizer step.
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
    tests/unit/test_m47_10_periodic_forward.py
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u -m scripts.m47_10_preflight \
    --stage gpu --repo "$REPO" --archive-root "$DATA" --reference-dir "$REFS" \
    --reference-audit "$REF_AUDIT" --qualified-manifest "$QUALIFIED" \
    --source-inventory "$INVENTORY" --selection-policy "$POLICY" \
    --output-root "$ROOT"
test -s "$ROOT/preflight/gpu.json"
test -s "$ROOT/preflight/ready.json"
echo "[M47.10 gpu] PASSED; A1 A2 A3 training authorized"
