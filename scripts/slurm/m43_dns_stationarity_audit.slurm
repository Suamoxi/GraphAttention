#!/bin/bash
#SBATCH --job-name=m43_dns_audit
#SBATCH --time=12:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Read-only DNS audit; GPU is reserved for Grace container compatibility
# but is NOT used by the CPU/FFT implementation.
# Default is metadata-only; full six-case processing needs explicit opt-in.
set -euo pipefail

REPO=${M43_AUDIT_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

MODE=${M43_AUDIT_MODE:-preflight}
CASE=${M43_AUDIT_CASE:-Re94/Ma0.2}
ROOT=${M43_AUDIT_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
OUT=${M43_AUDIT_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m43_dns_stationarity_audit}
CONFIRM=${M43_AUDIT_CONFIRM_FULL:-NO}
SAMPLES=${M43_AUDIT_MAX_SNAPSHOTS:-0}
CUTOFFS=${M43_AUDIT_CUTOFFS:-}

case "$MODE" in
    preflight|case|compare|all) ;;
    *) echo "Invalid M43_AUDIT_MODE=$MODE" >&2; exit 2 ;;
esac
if [ "$MODE" = all ] && [ "$CONFIRM" != YES ]; then
    echo "Six-case analysis requires M43_AUDIT_CONFIRM_FULL=YES" >&2
    exit 2
fi
if [ "$SAMPLES" -lt 0 ]; then
    echo "M43_AUDIT_MAX_SNAPSHOTS must be >=0" >&2
    exit 2
fi
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "Refusing dirty repository checkout" >&2
    exit 2
fi

echo "M43 DNS read-only audit | revision $(git rev-parse HEAD)"
echo "mode=$MODE case=$CASE root=$ROOT output=$OUT"
echo "No training checkpoints, splits, or physical snapshots are modified"

echo "== Synthetic correctness tests =="
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q tests/unit/test_m43_dns_stationarity.py

ARGS=(--mode "$MODE" --root "$ROOT" --output-dir "$OUT")
if [ "$MODE" = case ] || [ "$MODE" = compare ]; then
    ARGS+=(--case "$CASE")
fi
if [ "$SAMPLES" -gt 0 ]; then
    ARGS+=(--max-snapshots "$SAMPLES")
fi
if [ -n "$CUTOFFS" ]; then
    test -s "$CUTOFFS" || { echo "Missing reviewed cutoff file $CUTOFFS" >&2; exit 2; }
    ARGS+=(--cutoffs-json "$CUTOFFS")
fi
echo "== Run analysis =="
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.audit_m43_dns_stationarity "${ARGS[@]}"

echo "Completed audit mode $MODE; reports under $OUT"
