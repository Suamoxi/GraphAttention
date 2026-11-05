#!/bin/bash
#SBATCH --job-name=m43_similarity
#SBATCH --time=12:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Uses the previously validated Grace/Singularity launcher. This is CPU
# postprocessing; the GPU reservation is incidental to the known partition.
# All DNS archives, M43 training artifacts and M3.3 reference YAMLs are read-only.
set -euo pipefail

REPO=${M43_S_REPO:-/scratch/coop/theret/GraphAttention_dns_similarity}
MODE=${M43_S_MODE:-inspect}
AUDIT=${M43_S_AUDIT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m43_dns_stationarity_audit}
QUAL=${M43_S_QUAL_ROOT:-${AUDIT}/m43_qualification_v1}
OUT=${M43_S_OUT:-${AUDIT}/m43_reference_similarity_v1}
REFS=${M43_S_REFERENCE_DIR:-/scratch/coop/theret/HIT_DNS_COMP_case_references}
STATE=${M43_S_STATE_DIR:-/scratch/coop/theret/HIT_DNS_COMP_similarity_references}
CUTOFFS=${M43_S_REVIEWED_CUTOFFS:-}
REPEATS=${M43_S_BOOTSTRAP_REPEATS:-64}
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

case "$MODE" in inspect|compare) ;; *) echo "Invalid M43_S_MODE=$MODE" >&2; exit 2;; esac
if [ "$MODE" = "compare" ]; then
  if [ ! -f "$CUTOFFS" ]; then
    echo "Required explicit reviewed cutoff JSON missing: $CUTOFFS" >&2; exit 2
  fi
fi
if [ ! -d "$REFS" ]; then
  echo "Missing authoritative M3.3 references: $REFS" >&2; exit 2
fi
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing to run against a dirty checkout" >&2
  exit 2
fi

echo "M43 reference similarity | revision=$(git rev-parse HEAD) | mode=$MODE"
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q \
  tests/unit/test_m43_similarity.py tests/unit/test_m43_similarity_diagnostics.py
ARGS=(--mode "$MODE" --audit-root "$AUDIT" --qualification-root "$QUAL" \
      --output-dir "$OUT" --reference-dir "$REFS" --state-reference-dir "$STATE")
if [ "$MODE" = "compare" ]; then
  ARGS+=(--reviewed-cutoffs "$CUTOFFS" --bootstrap-repeats "$REPEATS")
fi
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.m43_similarity_diagnostics "${ARGS[@]}"
echo "M43 similarity completed: $OUT"
