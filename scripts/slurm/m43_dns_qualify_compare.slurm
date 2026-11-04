#!/bin/bash
#SBATCH --job-name=m43_dns_qualification
#SBATCH --time=12:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# DNS audit postprocessor: never touches existing M43-B training checkpoints.
set -euo pipefail

REPO=${M43_Q_REPO:-/scratch/coop/theret/GraphAttention_dns_qualification}
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
MODE=${M43_Q_MODE:-qualify}
AUDIT=${M43_Q_AUDIT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m43_dns_stationarity_audit}
OUT=${M43_Q_OUTPUT:-${AUDIT}/m43_qualification_v1}
CUTOFFS=${M43_Q_REVIEWED_CUTOFFS:-}
REFERENCE_DIR=${M43_Q_REFERENCE_DIR:-/scratch/coop/theret/HIT_DNS_COMP_case_references}
NODES=${M43_Q_NODES_PER_SNAPSHOT:-512}

case "$MODE" in qualify|sample|compare) ;; *) echo "Unknown M43_Q_MODE: $MODE" >&2; exit 2;; esac
if [ "$MODE" != qualify ] && [ ! -s "$CUTOFFS" ]; then
  echo "Sampling/comparison requires explicit human-reviewed cutoff JSON" >&2
  exit 2
fi
if [ "$MODE" != qualify ] && [ ! -d "$REFERENCE_DIR" ]; then
  echo "Missing authoritative M3.3 per-case YAML reference directory" >&2
  exit 2
fi

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing dirty repository checkout" >&2
  exit 2
fi

echo "M43 qualification | $(git rev-parse HEAD) | mode=$MODE"
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q \
  tests/unit/test_m43_qualification.py tests/unit/test_m43_qualify_compare.py

ARGS=(--mode "$MODE" --audit-root "$AUDIT" --output-dir "$OUT")
if [ -n "$CUTOFFS" ]; then
  ARGS+=(--reviewed-cutoffs "$CUTOFFS")
fi
if [ "$MODE" != qualify ]; then
  ARGS+=(--reference-dir "$REFERENCE_DIR")
fi
if [ "$MODE" = sample ]; then
  ARGS+=(--nodes-per-snapshot "$NODES")
fi
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.m43_qualify_compare "${ARGS[@]}"
echo "M43 qualification completed; outputs: $OUT"
