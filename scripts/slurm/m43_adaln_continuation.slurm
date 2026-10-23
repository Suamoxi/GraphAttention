#!/bin/bash
#SBATCH --job-name=m43_adaln_cont
#SBATCH --time=02:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Multi-optimizer-update, six-physical-mesh M43 training CONTINUATION GATE.
# This is an isolated diagnostic, NOT a production --resume.
# Produces numerical diagnostics only. Never overwrites scientific artifacts.
set -euo pipefail

REPO=${M43_CONT_REPO:-/scratch/coop/theret/GraphAttention_m43_fastpath}
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
TRAIN_RUN=${M43_CONT_TRAIN_RUN:-/scratch/coop/theret/GraphAttention_runs/m43_b_multires/train_pilot_s42}
ROOT=${M43_CONT_DNS_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REFS=${M43_CONT_REFERENCE_DIR:-/scratch/coop/theret/HIT_DNS_COMP_case_references}
OUT=${M43_CONT_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m43_adaln_cont_${SLURM_JOB_ID}}
STAGE=${M43_CONT_STAGE:-tests}
STEPS=${M43_CONT_STEPS:-3}
VAL_PER_CASE=${M43_CONT_VALIDATION_PER_CASE:-1}

case "$STAGE" in
  tests|continue) ;;
  *) echo "M43_CONT_STAGE must be tests or continue" >&2; exit 2 ;;
esac
case "$STEPS" in
  2|3|4|5|6) ;;
  *) echo "M43_CONT_STEPS must be in [2,6]" >&2; exit 2 ;;
esac
case "$VAL_PER_CASE" in
  1|2|3) ;;
  *) echo "M43_CONT_VALIDATION_PER_CASE must be 1,2,3" >&2; exit 2 ;;
esac

test -d "$REPO" || { echo "Missing isolated experiment worktree: $REPO" >&2; exit 2; }
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing uncommitted changes in isolated M43 continuation checkout" >&2
  git status --short >&2
  exit 2
fi

# Only the explicitly reviewed adaLN production edit and additive diagnostic
# files are allowed. No scientific M43-B training runner edits are allowed.
BASE=9440788253dc61f359a94ba3c68f50d6a7a6cb7d
git merge-base --is-ancestor "$BASE" HEAD || {
  echo "Source revision does not descend from known M43-B-compatible base" >&2
  exit 2
}
CHANGES=$(git diff --name-only "$BASE..HEAD")
if printf '%s\n' "$CHANGES" | grep -Ev \
  '^(src/graph_attention/models/dit_common.py|tests/unit/test_m43_adaln_broadcast.py|tests/unit/test_m43_adaln_continuation.py|scripts/benchmark_m43_adaln_broadcast.py|scripts/validate_m43_adaln_continuation.py|scripts/slurm/m43_adaln_broadcast.slurm|scripts/slurm/m43_adaln_continuation.slurm|docs/M43_ADALN_FASTPATH.md|docs/M43_ADALN_CONTINUATION.md)$'; then
  echo "Unexpected changes since M43-B-compatible base; refusing scientific claim" >&2
  exit 2
fi
for f in last.pt best.pt standardizers.pt run_signature.json history.json history.csv split.json; do
  test -s "$TRAIN_RUN/$f" || { echo "Missing original source file: $TRAIN_RUN/$f" >&2; exit 2; }
done
test -d "$ROOT"
test -d "$REFS"
test -f "$IMAGE"
if [ -e "$OUT" ]; then
  echo "Refusing collision with any old diagnostic output directory: $OUT" >&2
  exit 2
fi

export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"
echo "===== M43 isolated continuation comparison ====="
echo "Stage=$STAGE Git=$(git rev-parse HEAD) host=$(hostname)"
echo "Source checkpoint READ-ONLY: $TRAIN_RUN/last.pt"
echo "Diagnostic output ONLY: $OUT"
echo "Six original physical cases/update; $STEPS steps/arm; four independently restored arms"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "===== Unit and small CUDA numerical checks ====="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q \
    tests/unit/test_m43_adaln_broadcast.py \
    tests/unit/test_m43_adaln_continuation.py \
    tests/unit/test_dinat_dit.py::test_dinat_dit_alternates_local_and_exact_two_hop_topologies

if [ "$STAGE" = tests ]; then
  echo "Tests passed; no M43-B continuation steps were executed."
  exit 0
fi

echo "===== Two replays/mode from original checkpoint, no scientific writes ====="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.validate_m43_adaln_continuation \
    --root "$ROOT" \
    --reference-dir "$REFS" \
    --train-run-dir "$TRAIN_RUN" \
    --output-dir "$OUT" \
    --steps "$STEPS" \
    --validation-per-case "$VAL_PER_CASE"

test -s "$OUT/continuation_summary.json"
echo "Continuation diagnostic complete: $OUT/continuation_summary.json"
echo "No scientific training directory was changed or resumed."
