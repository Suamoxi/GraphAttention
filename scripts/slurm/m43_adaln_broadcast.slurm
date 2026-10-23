#!/bin/bash
#SBATCH --job-name=m43_adaln_fast
#SBATCH --time=02:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Read-only comparison against original eight-epoch checkpoint. The only
# scientific source delta must be the shared graph-conditioning broadcast.
set -euo pipefail

REPO=${M43_FAST_REPO:-/scratch/coop/theret/GraphAttention_m43_fastpath}
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
TRAIN_RUN=${M43_FAST_TRAIN_RUN:-/scratch/coop/theret/GraphAttention_runs/m43_b_multires/train_pilot_s42}
ROOT=${M43_FAST_DNS_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REFS=${M43_FAST_REFERENCE_DIR:-/scratch/coop/theret/HIT_DNS_COMP_case_references}
OUT=${M43_FAST_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m43_adaln_fast_${SLURM_JOB_ID}}
STAGE=${M43_FAST_STAGE:-tests}
TRIALS=${M43_FAST_TRIALS:-3}
PROFILE=${M43_FAST_PROFILE_KERNELS:-YES}

case "$STAGE" in
  tests|benchmark) ;;
  *) echo "M43_FAST_STAGE must be tests or benchmark" >&2; exit 2 ;;
esac
case "$PROFILE" in
  YES|NO) ;;
  *) echo "M43_FAST_PROFILE_KERNELS must be YES/NO" >&2; exit 2 ;;
esac
test -d "$REPO" || { echo "Missing source worktree: $REPO" >&2; exit 2; }
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing uncommitted changes in the isolated optimization worktree" >&2
  git status --short >&2
  exit 2
fi

BASE=9440788253dc61f359a94ba3c68f50d6a7a6cb7d
git merge-base --is-ancestor "$BASE" HEAD || {
  echo "Source is not descended from known eight-epoch compatible code" >&2
  exit 2
}
CHANGES=$(git diff --name-only "$BASE..HEAD")
if printf '%s\n' "$CHANGES" | grep -Ev \
  '^(src/graph_attention/models/dit_common.py|tests/unit/test_m43_adaln_broadcast.py|tests/unit/test_m43_adaln_continuation.py|scripts/benchmark_m43_adaln_broadcast.py|scripts/validate_m43_adaln_continuation.py|scripts/slurm/m43_adaln_broadcast.slurm|scripts/slurm/m43_adaln_continuation.slurm|docs/M43_ADALN_FASTPATH.md|docs/M43_ADALN_CONTINUATION.md); then
  echo "Non-optimization source changes detected; refusing provenance claim" >&2
  exit 2
fi
test -s "$TRAIN_RUN/run_signature.json"
test -s "$TRAIN_RUN/last.pt"
test -s "$TRAIN_RUN/standardizers.pt"
test -d "$ROOT"
test -d "$REFS"

export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"
echo "===== M43 adaLN broadcast optimization ====="
echo "Stage=$STAGE, Git=$(git rev-parse HEAD), host=$(hostname)"
echo "Read-only source checkpoint: $TRAIN_RUN/last.pt"
echo "Separate output: $OUT"
echo "No scientific training job launched"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "===== Reference parity: heterogeneous meshes/times and checkpointing ====="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q \
    tests/unit/test_m43_adaln_broadcast.py \
    tests/unit/test_dinat_dit.py::test_dinat_dit_alternates_local_and_exact_two_hop_topologies

if [ "$STAGE" = tests ]; then
  echo "Test-only gate completed. GH200 comparison not started."
  exit 0
fi
case "$TRIALS" in
  1|2|3|4|5|6|7|8) ;;
  *) echo "M43_FAST_TRIALS must be an integer in [1,8]" >&2; exit 2 ;;
esac

ARGS=(
  --root "$ROOT"
  --reference-dir "$REFS"
  --train-run-dir "$TRAIN_RUN"
  --output-dir "$OUT"
  --trials "$TRIALS"
  --case-id "Re94/Ma0.2"
  --snapshot-number 51
)
if [ "$PROFILE" = NO ]; then
  ARGS+=(--no-profile-kernels)
fi
echo "===== GH200 paired full 129^3 checkpoint benchmark ====="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.benchmark_m43_adaln_broadcast "${ARGS[@]}"
test -s "$OUT/benchmark_summary.json"
echo "Results: $OUT/benchmark_summary.json"
echo "No checkpoint/split/scaler/history was written."
; then
  echo "Non-optimization source changes detected; refusing provenance claim" >&2
  exit 2
fi
test -s "$TRAIN_RUN/run_signature.json"
test -s "$TRAIN_RUN/last.pt"
test -s "$TRAIN_RUN/standardizers.pt"
test -d "$ROOT"
test -d "$REFS"

export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"
echo "===== M43 adaLN broadcast optimization ====="
echo "Stage=$STAGE, Git=$(git rev-parse HEAD), host=$(hostname)"
echo "Read-only source checkpoint: $TRAIN_RUN/last.pt"
echo "Separate output: $OUT"
echo "No scientific training job launched"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "===== Reference parity: heterogeneous meshes/times and checkpointing ====="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q \
    tests/unit/test_m43_adaln_broadcast.py \
    tests/unit/test_dinat_dit.py::test_dinat_dit_alternates_local_and_exact_two_hop_topologies

if [ "$STAGE" = tests ]; then
  echo "Test-only gate completed. GH200 comparison not started."
  exit 0
fi
case "$TRIALS" in
  1|2|3|4|5|6|7|8) ;;
  *) echo "M43_FAST_TRIALS must be an integer in [1,8]" >&2; exit 2 ;;
esac

ARGS=(
  --root "$ROOT"
  --reference-dir "$REFS"
  --train-run-dir "$TRAIN_RUN"
  --output-dir "$OUT"
  --trials "$TRIALS"
  --case-id "Re94/Ma0.2"
  --snapshot-number 51
)
if [ "$PROFILE" = NO ]; then
  ARGS+=(--no-profile-kernels)
fi
echo "===== GH200 paired full 129^3 checkpoint benchmark ====="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.benchmark_m43_adaln_broadcast "${ARGS[@]}"
test -s "$OUT/benchmark_summary.json"
echo "Results: $OUT/benchmark_summary.json"
echo "No checkpoint/split/scaler/history was written."
