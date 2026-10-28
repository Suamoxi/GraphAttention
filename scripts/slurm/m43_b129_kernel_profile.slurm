#!/bin/bash
#SBATCH --job-name=m43_b129_cuda_profile
#SBATCH --time=02:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Standalone measurement, NOT a training job. No write into the M43-B run.
# Default is metadata inspection; CUDA profiling requires explicit opt-in.
set -euo pipefail

REPO=${M43_PROFILE_REPO:-/scratch/coop/theret/GraphAttention_m43_profile}
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

ROOT=${M43_PROFILE_DATA_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REFS=${M43_PROFILE_REFS:-/scratch/coop/theret/HIT_DNS_COMP_case_references}
TRAIN_RUN=${M43_PROFILE_TRAIN_RUN:-/scratch/coop/theret/GraphAttention_runs/m43_b_multires/train_pilot_s42}
OUT=${M43_PROFILE_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m43_b129_cuda_profile_${SLURM_JOB_ID}}
MODE=${M43_PROFILE_MODE:-inspect}
CASE=${M43_PROFILE_CASE:-Re94/Ma0.2}
SNAPSHOT=${M43_PROFILE_SNAPSHOT:-51}
CONFIRM=${M43_PROFILE_CONFIRM:-NO}
TRACE=${M43_PROFILE_SAVE_CHROME_TRACE:-YES}

if [ "$MODE" != inspect ] && [ "$MODE" != profile ]; then
    echo "M43_PROFILE_MODE must be inspect or profile" >&2
    exit 2
fi
if [ "$MODE" = profile ] && [ "$CONFIRM" != YES ]; then
    echo "Active CUDA profiling requires M43_PROFILE_CONFIRM=YES" >&2
    exit 2
fi
if [ "$TRACE" != YES ] && [ "$TRACE" != NO ]; then
    echo "M43_PROFILE_SAVE_CHROME_TRACE must be YES/NO" >&2
    exit 2
fi
test -d "$REPO" || { echo "Missing profiler worktree $REPO" >&2; exit 2; }
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "Refusing dirty profiler source checkout" >&2
    git status --short >&2
    exit 2
fi

# Reject accidental model/runner edits on this profiling-only branch.
BASE=9440788253dc61f359a94ba3c68f50d6a7a6cb7d
git merge-base --is-ancestor "$BASE" HEAD || {
    echo "Profiler must descend from the exact known M43-B training commit" >&2
    exit 2
}
if git diff --name-only "$BASE..HEAD" | grep -Ev \
 '^(scripts/profile_m43_b_129_kernels.py|src/graph_attention/training/m43_kernel_profile_report.py|scripts/slurm/m43_b129_kernel_profile.slurm|tests/unit/test_m43_b_kernel_profile.py|docs/M43_B129_KERNEL_PROFILE.md)$' ; then
    echo "Nonprofiling source files changed: cannot claim unchanged M37" >&2
    exit 2
fi

test -s "$TRAIN_RUN/run_signature.json"
test -s "$TRAIN_RUN/last.pt"
test -s "$TRAIN_RUN/standardizers.pt"
test -d "$ROOT"
test -d "$REFS"
mkdir -p /scratch/coop/theret/GraphAttention

export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"
echo "==== M43 unchanged 129^3 kernel profiling ===="
echo "Host $(hostname), Git $(git rev-parse HEAD), Job ${SLURM_JOB_ID:-unknown}"
echo "Mode $MODE, case $CASE, snapshot $SNAPSHOT, profile output $OUT"
echo "Source M43-B last.pt $TRAIN_RUN/last.pt (read-only)"
echo "No original scientific checkpoint, optimizer or signatures are written"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "==== CPU unit correctness gate ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q tests/unit/test_m43_b_kernel_profile.py

ARGS=(
  --mode "$MODE"
  --root "$ROOT"
  --reference-dir "$REFS"
  --train-run-dir "$TRAIN_RUN"
  --output-dir "$OUT"
  --case-id "$CASE"
  --snapshot-number "$SNAPSHOT"
)
if [ "$TRACE" = NO ]; then ARGS+=(--skip-trace); fi

echo "==== Inspect/profile (only if explicitly requested) ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.profile_m43_b_129_kernels "${ARGS[@]}"

echo "Done. Training resume has NOT been started."
