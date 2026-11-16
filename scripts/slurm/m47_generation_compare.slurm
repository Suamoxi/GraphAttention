#!/bin/bash
#SBATCH --job-name=m47_gen_cmp
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# Only after ALL three full ensemble Slurm-array tasks pass.
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
OUT_ROOT=${M47_GEN_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_generated}
TAG=${M47_GEN_BASENAME:-ensemble_s42_h50_n8_seed20261116}
cd "$REPO"
test -f "$IMAGE"
test -x "$PYTHON"
test -z "$(git status --porcelain)" || { echo 'Dirty code checkout' >&2; exit 2; }
for mode in none mach mach_re; do
    test -f "$OUT_ROOT/$mode/$TAG/summary.json" || {
        echo "Missing completed full ensemble: $mode/$TAG" >&2; exit 2;
    }
done
if [ -f "$ENV_ROOT/bin/activate" ]; then source "$ENV_ROOT/bin/activate"; fi
singularity exec -B /home:/home -B /scratch:/scratch "$IMAGE" "$PYTHON" \
    -m pytest -q tests/unit/test_m47_generation.py
singularity exec -B /home:/home -B /scratch:/scratch "$IMAGE" "$PYTHON" \
    -u -m scripts.m47_generation_compare --root "$OUT_ROOT" --basename "$TAG"
