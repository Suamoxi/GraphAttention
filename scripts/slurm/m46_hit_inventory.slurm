#!/bin/bash
#SBATCH --job-name=m46_inventory
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# M46 inventory: metadata-only, all snapshots, no stationarity selection.
# Submit from the GraphAttention repository with:
#   sbatch scripts/slurm/m46_hit_inventory.slurm
set -euo pipefail

REPO=${M46_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M46_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M46_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON="${M46_PYTHON:-$ENV_ROOT/bin/python3}"
ARCHIVE=${M46_ARCHIVE_ROOT:-}
if [ -z "$ARCHIVE" ]; then
    # /archive2 is visible from some login nodes but not from Grace nodes.
    for candidate in /archivenfs/phlusim/HIT_DNS_COMP /archive2/cfd-phlusim/HIT_DNS_COMP; do
        if [ -d "$candidate/Re54" ] && [ -d "$candidate/Re94" ] &&
           [ -d "$candidate/ReT060" ] && [ -d "$candidate/ReT100" ]; then
            ARCHIVE=$candidate
            break
        fi
    done
fi
if [ -z "$ARCHIVE" ] || [ ! -d "$ARCHIVE" ]; then
    echo "ERROR: neither complete archive root is visible from compute node $(hostname)." >&2
    echo "Tried /archivenfs/phlusim/HIT_DNS_COMP and /archive2/cfd-phlusim/HIT_DNS_COMP." >&2
    echo "Set M46_ARCHIVE_ROOT to the actual compute-node mount using sbatch --export." >&2
    for candidate in /archivenfs/phlusim/HIT_DNS_COMP /archive2/cfd-phlusim/HIT_DNS_COMP; do
        ls -ld "$candidate" >&2 || true
    done
    exit 2
fi
PILOT=${M46_PILOT_ROOT:-/scratch/ai/HIT_DNS_pilot}
OUT=${M46_OUTPUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m46_hit_similarity}

printf "M46 inventory job=%s host=%s archive=%s pilot=%s\n" "$SLURM_JOB_ID" "$(hostname)" "$ARCHIVE" "$PILOT"
for path in "$REPO" "$IMAGE" "$ARCHIVE" "$PILOT/data"; do
    test -e "$path" || { echo "Missing required path: $path" >&2; exit 2; }
done
test -x "$PYTHON" || { echo "Missing environment Python: $PYTHON" >&2; exit 2; }

# Use the pinned virtual environment executable even if activate is unavailable.
if [ -f "$ENV_ROOT/bin/activate" ]; then
    source "$ENV_ROOT/bin/activate"
fi
cd "$REPO"
mkdir -p "$OUT"
printf '%s\n' "$(git rev-parse HEAD)" > "$OUT/inventory_git_commit.txt"
git status --short > "$OUT/inventory_git_status.txt"
printf 'M46 inventory job=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname)"
printf 'archive=%s pilot=%s output=%s\n' "$ARCHIVE" "$PILOT" "$OUT"

BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
singularity exec "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q     tests/unit/test_m46_hit_similarity.py

singularity exec "${BIND[@]}" "$IMAGE" "$PYTHON" -u -m scripts.m46_hit_similarity inventory     --archive-root "$ARCHIVE"     --pilot-root "$PILOT"     --output "$OUT"

test -s "$OUT/inventory.json"
echo "M46 inventory complete: $OUT/inventory.json"
