#!/bin/bash
#SBATCH --job-name=m49_les_archive
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# CPU/I/O metadata audit on Grace because the archive is mounted there.
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
ARCHIVE=${M49_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_LES_FORCED}
SCRATCH=${M49_SCRATCH_HIT_ROOT:-/scratch/coop/theret/HIT_LES_FORCED}
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout"; exit 2; }
test -d "$ARCHIVE" && test -d "$SCRATCH" || { echo "Missing archive or scratch HIT root"; exit 2; }
OUT="$ROOT/archive_audit/$SLURM_JOB_ID"
mkdir -p "$OUT"
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
run() { singularity exec "${BIND[@]}" --env "PYTHONPATH=$REPO/src:$REPO" "$IMAGE" "$PYTHON" "$@"; }
run -m pytest -q tests/unit/test_m49_hit_archive.py --junitxml="$OUT/tests.xml"
run -u -m scripts.audit_m49_hit_archive --archive "$ARCHIVE" --scratch "$SCRATCH" --output "$OUT/inventory.json"
test -s "$OUT/inventory.json"
echo "Inventory=$OUT/inventory.json"
