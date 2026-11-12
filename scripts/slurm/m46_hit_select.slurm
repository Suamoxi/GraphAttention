#!/bin/bash
#SBATCH --job-name=m46_select
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# Read-only snapshot eligibility classification. No field data or model training.
# sbatch scripts/slurm/m46_hit_select.slurm
set -euo pipefail
REPO=${M46_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M46_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M46_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M46_PYTHON:-$ENV_ROOT/bin/python3}
BASE=${M46_OUTPUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m46_hit_similarity}
POLICY=${M46_SELECTION_POLICY:-$REPO/configs/m46_stationary_selection_reviewed_legacy.json}
for path in "$REPO" "$IMAGE" "$BASE/inventory.json" "$POLICY"; do
  test -e "$path" || { echo "Missing required path: $path" >&2; exit 2; }
done
test -x "$PYTHON" || { echo "Missing environment Python: $PYTHON" >&2; exit 2; }
if [ -f "$ENV_ROOT/bin/activate" ]; then source "$ENV_ROOT/bin/activate"; fi
cd "$REPO"
echo "M46 qualification job=$SLURM_JOB_ID host=$(hostname) git=$(git rev-parse HEAD)"
echo "Policy=$POLICY full inventory=$BASE/inventory.json"
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
  if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
singularity exec "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
  tests/unit/test_m46_stationarity_selection.py
singularity exec "${BIND[@]}" "$IMAGE" "$PYTHON" -u -m scripts.m46_stationarity_selection \
  --inventory "$BASE/inventory.json" \
  --policy "$POLICY" \
  --output "$BASE"
echo "M46 selection complete: $BASE/qualified_manifest.json"