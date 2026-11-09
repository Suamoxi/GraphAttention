#!/bin/bash
#SBATCH --job-name=m45_6_aggregate
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/m45_6_aggregate_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/m45_6_aggregate_%j.err
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
ROOT=/scratch/coop/theret/GraphAttention_runs/m45_6_depth_full_all
OUT_ROOT=$(printenv M45_6_AGG_OUT_ROOT || echo /scratch/coop/theret/GraphAttention_runs/m45_6_depth_comparison)
OUT="$OUT_ROOT/job_$SLURM_JOB_ID"

# Drive inventory reveals job_493421 re-ran seed=2028, samples=256, depth=2
# already covered by job_493366. Keep the earlier run as canonical.
# Set M45_6_AGG_EXCLUDE_JOB=none to restore strict duplicate rejection.
EXCLUDE=$(printenv M45_6_AGG_EXCLUDE_JOB || echo job_493421)

cd "$REPO"
[ -z "$(git status --porcelain)" ] || {
    echo 'Dirty checkout; refusing to aggregate mixed revisions' >&2
    exit 2
}
[ -s "$IMAGE" ] && [ -x "$PYTHON" ] || {
    echo 'Missing Singularity image or Python interpreter' >&2
    exit 2
}
[ -d "$ROOT" ] || { echo "Missing input directory: $ROOT" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "Refusing existing output: $OUT" >&2; exit 2; }

echo "M45.6 depth aggregation commit=$(git rev-parse HEAD)"
echo "Input=$ROOT output=$OUT excluded=$EXCLUDE"
echo "Input summaries:"
find "$ROOT" -maxdepth 2 -name summary.json -print | sort

singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile scripts/m45_6_compare_depth.py \
    tests/unit/test_m45_6_depth_aggregate.py

singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q tests/unit/test_m45_6_depth_aggregate.py

set -- --root "$ROOT" --output-dir "$OUT"
if [ "$EXCLUDE" != "none" ]; then
    set -- "$@" --exclude-job "$EXCLUDE"
fi
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_6_compare_depth "$@"

test -s "$OUT/summary.json"
test -s "$OUT/per_job.csv"
test -s "$OUT/matched_depth.csv"
test -s "$OUT/aggregates.csv"

# Complete matrix: 12 jobs * 2 models = 24 rows; 12 matched depth pairs.
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["n_runs"] == 24 and s["n_matched"] == 12, s' \
    "$OUT/summary.json"

echo "M45.6 depth aggregation complete: $OUT"
