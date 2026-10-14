#!/bin/bash
#SBATCH --job-name=m41_4_scaling_prepare
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Uses established Grace resources; no explicit --mem cap.
# Repartition ONLY, do NOT rebuild the base one-hop CSR or run inference.
set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
RUN_ROOT=/scratch/coop/theret/GraphAttention_runs
BASE_PREP=${M41_4_BASE_PREP:-${RUN_ROOT}/m41_4_257_generic_graph}
SWEEP_ROOT=${M41_4_SWEEP_ROOT:-${RUN_ROOT}/m41_4_257_scaling}
LAYOUTS=${M41_4_SWEEP_LAYOUTS:-"16 32 128"}
cd "$REPO"
test -s "$BASE_PREP/prepare.json" || { echo "Missing base graph $BASE_PREP/prepare.json" >&2; exit 1; }
mkdir -p "$SWEEP_ROOT/layouts"
git rev-parse HEAD > "$SWEEP_ROOT/repartition_commit.txt"
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: checkout must be clean for traceability" >&2
    git status --short >&2
    exit 1
fi
echo "M41.4 generic CSR repartitioning, host $(hostname), commit $(git rev-parse HEAD)"
echo "Reusing CSR: $BASE_PREP/csr"

echo "==== regression tests BEFORE real 257-cube repartitioning ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest \
    tests/unit/test_m41_4_scaling.py \
    tests/unit/test_m41_4_generic_csr.py -q

for layout in $LAYOUTS; do
    case "$layout" in
        16|32|128) ;;
        64) echo "64-partition original already prepared: $BASE_PREP"; continue ;;
        *) echo "Unsupported layout $layout" >&2; exit 1 ;;
    esac
    target=$(printf "%s/layouts/layout_%03d" "$SWEEP_ROOT" "$layout")
    if [ -s "$target/prepare.json" ]; then
        echo "Already prepared; keeping immutable layout $layout: $target"
        continue
    fi
    if [ -e "$target" ]; then
        echo "ERROR: incomplete directory $target; inspect before retry" >&2
        exit 1
    fi
    echo "==== prepare generic $layout-partition graph ===="
    singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
        "$PYTHON" -u -m scripts.m41_4_repartition \
        --base-dir "$BASE_PREP" \
        --layout-root "$SWEEP_ROOT/layouts" \
        --layout "$layout"
    test -s "$target/prepare.json"
done
echo "All requested generic CSR partition layouts prepared, without rebuilding global one-hop edges"
