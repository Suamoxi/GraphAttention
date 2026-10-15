#!/bin/bash
#SBATCH --job-name=m41_6_cache_prepare
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Numerically validate M41.6 modes before compiling per-partition mmap cache.
# Exact physical graph, M37 weights and 32/64 ownership plans remain IMMUTABLE.
# Grace: 1 GPU, 4 CPUs, 4 hours, no explicit --mem constraint.
set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
RUN_ROOT=/scratch/coop/theret/GraphAttention_runs
BASE_PREP=${M41_6_BASE_PREP:-${RUN_ROOT}/m41_4_257_generic_graph}
SWEEP_ROOT=${M41_6_SWEEP_ROOT:-${RUN_ROOT}/m41_4_257_scaling}
CACHE_ROOT=${M41_6_CACHE_ROOT:-${RUN_ROOT}/m41_6_257_metadata_cache}
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: checkout must be clean" >&2
    git status --short >&2
    exit 1
fi
echo "==== M41.6 prepare on $(hostname), commit $(git rev-parse HEAD) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv
for source in "$BASE_PREP/prepare.json" "$SWEEP_ROOT/layouts/layout_032/prepare.json"; do
    test -s "$source" || { echo "Missing prepared 257^3 graph $source" >&2; exit 1; }
done

echo "==== Python syntax gate ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    src/graph_attention/models/owned_query_dinat.py \
    src/graph_attention/models/streamed_dinat.py \
    src/graph_attention/models/streamed_stage_profile.py \
    src/graph_attention/models/disk_streamed_dinat.py \
    src/graph_attention/models/selective_disk_dinat.py \
    scripts/m41_4_generic_large.py \
    scripts/m41_6_compile_metadata.py \
    scripts/benchmark_m41_6_metadata.py \
    tests/unit/test_m41_6_metadata.py

echo "==== Small irregular/unstructured exact output and cache regressions ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m41_6_metadata.py \
    tests/unit/test_m41_4_generic_csr.py \
    tests/unit/test_streamed_dinat.py \
    tests/unit/test_m41_5_paired_profile.py

mkdir -p "$CACHE_ROOT"
echo "Scratch cache root: $CACHE_ROOT"
echo "Existing disk capacity:"
df -h "$CACHE_ROOT"

for layout in 32 64; do
    if [ "$layout" = 32 ]; then
        PREPARED="$SWEEP_ROOT/layouts/layout_032"
    else
        PREPARED="$BASE_PREP"
    fi
    OUTPUT=$(printf "%s/layout_%03d" "$CACHE_ROOT" "$layout")
    if [ -s "$OUTPUT/cache.json" ]; then
        echo "Already built cache: $OUTPUT; do not modify"
        continue
    fi
    if [ -e "$OUTPUT" ]; then
        echo "ERROR: partial cache $OUTPUT exists without complete manifest" >&2
        echo "Inspect it manually before retrying. No silent overwrite." >&2
        exit 1
    fi
    echo "==== compile $layout-partition CPU mmap metadata only ===="
    singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
        "$PYTHON" -u -m scripts.m41_6_compile_metadata \
        --prepared-dir "$PREPARED" --output-dir "$OUTPUT"
    test -s "$OUTPUT/cache.json"
done

echo "M41.6 mmap compilation complete. Source graph never rebuilt."
