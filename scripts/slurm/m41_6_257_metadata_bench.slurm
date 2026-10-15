#!/bin/bash
#SBATCH --job-name=m41_6_metadata_bench
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# 32/64 partitions; baseline vs selective NPZ vs precompiled mmap static
# metadata, three unprofiled runs per configuration in clean subprocesses,
# then separate synchronized diagnostics. Exactly unchanged M37 math.
# Grace one GPU, four CPUs, four hours, no explicit --mem.
set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
RUN_ROOT=/scratch/coop/theret/GraphAttention_runs
BASE_PREP=${M41_6_BASE_PREP:-${RUN_ROOT}/m41_4_257_generic_graph}
SWEEP_ROOT=${M41_6_SWEEP_ROOT:-${RUN_ROOT}/m41_4_257_scaling}
CACHE_ROOT=${M41_6_CACHE_ROOT:-${RUN_ROOT}/m41_6_257_metadata_cache}
M37_RUN=${M37_RUN:-${RUN_ROOT}/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16}
CHECKPOINT=${CHECKPOINT:-best.pt}
REPEATS=${M41_6_REPEATS:-3}
SEED=${M41_6_SEED:-5678}
OUTPUT=${M41_6_OUTPUT:-${RUN_ROOT}/m41_6_257_metadata_benchmark}
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: checkout must be clean" >&2
    git status --short >&2
    exit 1
fi
for source in \
    "$BASE_PREP/prepare.json" \
    "$SWEEP_ROOT/layouts/layout_032/prepare.json" \
    "$CACHE_ROOT/layout_032/cache.json" \
    "$CACHE_ROOT/layout_064/cache.json" \
    "$M37_RUN/$CHECKPOINT" \
    "$M37_RUN/resolved_config.yaml" \
    "$M37_RUN/standardizers.pt"; do
    test -s "$source" || { echo "Missing prerequisite $source" >&2; exit 1; }
done
echo "==== M41.6 benchmark host $(hostname), revision $(git rev-parse HEAD) ===="
nvidia-smi
echo "==== Static syntax validation ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    scripts/benchmark_m41_6_metadata.py \
    src/graph_attention/models/selective_disk_dinat.py

echo "==== A/B/C 257^3 forward benchmark: 32 and 64 partitions ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_m41_6_metadata \
    --base-dir "$BASE_PREP" \
    --layout-root "$SWEEP_ROOT/layouts" \
    --cache-root "$CACHE_ROOT" \
    --run-dir "$M37_RUN" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT" \
    --repeats "$REPEATS" \
    --seed "$SEED"
echo "M41.6 benchmark complete: $OUTPUT/summary.json"
