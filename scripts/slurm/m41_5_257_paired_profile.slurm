#!/bin/bash
#SBATCH --job-name=m41_5_pair_profile
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# M41.5: 6 AB/BA paired unprofiled forwards each at 32 and 64 parts,
# followed by a synchronized diagnostic forward at each layout.
# All on ONE Grace allocation, each forward a clean subprocess.
# No explicit --mem limit. No global one-hop CSR or partition rebuild.
set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
RUN_ROOT=/scratch/coop/theret/GraphAttention_runs
BASE_PREP=${M41_5_BASE_PREP:-${RUN_ROOT}/m41_4_257_generic_graph}
SWEEP_ROOT=${M41_5_SWEEP_ROOT:-${RUN_ROOT}/m41_4_257_scaling}
M37_RUN=${M37_RUN:-${RUN_ROOT}/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16}
CHECKPOINT=${CHECKPOINT:-best.pt}
REPEATS=${M41_5_REPEATS:-6}
SEED=${M41_5_SEED:-5678}
OUTPUT=${M41_5_OUTPUT:-${RUN_ROOT}/m41_5_257_paired_profile}

cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: checkout must be clean for benchmark traceability" >&2
    git status --short >&2
    exit 1
fi
for path in \
    "$BASE_PREP/prepare.json" \
    "$BASE_PREP/partitions/partitions.json" \
    "$SWEEP_ROOT/layouts/layout_032/prepare.json" \
    "$SWEEP_ROOT/layouts/layout_032/partitions/partitions.json" \
    "$M37_RUN/$CHECKPOINT" \
    "$M37_RUN/resolved_config.yaml" \
    "$M37_RUN/standardizers.pt"; do
    test -s "$path" || { echo "Missing prerequisite $path" >&2; exit 1; }
done
echo "==== M41.5 paired and diagnostic profile ===="
echo "Host: $(hostname), revision: $(git rev-parse HEAD)"
echo "Repetitions per layout: $REPEATS; seed: $SEED"
echo "Source graph: $BASE_PREP"
echo "Prepared layouts: 32 and 64; no connectivity rebuilt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "==== Python syntax gate ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    src/graph_attention/models/streamed_stage_profile.py \
    src/graph_attention/models/streamed_dinat.py \
    scripts/m41_4_generic_large.py \
    scripts/benchmark_m41_5_paired_profile.py \
    tests/unit/test_streamed_dinat.py \
    tests/unit/test_m41_5_paired_profile.py

echo "==== Numerical/profile and orchestration unit regressions ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_streamed_dinat.py \
    tests/unit/test_m41_5_paired_profile.py \
    tests/unit/test_m41_4_generic_csr.py

echo "==== 32-vs-64 balanced crossover: 12 unprofiled + 2 diagnostics ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_m41_5_paired_profile \
    --base-dir "$BASE_PREP" \
    --layout-root "$SWEEP_ROOT/layouts" \
    --run-dir "$M37_RUN" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT" \
    --repeats "$REPEATS" \
    --seed "$SEED"

echo "===== M41.5 complete: $OUTPUT/summary.json ====="
