#!/bin/bash
#SBATCH --job-name=m41_4_scaling
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Controlled one-GPU benchmarking. No explicit --mem limit.
# Each individual forward runs in a fresh child Python process, all on the
# same allocated GH200; alternate ordering in the paired preloaded/lazy run.
set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
RUN_ROOT=/scratch/coop/theret/GraphAttention_runs
BASE_PREP=${M41_4_BASE_PREP:-${RUN_ROOT}/m41_4_257_generic_graph}
SWEEP_ROOT=${M41_4_SWEEP_ROOT:-${RUN_ROOT}/m41_4_257_scaling}
M37_RUN=${M37_RUN:-${RUN_ROOT}/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16}
CHECKPOINT=${CHECKPOINT:-best.pt}
SERIES=${M41_4_SCALING_SERIES:-repeated}
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: checkout must be clean for traceability" >&2
    git status --short >&2
    exit 1
fi
case "$SERIES" in
    repeated)
        LAYOUTS=${M41_4_SCALING_LAYOUTS:-64}
        STRATEGIES=${M41_4_SCALING_STRATEGIES:-"lazy preload"}
        REPEATS=${M41_4_SCALING_REPEATS:-3}
        ;;
    sweep)
        LAYOUTS=${M41_4_SCALING_LAYOUTS:-"16 32 64 128"}
        STRATEGIES=${M41_4_SCALING_STRATEGIES:-lazy}
        REPEATS=${M41_4_SCALING_REPEATS:-2}
        ;;
    *) echo "M41_4_SCALING_SERIES must be repeated or sweep" >&2; exit 1 ;;
esac
read -r -a MODES <<< "$STRATEGIES"
test "${#MODES[@]}" -gt 0 || { echo "No benchmark strategies" >&2; exit 1; }
for name in "$M37_RUN/$CHECKPOINT" "$M37_RUN/resolved_config.yaml" "$M37_RUN/standardizers.pt"; do
    test -s "$name" || { echo "Missing M37 artifact $name" >&2; exit 1; }
done
test -s "$BASE_PREP/prepare.json" || { echo "Missing prepared 64-partition base" >&2; exit 1; }
mkdir -p "$SWEEP_ROOT/benchmarks/$SERIES"
echo "M41.4 $SERIES: layouts=$LAYOUTS strategies=$STRATEGIES repeats=$REPEATS"
echo "Host $(hostname), commit $(git rev-parse HEAD)"
nvidia-smi

for layout in $LAYOUTS; do
    case "$layout" in
        16|32|64|128) ;;
        *) echo "Invalid layout=$layout" >&2; exit 1 ;;
    esac
    if [ "$layout" != "64" ]; then
        test -s "$(printf '%s/layouts/layout_%03d/prepare.json' "$SWEEP_ROOT" "$layout")" || {
            echo "Missing prepared $layout layout; run m41_4_257_scaling_prepare.slurm" >&2
            exit 1
        }
    fi
    out=$(printf '%s/benchmarks/%s/layout_%03d' "$SWEEP_ROOT" "$SERIES" "$layout")
    echo "==== $layout partitions: ${MODES[*]}, $REPEATS repeats ===="
    singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
        "$PYTHON" -u -m scripts.benchmark_m41_4_scalability \
        --base-dir "$BASE_PREP" \
        --layout-root "$SWEEP_ROOT/layouts" \
        --run-dir "$M37_RUN" \
        --checkpoint "$CHECKPOINT" \
        --layout "$layout" \
        --strategies "${MODES[@]}" \
        --repeats "$REPEATS" \
        --output-dir "$out"
done

echo "==== aggregate experiments ===="
if [ "$SERIES" = "sweep" ]; then
    extra=(--require-repeated --require-layouts 16 32 64 128)
else
    extra=()
fi
singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m scripts.summarize_m41_4_scalability \
    --root "$SWEEP_ROOT" "${extra[@]}"
echo "Finished M41.4 $SERIES: $SWEEP_ROOT/scaling_summary.md"
