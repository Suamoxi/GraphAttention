#!/bin/bash
#SBATCH --job-name=m25_m26_dit_edm_bench
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
RUN_ROOT=/scratch/coop/theret/GraphAttention_runs

FULL_RUN=${RUN_ROOT}/m25_full_dit_edm_1000e_h128_l10_h4
LOCAL_RUN=${RUN_ROOT}/m26_local_dit_edm_1000e_h128_l10_h4
CHECKPOINT=best.pt

BATCH_SIZE=128
NUM_WORKERS=4
SAMPLING_STEPS=18
SIGMA_MIN=0.004
SIGMA_MAX=160.0
RHO=7.0
SOLVER=heun
SAMPLING_SEED=5678

GEN_NAME=edm_karras_heun_steps18_sigmin0p004_sigmax160_rho7_seed5678
FULL_GEN=${FULL_RUN}/generations/${GEN_NAME}
LOCAL_GEN=${LOCAL_RUN}/generations/${GEN_NAME}

FULL_BENCH_ROOT=${RUN_ROOT}/benchmark_m25_full_dit_edm
LOCAL_BENCH_ROOT=${RUN_ROOT}/benchmark_m26_local_dit_edm
FULL_BENCH=${FULL_BENCH_ROOT}/${GEN_NAME}
LOCAL_BENCH=${LOCAL_BENCH_ROOT}/${GEN_NAME}

RELATIVE_ROOT=${RUN_ROOT}/relative_benchmark_comparisons
RELATIVE_NAME=m25_full_dit_vs_m26_local_dit_edm
RELATIVE_DIR=${RELATIVE_ROOT}/${RELATIVE_NAME}

cd "$REPO"

echo "============================================================"
echo "M25/M26 - matched Full DiT vs Local DiT EDM generation"
echo "============================================================"
echo "Full:  $FULL_RUN"
echo "Local: $LOCAL_RUN"
echo "Sampler: Karras Heun, 18 steps, seed 5678"
echo "============================================================"

git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for run_dir in "$FULL_RUN" "$LOCAL_RUN"; do
    if [ ! -f "${run_dir}/${CHECKPOINT}" ]; then
        echo "ERROR: missing checkpoint: ${run_dir}/${CHECKPOINT}"
        exit 1
    fi
done

for path in "$FULL_GEN" "$LOCAL_GEN" "$FULL_BENCH" "$LOCAL_BENCH" "$RELATIVE_DIR"; do
    if [ -e "$path" ]; then
        echo "ERROR: output already exists: $path"
        exit 1
    fi
done

nvidia-smi

generate_one () {
    local run_dir="$1"
    echo
    echo "===== Generate: ${run_dir} ====="
    singularity exec --nv \
        -B /home \
        -B /scratch \
        "$IMAGE" \
        "$PYTHON" -u -m scripts.generate_slice_edm \
            run_dir="$run_dir" \
            checkpoint="$CHECKPOINT" \
            device=cuda \
            batch_size="$BATCH_SIZE" \
            num_workers="$NUM_WORKERS" \
            sampling.steps="$SAMPLING_STEPS" \
            sampling.sigma_min="$SIGMA_MIN" \
            sampling.sigma_max="$SIGMA_MAX" \
            sampling.rho="$RHO" \
            sampling.solver="$SOLVER" \
            sampling.seed="$SAMPLING_SEED" \
            output_name="$GEN_NAME"
}

benchmark_one () {
    local gen_dir="$1"
    local bench_root="$2"
    echo
    echo "===== Benchmark: ${gen_dir} ====="
    singularity exec \
        -B /home \
        -B /scratch \
        "$IMAGE" \
        "$PYTHON" -u -m scripts.benchmark_generation \
            run_dir="$gen_dir" \
            output_root="$bench_root"
}

generate_one "$FULL_RUN"
generate_one "$LOCAL_RUN"
benchmark_one "$FULL_GEN" "$FULL_BENCH_ROOT"
benchmark_one "$LOCAL_GEN" "$LOCAL_BENCH_ROOT"

echo
echo "===== Relative comparison: Full DiT vs Local DiT ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.compare_generation_benchmarks \
        baseline_dir="$FULL_BENCH" \
        candidate_dir="$LOCAL_BENCH" \
        output_root="$RELATIVE_ROOT" \
        comparison_name="$RELATIVE_NAME"

for file in     "$FULL_GEN/generated_test.pt"     "$LOCAL_GEN/generated_test.pt"     "$FULL_BENCH/summary.json"     "$LOCAL_BENCH/summary.json"     "$RELATIVE_DIR/summary.json"     "$RELATIVE_DIR/relative_comparison.csv"
do
    test -f "$file"
done

echo
echo "===== Full DiT benchmark ====="
cat "$FULL_BENCH/summary.json"
echo
echo "===== Local DiT benchmark ====="
cat "$LOCAL_BENCH/summary.json"
echo
echo "===== Full vs Local comparison ====="
cat "$RELATIVE_DIR/summary.json"

echo
echo "============================================================"
echo "M25/M26 DIT EDM GENERATION + BENCHMARK COMPLETE"
echo "Full benchmark:  $FULL_BENCH"
echo "Local benchmark: $LOCAL_BENCH"
echo "Relative:        $RELATIVE_DIR"
echo "============================================================"
