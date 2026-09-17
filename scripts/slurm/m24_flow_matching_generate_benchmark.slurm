#!/bin/bash
#SBATCH --job-name=m24_flow_gen_bench
#SBATCH --time=02:00:00
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
RUN_NAME=m24_m12_flow_1000e_h128_l10_h4
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
CHECKPOINT=best.pt

BATCH_SIZE=128
NUM_WORKERS=4
SAMPLING_STEPS=50
SOLVER=heun
SAMPLING_SEED=5678

GEN_NAME=flow_heun_steps50_seed5678
GEN_DIR=${RUN_DIR}/generations/${GEN_NAME}
BENCH_ROOT=${RUN_ROOT}/benchmark_m24_flow
BENCH_DIR=${BENCH_ROOT}/${GEN_NAME}

OLD_FLOW_BENCH=${RUN_ROOT}/benchmark_300e_comparison/m12_flow_l10/m12_dinat_exact2_flow_300e_h128_l10_h4
M22_BENCH=${RUN_ROOT}/benchmark_m22_edm_karras/edm_karras_heun_steps18_sigmin0p004_sigmax160_rho7_seed5678
RELATIVE_ROOT=${RUN_ROOT}/relative_benchmark_comparisons
FLOW_RELATIVE_NAME=m11_flow300_vs_m24_flow1000
FLOW_RELATIVE_DIR=${RELATIVE_ROOT}/${FLOW_RELATIVE_NAME}
EDM_RELATIVE_NAME=m22_edm_vs_m24_flow1000
EDM_RELATIVE_DIR=${RELATIVE_ROOT}/${EDM_RELATIVE_NAME}

cd "$REPO"

printf '%s\n' "============================================================"
printf '%s\n' "M24 - Flow Matching generation + distribution benchmark"
printf '%s\n' "============================================================"
printf 'Source run:     %s\n' "$RUN_DIR"
printf 'Checkpoint:     %s\n' "$CHECKPOINT"
printf 'Sampler:        flow ODE"
printf 'Steps:          %s\n' "$SAMPLING_STEPS"
printf 'Solver:         %s\n' "$SOLVER"
printf 'Sampling seed:  %s\n' "$SAMPLING_SEED"
printf 'Expected NFE:   %s\n' "$((2 * SAMPLING_STEPS))"
printf '%s\n' "============================================================"

echo
echo "===== Git ====="
git rev-parse HEAD
git status --short

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

if [ ! -f "${RUN_DIR}/${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint does not exist: ${RUN_DIR}/${CHECKPOINT}"
    exit 1
fi

for path in "$GEN_DIR" "$BENCH_DIR" "$FLOW_RELATIVE_DIR" "$EDM_RELATIVE_DIR"
do
    if [ -e "$path" ]; then
        echo "ERROR: output already exists: $path"
        exit 1
    fi
done

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== Flow Matching generation ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.generate_generative \
        run_dir="$RUN_DIR" \
        checkpoint="$CHECKPOINT" \
        device=cuda \
        batch_size="$BATCH_SIZE" \
        num_workers="$NUM_WORKERS" \
        sampling.steps="$SAMPLING_STEPS" \
        sampling.method=flow_ode \
        sampling.solver="$SOLVER" \
        sampling.sampling_eps=0.0 \
        sampling.final_denoise=false \
        sampling.seed="$SAMPLING_SEED" \
        output_name="$GEN_NAME"

echo
echo "===== Distribution benchmark ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_generation \
        run_dir="$GEN_DIR" \
        output_root="$BENCH_ROOT"

test -f "${GEN_DIR}/generated_test.pt"
test -f "${GEN_DIR}/summary.json"
test -f "${BENCH_DIR}/summary.json"
test -f "${BENCH_DIR}/channel_metrics.csv"
test -f "${BENCH_DIR}/spectral_bands.csv"
test -f "${BENCH_DIR}/physical_metrics.csv"

if [ -d "$OLD_FLOW_BENCH" ]; then
    echo
    echo "===== Relative comparison: 300e Flow vs M24 1000e Flow ====="
    singularity exec \
        -B /home \
        -B /scratch \
        "$IMAGE" \
        "$PYTHON" -u -m scripts.compare_generation_benchmarks \
            baseline_dir="$OLD_FLOW_BENCH" \
            candidate_dir="$BENCH_DIR" \
            output_root="$RELATIVE_ROOT" \
            comparison_name="$FLOW_RELATIVE_NAME"

    test -f "${FLOW_RELATIVE_DIR}/summary.json"
    test -f "${FLOW_RELATIVE_DIR}/relative_comparison.csv"
else
    echo
    echo "Old 300e Flow benchmark not found; skipping: $OLD_FLOW_BENCH"
fi

if [ -d "$M22_BENCH" ]; then
    echo
    echo "===== Relative comparison: M22 EDM vs M24 Flow ====="
    singularity exec \
        -B /home \
        -B /scratch \
        "$IMAGE" \
        "$PYTHON" -u -m scripts.compare_generation_benchmarks \
            baseline_dir="$M22_BENCH" \
            candidate_dir="$BENCH_DIR" \
            output_root="$RELATIVE_ROOT" \
            comparison_name="$EDM_RELATIVE_NAME"

    test -f "${EDM_RELATIVE_DIR}/summary.json"
    test -f "${EDM_RELATIVE_DIR}/relative_comparison.csv"
else
    echo
    echo "M22 EDM benchmark not found; skipping: $M22_BENCH"
fi

echo
echo "===== M24 benchmark summary ====="
cat "${BENCH_DIR}/summary.json"

if [ -f "${FLOW_RELATIVE_DIR}/summary.json" ]; then
    echo
    echo "===== 300e Flow vs M24 summary ====="
    cat "${FLOW_RELATIVE_DIR}/summary.json"
fi

if [ -f "${EDM_RELATIVE_DIR}/summary.json" ]; then
    echo
    echo "===== M22 EDM vs M24 summary ====="
    cat "${EDM_RELATIVE_DIR}/summary.json"
fi

echo
echo "============================================================"
echo "M24 FLOW GENERATION + BENCHMARK COMPLETE"
echo "Generation: ${GEN_DIR}"
echo "Benchmark:  ${BENCH_DIR}"
if [ -d "$FLOW_RELATIVE_DIR" ]; then
    echo "Relative Flow: ${FLOW_RELATIVE_DIR}"
fi
if [ -d "$EDM_RELATIVE_DIR" ]; then
    echo "Relative EDM:  ${EDM_RELATIVE_DIR}"
fi
echo "============================================================"
