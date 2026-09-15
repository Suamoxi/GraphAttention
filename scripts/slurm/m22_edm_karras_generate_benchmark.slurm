#!/bin/bash
#SBATCH --job-name=m22_edm_karras_gen_bench
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
RUN_NAME=m22_m12_edm_1000e_h128_l10_h4
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
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
GEN_DIR=${RUN_DIR}/generations/${GEN_NAME}
BENCH_ROOT=${RUN_ROOT}/benchmark_m22_edm_karras
BENCH_DIR=${BENCH_ROOT}/${GEN_NAME}

M19_BASELINE=${RUN_ROOT}/benchmark_m19_m13_start999/ddpm_ancestral_gaussian_restart_start999_steps999_eta1_seed5678
RELATIVE_ROOT=${RUN_ROOT}/relative_benchmark_comparisons
RELATIVE_NAME=m19_ddpm_start999_vs_m22_edm_karras18
RELATIVE_DIR=${RELATIVE_ROOT}/${RELATIVE_NAME}

cd "$REPO"

printf '%s\n' "============================================================"
printf '%s\n' "M22 - EDM/Karras generation + distribution benchmark"
printf '%s\n' "============================================================"
printf 'Source run:     %s\n' "$RUN_DIR"
printf 'Checkpoint:     %s\n' "$CHECKPOINT"
printf 'Steps:          %s\n' "$SAMPLING_STEPS"
printf 'Sigma range:    %s -> %s\n' "$SIGMA_MAX" "$SIGMA_MIN"
printf 'rho:            %s\n' "$RHO"
printf 'Solver:         %s\n' "$SOLVER"
printf 'Sampling seed:  %s\n' "$SAMPLING_SEED"
printf 'Expected NFE:   %s\n' "$((2 * SAMPLING_STEPS - 1))"
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

for path in "$GEN_DIR" "$BENCH_DIR" "$RELATIVE_DIR"
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
echo "===== EDM generation ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.generate_slice_edm \
        run_dir="$RUN_DIR" \
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

if [ -d "$M19_BASELINE" ]; then
    echo
    echo "===== Relative comparison: M19 stable DDPM vs M22 EDM ====="
    singularity exec \
        -B /home \
        -B /scratch \
        "$IMAGE" \
        "$PYTHON" -u -m scripts.compare_generation_benchmarks \
            baseline_dir="$M19_BASELINE" \
            candidate_dir="$BENCH_DIR" \
            output_root="$RELATIVE_ROOT" \
            comparison_name="$RELATIVE_NAME"

    test -f "${RELATIVE_DIR}/summary.json"
    test -f "${RELATIVE_DIR}/relative_comparison.csv"
else
    echo
    echo "M19 baseline not found; skipping relative comparison: $M19_BASELINE"
fi

echo
echo "===== EDM benchmark summary ====="
cat "${BENCH_DIR}/summary.json"

if [ -f "${RELATIVE_DIR}/summary.json" ]; then
    echo
    echo "===== Relative comparison summary ====="
    cat "${RELATIVE_DIR}/summary.json"
fi

echo
echo "============================================================"
echo "M22 COMPLETE"
echo "Generation: ${GEN_DIR}"
echo "Benchmark:  ${BENCH_DIR}"
if [ -d "$RELATIVE_DIR" ]; then
    echo "Relative:   ${RELATIVE_DIR}"
fi
echo "============================================================"
