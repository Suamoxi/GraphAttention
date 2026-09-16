#!/bin/bash
#SBATCH --job-name=m23_vp_sde_gen_bench
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
RUN_NAME=m23_m12_vp_sde_1000e_h128_l10_h4
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
CHECKPOINT=best.pt

BATCH_SIZE=128
NUM_WORKERS=4
SAMPLING_STEPS=1000
SAMPLING_EPS=0.001
SAMPLING_SEED=5678
FINAL_DENOISE=true

PF_METHOD=probability_flow_ode
PF_SOLVER=heun
PF_NAME=vp_probability_flow_heun_steps1000_denoise_eps0p001_seed5678
PF_GEN_DIR=${RUN_DIR}/generations/${PF_NAME}

SDE_METHOD=reverse_sde
SDE_SOLVER=euler_maruyama
SDE_NAME=vp_reverse_sde_euler_maruyama_steps1000_denoise_eps0p001_seed5678
SDE_GEN_DIR=${RUN_DIR}/generations/${SDE_NAME}

BENCH_ROOT=${RUN_ROOT}/benchmark_m23_vp_sde
PF_BENCH_DIR=${BENCH_ROOT}/${PF_NAME}
SDE_BENCH_DIR=${BENCH_ROOT}/${SDE_NAME}

RELATIVE_ROOT=${RUN_ROOT}/relative_benchmark_comparisons
RELATIVE_NAME=m23_pf_ode_vs_reverse_sde_1000
RELATIVE_DIR=${RELATIVE_ROOT}/${RELATIVE_NAME}

cd "$REPO"

printf '%s\n' "============================================================"
printf '%s\n' "M23 - VP-SDE dual-sampler generation + benchmark"
printf '%s\n' "============================================================"
printf 'Source run:       %s\n' "$RUN_DIR"
printf 'Checkpoint:       %s\n' "$CHECKPOINT"
printf 'Sampling steps:   %s\n' "$SAMPLING_STEPS"
printf 'Sampling eps:     %s\n' "$SAMPLING_EPS"
printf 'Sampling seed:    %s\n' "$SAMPLING_SEED"
printf 'Final denoise:    %s\n' "$FINAL_DENOISE"
printf 'M23a:             PF-ODE / Heun\n'
printf 'M23a expected NFE:%s\n' "$((2 * SAMPLING_STEPS + 1))"
printf 'M23b:             reverse SDE / Euler-Maruyama\n'
printf 'M23b expected NFE:%s\n' "$((SAMPLING_STEPS + 1))"
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

for path in \
    "$PF_GEN_DIR" \
    "$SDE_GEN_DIR" \
    "$PF_BENCH_DIR" \
    "$SDE_BENCH_DIR" \
    "$RELATIVE_DIR"
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
echo "===== M23a generation: probability-flow ODE / Heun ====="
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
        sampling.method="$PF_METHOD" \
        sampling.solver="$PF_SOLVER" \
        sampling.sampling_eps="$SAMPLING_EPS" \
        sampling.final_denoise="$FINAL_DENOISE" \
        sampling.seed="$SAMPLING_SEED" \
        output_name="$PF_NAME"

echo
echo "===== M23a distribution benchmark ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_generation \
        run_dir="$PF_GEN_DIR" \
        output_root="$BENCH_ROOT"

echo
echo "===== M23b generation: reverse SDE / Euler-Maruyama ====="
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
        sampling.method="$SDE_METHOD" \
        sampling.solver="$SDE_SOLVER" \
        sampling.sampling_eps="$SAMPLING_EPS" \
        sampling.final_denoise="$FINAL_DENOISE" \
        sampling.seed="$SAMPLING_SEED" \
        output_name="$SDE_NAME"

echo
echo "===== M23b distribution benchmark ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_generation \
        run_dir="$SDE_GEN_DIR" \
        output_root="$BENCH_ROOT"

echo
echo "===== Relative comparison: M23a PF-ODE vs M23b reverse SDE ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.compare_generation_benchmarks \
        baseline_dir="$PF_BENCH_DIR" \
        candidate_dir="$SDE_BENCH_DIR" \
        output_root="$RELATIVE_ROOT" \
        comparison_name="$RELATIVE_NAME"

echo
echo "===== Checking outputs ====="
for path in \
    "${PF_GEN_DIR}/generated_test.pt" \
    "${PF_GEN_DIR}/summary.json" \
    "${SDE_GEN_DIR}/generated_test.pt" \
    "${SDE_GEN_DIR}/summary.json" \
    "${PF_BENCH_DIR}/summary.json" \
    "${PF_BENCH_DIR}/channel_metrics.csv" \
    "${PF_BENCH_DIR}/spectral_bands.csv" \
    "${PF_BENCH_DIR}/physical_metrics.csv" \
    "${SDE_BENCH_DIR}/summary.json" \
    "${SDE_BENCH_DIR}/channel_metrics.csv" \
    "${SDE_BENCH_DIR}/spectral_bands.csv" \
    "${SDE_BENCH_DIR}/physical_metrics.csv" \
    "${RELATIVE_DIR}/summary.json" \
    "${RELATIVE_DIR}/relative_comparison.csv"
do
    test -f "$path"
done

echo
echo "===== M23a benchmark summary ====="
cat "${PF_BENCH_DIR}/summary.json"

echo
echo "===== M23b benchmark summary ====="
cat "${SDE_BENCH_DIR}/summary.json"

echo
echo "===== M23a vs M23b relative comparison ====="
cat "${RELATIVE_DIR}/summary.json"

echo
echo "============================================================"
echo "M23 GENERATION + BENCHMARK COMPLETE"
echo "PF generation:  ${PF_GEN_DIR}"
echo "PF benchmark:   ${PF_BENCH_DIR}"
echo "SDE generation: ${SDE_GEN_DIR}"
echo "SDE benchmark:  ${SDE_BENCH_DIR}"
echo "Relative:       ${RELATIVE_DIR}"
echo "============================================================"
