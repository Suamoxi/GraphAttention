#!/bin/bash
#SBATCH --job-name=m29_m30_gen_bench
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

CHECKPOINT=best.pt
BATCH_SIZE=128
NUM_WORKERS=4
SAMPLING_STEPS=50
SAMPLING_SEED=5678
GEN_NAME=flow_heun_steps50_seed5678

M29_RUN=${RUN_ROOT}/m29_full_dit_flow_tscale1000_1000e_h128_l10_h4
M30_RUN=${RUN_ROOT}/m30_full_dit_flow_tscale1000_d4arch_1000e_h128_l6_h4_temb256

M29_GEN=${M29_RUN}/generations/${GEN_NAME}
M30_GEN=${M30_RUN}/generations/${GEN_NAME}

M29_BENCH_ROOT=${RUN_ROOT}/benchmark_m29_full_dit_flow_tscale1000
M30_BENCH_ROOT=${RUN_ROOT}/benchmark_m30_full_dit_flow_d4arch
M29_BENCH=${M29_BENCH_ROOT}/${GEN_NAME}
M30_BENCH=${M30_BENCH_ROOT}/${GEN_NAME}

cd "$REPO"

echo "============================================================"
echo "M29/M30 - Full DiT Flow generation + benchmark"
echo "============================================================"
echo "Sampler:       Flow ODE / Heun"
echo "Steps:         $SAMPLING_STEPS"
echo "Sampling seed: $SAMPLING_SEED"
echo "Checkpoint:    $CHECKPOINT"
echo "Energy spectra enabled with 10-90% population bands"
echo "============================================================"

git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for run_dir in "$M29_RUN" "$M30_RUN"; do
    for artifact in "$CHECKPOINT" summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
        test -f "${run_dir}/${artifact}" || {
            echo "ERROR: missing training artifact: ${run_dir}/${artifact}"
            exit 1
        }
    done
done

nvidia-smi

ensure_flow_generation () {
    local run_dir="$1"
    local gen_dir="$2"
    local label="$3"

    if [ -f "${gen_dir}/generated_test.pt" ] && [ -f "${gen_dir}/summary.json" ]; then
        echo "${label} generation already complete; reusing: ${gen_dir}"
        return
    fi

    local overwrite=false
    if [ -e "$gen_dir" ]; then overwrite=true; fi

    echo
    echo "===== Generate ${label}: 50-step Heun ====="
    singularity exec --nv         -B /home         -B /scratch         "$IMAGE"         "$PYTHON" -u -m scripts.generate_generative             run_dir="$run_dir"             checkpoint="$CHECKPOINT"             device=cuda             batch_size="$BATCH_SIZE"             num_workers="$NUM_WORKERS"             overwrite="$overwrite"             sampling.steps="$SAMPLING_STEPS"             sampling.method=flow_ode             sampling.solver=heun             sampling.sampling_eps=0.0             sampling.final_denoise=false             sampling.seed="$SAMPLING_SEED"             output_name="$GEN_NAME"
}

rebuild_benchmark () {
    local gen_dir="$1"
    local bench_root="$2"
    local bench_dir="$3"
    local label="$4"

    echo
    echo "===== Benchmark ${label} ====="
    singularity exec         -B /home         -B /scratch         "$IMAGE"         "$PYTHON" -u -m scripts.benchmark_generation             run_dir="$gen_dir"             output_root="$bench_root"             overwrite=true             energy_spectrum.enabled=true             energy_spectrum.lower_quantile=0.10             energy_spectrum.upper_quantile=0.90             plots.energy_spectrum=true

    for artifact in         summary.json         channel_metrics.csv         spectral_bands.csv         physical_metrics.csv         energy_spectrum_per_sample.csv         energy_spectrum_summary.csv         energy_spectral_bands.csv
    do
        test -s "${bench_dir}/${artifact}"
    done
    test -s "${bench_dir}/plots/energy_spectrum/energy_spectrum_mean.png"
    test -s "${bench_dir}/plots/energy_spectrum/energy_spectrum_median.png"
}

ensure_flow_generation "$M29_RUN" "$M29_GEN" "M29"
ensure_flow_generation "$M30_RUN" "$M30_GEN" "M30"

rebuild_benchmark "$M29_GEN" "$M29_BENCH_ROOT" "$M29_BENCH" "M29"
rebuild_benchmark "$M30_GEN" "$M30_BENCH_ROOT" "$M30_BENCH" "M30"

echo
echo "============================================================"
echo "M29/M30 GENERATION + BENCHMARK COMPLETE"
echo "M29: $M29_BENCH"
echo "M30: $M30_BENCH"
echo "============================================================"
