#!/bin/bash
#SBATCH --job-name=m31_m33_gen_bench
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

M31_RUN=${RUN_ROOT}/m31_dinat_dit_linear_ddpm_1000e_h128_l10_h4
M32_RUN=${RUN_ROOT}/m32_dinat_dit_edm_1000e_h128_l10_h4
M33_RUN=${RUN_ROOT}/m33_dinat_dit_flow_tscale1000_1000e_h128_l10_h4

DDPM_GEN_NAME=linear_ddpm_reverse_sde_euler_maruyama_steps50_denoise_eps0p001_seed5678
EDM_GEN_NAME=edm_karras_heun_steps50_sigmin0p004_sigmax160_rho7_seed5678
FLOW_GEN_NAME=flow_heun_steps50_seed5678

M31_GEN=${M31_RUN}/generations/${DDPM_GEN_NAME}
M32_GEN=${M32_RUN}/generations/${EDM_GEN_NAME}
M33_GEN=${M33_RUN}/generations/${FLOW_GEN_NAME}

M31_BENCH_ROOT=${RUN_ROOT}/benchmark_m31_dinat_dit_linear_ddpm
M32_BENCH_ROOT=${RUN_ROOT}/benchmark_m32_dinat_dit_edm
M33_BENCH_ROOT=${RUN_ROOT}/benchmark_m33_dinat_dit_flow

M31_BENCH=${M31_BENCH_ROOT}/${DDPM_GEN_NAME}
M32_BENCH=${M32_BENCH_ROOT}/${EDM_GEN_NAME}
M33_BENCH=${M33_BENCH_ROOT}/${FLOW_GEN_NAME}

cd "$REPO"

echo "============================================================"
echo "M31/M32/M33 - DiNAT-DiT generation + benchmark"
echo "============================================================"
echo "Backbone:       DiNAT-DiT h128 L10 H4"
echo "Attention:      local / exact-two-hop alternating"
echo "Sampling steps: $SAMPLING_STEPS"
echo "Sampling seed:  $SAMPLING_SEED"
echo "Energy spectra enabled with 10-90% population bands"
echo "============================================================"

git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for run_dir in "$M31_RUN" "$M32_RUN" "$M33_RUN"; do
    for artifact in "$CHECKPOINT" summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
        test -f "${run_dir}/${artifact}" || {
            echo "ERROR: missing training artifact: ${run_dir}/${artifact}"
            exit 1
        }
    done
done

nvidia-smi

ensure_ddpm_generation () {
    if [ -f "$M31_GEN/generated_test.pt" ] && [ -f "$M31_GEN/summary.json" ]; then
        echo "M31 generation already complete; reusing: $M31_GEN"
        return
    fi
    local overwrite=false
    if [ -e "$M31_GEN" ]; then overwrite=true; fi

    echo
    echo "===== Generate M31 DDPM: reverse SDE / Euler-Maruyama ====="
    singularity exec --nv         -B /home         -B /scratch         "$IMAGE"         "$PYTHON" -u -m scripts.generate_generative             run_dir="$M31_RUN"             checkpoint="$CHECKPOINT"             device=cuda             batch_size="$BATCH_SIZE"             num_workers="$NUM_WORKERS"             overwrite="$overwrite"             sampling.steps="$SAMPLING_STEPS"             sampling.method=reverse_sde             sampling.solver=euler_maruyama             sampling.sampling_eps=0.001             sampling.final_denoise=true             sampling.seed="$SAMPLING_SEED"             output_name="$DDPM_GEN_NAME"
}

ensure_edm_generation () {
    if [ -f "$M32_GEN/generated_test.pt" ] && [ -f "$M32_GEN/summary.json" ]; then
        echo "M32 generation already complete; reusing: $M32_GEN"
        return
    fi
    local overwrite=false
    if [ -e "$M32_GEN" ]; then overwrite=true; fi

    echo
    echo "===== Generate M32 EDM: Karras / Heun ====="
    singularity exec --nv         -B /home         -B /scratch         "$IMAGE"         "$PYTHON" -u -m scripts.generate_slice_edm             run_dir="$M32_RUN"             checkpoint="$CHECKPOINT"             device=cuda             batch_size="$BATCH_SIZE"             num_workers="$NUM_WORKERS"             overwrite="$overwrite"             sampling.steps="$SAMPLING_STEPS"             sampling.sigma_min=0.004             sampling.sigma_max=160.0             sampling.rho=7.0             sampling.solver=heun             sampling.seed="$SAMPLING_SEED"             output_name="$EDM_GEN_NAME"
}

ensure_flow_generation () {
    if [ -f "$M33_GEN/generated_test.pt" ] && [ -f "$M33_GEN/summary.json" ]; then
        echo "M33 generation already complete; reusing: $M33_GEN"
        return
    fi
    local overwrite=false
    if [ -e "$M33_GEN" ]; then overwrite=true; fi

    echo
    echo "===== Generate M33 Flow: Heun ====="
    singularity exec --nv         -B /home         -B /scratch         "$IMAGE"         "$PYTHON" -u -m scripts.generate_generative             run_dir="$M33_RUN"             checkpoint="$CHECKPOINT"             device=cuda             batch_size="$BATCH_SIZE"             num_workers="$NUM_WORKERS"             overwrite="$overwrite"             sampling.steps="$SAMPLING_STEPS"             sampling.method=flow_ode             sampling.solver=heun             sampling.sampling_eps=0.0             sampling.final_denoise=false             sampling.seed="$SAMPLING_SEED"             output_name="$FLOW_GEN_NAME"
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

ensure_ddpm_generation
ensure_edm_generation
ensure_flow_generation

rebuild_benchmark "$M31_GEN" "$M31_BENCH_ROOT" "$M31_BENCH" "M31 DDPM"
rebuild_benchmark "$M32_GEN" "$M32_BENCH_ROOT" "$M32_BENCH" "M32 EDM"
rebuild_benchmark "$M33_GEN" "$M33_BENCH_ROOT" "$M33_BENCH" "M33 Flow"

echo
echo "============================================================"
echo "M31/M32/M33 GENERATION + BENCHMARK COMPLETE"
echo "M31: $M31_BENCH"
echo "M32: $M32_BENCH"
echo "M33: $M33_BENCH"
echo "============================================================"
