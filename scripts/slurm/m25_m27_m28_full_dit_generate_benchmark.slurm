#!/bin/bash
#SBATCH --job-name=full_dit_gen_bench
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

EDM_RUN=${RUN_ROOT}/m25_full_dit_edm_1000e_h128_l10_h4
DDPM_RUN=${RUN_ROOT}/m27_full_dit_linear_ddpm_1000e_h128_l10_h4
FLOW_RUN=${RUN_ROOT}/m28_full_dit_flow_1000e_h128_l10_h4

EDM_GEN_NAME=edm_karras_heun_steps50_sigmin0p004_sigmax160_rho7_seed5678
DDPM_GEN_NAME=linear_ddpm_reverse_sde_euler_maruyama_steps50_denoise_eps0p001_seed5678
FLOW_GEN_NAME=flow_heun_steps50_seed5678

EDM_GEN=${EDM_RUN}/generations/${EDM_GEN_NAME}
DDPM_GEN=${DDPM_RUN}/generations/${DDPM_GEN_NAME}
FLOW_GEN=${FLOW_RUN}/generations/${FLOW_GEN_NAME}

EDM_BENCH_ROOT=${RUN_ROOT}/benchmark_m25_full_dit_edm
DDPM_BENCH_ROOT=${RUN_ROOT}/benchmark_m27_full_dit_linear_ddpm
FLOW_BENCH_ROOT=${RUN_ROOT}/benchmark_m28_full_dit_flow

EDM_BENCH=${EDM_BENCH_ROOT}/${EDM_GEN_NAME}
DDPM_BENCH=${DDPM_BENCH_ROOT}/${DDPM_GEN_NAME}
FLOW_BENCH=${FLOW_BENCH_ROOT}/${FLOW_GEN_NAME}

RELATIVE_ROOT=${RUN_ROOT}/relative_benchmark_comparisons
DDPM_EDM_NAME=m27_reverse_sde50_vs_m25_edm50_full_dit
DDPM_FLOW_NAME=m27_reverse_sde50_vs_m28_flow50_full_dit
EDM_FLOW_NAME=m25_edm50_vs_m28_flow50_full_dit
DDPM_EDM_DIR=${RELATIVE_ROOT}/${DDPM_EDM_NAME}
DDPM_FLOW_DIR=${RELATIVE_ROOT}/${DDPM_FLOW_NAME}
EDM_FLOW_DIR=${RELATIVE_ROOT}/${EDM_FLOW_NAME}

cd "$REPO"

echo "============================================================"
echo "Full DiT generative benchmark: DDPM vs EDM vs Flow Matching"
echo "============================================================"
echo "Backbone:       Full DiT h128 L10 H4"
echo "Checkpoint:     best.pt"
echo "Solver steps:   $SAMPLING_STEPS for every method"
echo "Sampling seed:  $SAMPLING_SEED"
echo "DDPM sampler:   reverse VP-SDE / Euler-Maruyama"
echo "DDPM endpoint:  t=0.001 + final x0 denoise"
echo "EDM sampler:    Karras ODE / Heun"
echo "Flow sampler:   flow ODE / Heun"
echo "NFE:            DDPM=51, EDM=99, Flow=100"
echo "============================================================"

git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for run_dir in "$DDPM_RUN" "$EDM_RUN" "$FLOW_RUN"; do
    for artifact in "$CHECKPOINT" summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
        if [ ! -f "${run_dir}/${artifact}" ]; then
            echo "ERROR: missing training artifact: ${run_dir}/${artifact}"
            exit 1
        fi
    done
done

echo
echo "===== Checking controlled split ====="
singularity exec -B /home -B /scratch "$IMAGE" "$PYTHON" - "$DDPM_RUN" "$EDM_RUN" "$FLOW_RUN" <<'PY'
import json
import sys
from pathlib import Path

runs = [Path(value) for value in sys.argv[1:]]
splits = []
for run in runs:
    manifest = json.loads((run / "dataset_split_manifest.json").read_text())
    splits.append(
        (
            tuple(manifest.get("train_ids", ())),
            tuple(manifest.get("validation_ids", ())),
            tuple(manifest.get("test_ids", ())),
        )
    )

if not splits[0][0] or not splits[0][1] or not splits[0][2]:
    raise SystemExit("ERROR: split manifest is missing train/validation/test IDs")
if not all(split == splits[0] for split in splits[1:]):
    raise SystemExit("ERROR: DDPM, EDM, and Flow runs do not use identical data splits")

print(
    "Controlled split verified:",
    f"train={len(splits[0][0])}",
    f"validation={len(splits[0][1])}",
    f"test={len(splits[0][2])}",
)
PY

nvidia-smi

ensure_ddpm_generation () {
    if [ -f "${DDPM_GEN}/generated_test.pt" ] && [ -f "${DDPM_GEN}/summary.json" ]; then
        echo "DDPM generation already complete; reusing: $DDPM_GEN"
        return
    fi
    local overwrite=false
    if [ -e "$DDPM_GEN" ]; then overwrite=true; fi

    echo
    echo "===== Generate DDPM: reverse SDE, 50 Euler-Maruyama steps ====="
    singularity exec --nv -B /home -B /scratch "$IMAGE" "$PYTHON" -u -m scripts.generate_generative run_dir="$DDPM_RUN" checkpoint="$CHECKPOINT" device=cuda batch_size="$BATCH_SIZE" num_workers="$NUM_WORKERS" overwrite="$overwrite" sampling.steps="$SAMPLING_STEPS" sampling.method=reverse_sde sampling.solver=euler_maruyama sampling.sampling_eps=0.001 sampling.final_denoise=true sampling.seed="$SAMPLING_SEED" output_name="$DDPM_GEN_NAME"
}

ensure_edm_generation () {
    if [ -f "${EDM_GEN}/generated_test.pt" ] && [ -f "${EDM_GEN}/summary.json" ]; then
        echo "EDM generation already complete; reusing: $EDM_GEN"
        return
    fi
    local overwrite=false
    if [ -e "$EDM_GEN" ]; then overwrite=true; fi

    echo
    echo "===== Generate EDM: 50 Karras-Heun steps ====="
    singularity exec --nv -B /home -B /scratch "$IMAGE" "$PYTHON" -u -m scripts.generate_slice_edm run_dir="$EDM_RUN" checkpoint="$CHECKPOINT" device=cuda batch_size="$BATCH_SIZE" num_workers="$NUM_WORKERS" overwrite="$overwrite" sampling.steps="$SAMPLING_STEPS" sampling.sigma_min=0.004 sampling.sigma_max=160.0 sampling.rho=7.0 sampling.solver=heun sampling.seed="$SAMPLING_SEED" output_name="$EDM_GEN_NAME"
}

ensure_flow_generation () {
    if [ -f "${FLOW_GEN}/generated_test.pt" ] && [ -f "${FLOW_GEN}/summary.json" ]; then
        echo "Flow generation already complete; reusing: $FLOW_GEN"
        return
    fi
    local overwrite=false
    if [ -e "$FLOW_GEN" ]; then overwrite=true; fi

    echo
    echo "===== Generate Flow Matching: 50 Heun steps ====="
    singularity exec --nv -B /home -B /scratch "$IMAGE" "$PYTHON" -u -m scripts.generate_generative run_dir="$FLOW_RUN" checkpoint="$CHECKPOINT" device=cuda batch_size="$BATCH_SIZE" num_workers="$NUM_WORKERS" overwrite="$overwrite" sampling.steps="$SAMPLING_STEPS" sampling.method=flow_ode sampling.solver=heun sampling.sampling_eps=0.0 sampling.final_denoise=false sampling.seed="$SAMPLING_SEED" output_name="$FLOW_GEN_NAME"
}

ensure_benchmark () {
    local gen_dir="$1"
    local bench_root="$2"
    local bench_dir="$3"
    local label="$4"

    if [ -f "${bench_dir}/summary.json" ]; then
        echo "${label} benchmark already complete; reusing: ${bench_dir}"
        return
    fi
    local overwrite=false
    if [ -e "$bench_dir" ]; then overwrite=true; fi

    echo
    echo "===== Benchmark ${label} ====="
    singularity exec -B /home -B /scratch "$IMAGE" "$PYTHON" -u -m scripts.benchmark_generation run_dir="$gen_dir" output_root="$bench_root" overwrite="$overwrite"
}

ensure_comparison () {
    local baseline="$1"
    local candidate="$2"
    local name="$3"
    local output_dir="$4"
    local label="$5"

    if [ -f "${output_dir}/summary.json" ] && [ -f "${output_dir}/relative_comparison.csv" ]; then
        echo "${label} comparison already complete; reusing: ${output_dir}"
        return
    fi
    local overwrite=false
    if [ -e "$output_dir" ]; then overwrite=true; fi

    echo
    echo "===== Relative comparison: ${label} ====="
    singularity exec -B /home -B /scratch "$IMAGE" "$PYTHON" -u -m scripts.compare_generation_benchmarks baseline_dir="$baseline" candidate_dir="$candidate" output_root="$RELATIVE_ROOT" comparison_name="$name" overwrite="$overwrite"
}

ensure_ddpm_generation
ensure_edm_generation
ensure_flow_generation

for file in "$DDPM_GEN/generated_test.pt" "$EDM_GEN/generated_test.pt" "$FLOW_GEN/generated_test.pt"; do
    test -f "$file"
done

ensure_benchmark "$DDPM_GEN" "$DDPM_BENCH_ROOT" "$DDPM_BENCH" "DDPM"
ensure_benchmark "$EDM_GEN" "$EDM_BENCH_ROOT" "$EDM_BENCH" "EDM"
ensure_benchmark "$FLOW_GEN" "$FLOW_BENCH_ROOT" "$FLOW_BENCH" "Flow"

ensure_comparison "$DDPM_BENCH" "$EDM_BENCH" "$DDPM_EDM_NAME" "$DDPM_EDM_DIR" "DDPM -> EDM"
ensure_comparison "$DDPM_BENCH" "$FLOW_BENCH" "$DDPM_FLOW_NAME" "$DDPM_FLOW_DIR" "DDPM -> Flow"
ensure_comparison "$EDM_BENCH" "$FLOW_BENCH" "$EDM_FLOW_NAME" "$EDM_FLOW_DIR" "EDM -> Flow"

echo
echo "===== DDPM benchmark summary ====="
cat "$DDPM_BENCH/summary.json"
echo
echo "===== EDM benchmark summary ====="
cat "$EDM_BENCH/summary.json"
echo
echo "===== Flow benchmark summary ====="
cat "$FLOW_BENCH/summary.json"

echo
echo "============================================================"
echo "FULL DIT 50-STEP GENERATIVE BENCHMARK COMPLETE"
echo "DDPM benchmark: $DDPM_BENCH"
echo "EDM benchmark:  $EDM_BENCH"
echo "Flow benchmark: $FLOW_BENCH"
echo "DDPM vs EDM:    $DDPM_EDM_DIR"
echo "DDPM vs Flow:   $DDPM_FLOW_DIR"
echo "EDM vs Flow:    $EDM_FLOW_DIR"
echo "============================================================"
