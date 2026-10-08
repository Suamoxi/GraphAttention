#!/bin/bash
#SBATCH --job-name=m33_m39_self_cmp
#SBATCH --time=01:30:00
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

M33_RUN=${RUN_ROOT}/m33_dinat_dit_flow_tscale1000_1000e_h128_l10_h4
M39_RUN=${RUN_ROOT}/m39_dinat_dit_self_flow_tscale1000_1000e_h128_l10_h4

GEN_NAME=flow_heun_steps50_seed5678
M33_GEN=${M33_RUN}/generations/${GEN_NAME}
M39_GEN=${M39_RUN}/generations/${GEN_NAME}

OUTPUT_ROOT=${RUN_ROOT}/m33_m39_dinat_self_attention_comparison
TRAIN_OUT=${OUTPUT_ROOT}/training
M33_REBENCH_ROOT=${OUTPUT_ROOT}/rebenchmark_m33
M39_REBENCH_ROOT=${OUTPUT_ROOT}/rebenchmark_m39
M33_BENCH=${M33_REBENCH_ROOT}/${GEN_NAME}
M39_BENCH=${M39_REBENCH_ROOT}/${GEN_NAME}
QUALITY_OUT=${OUTPUT_ROOT}/generation_quality
PLOT_OUT=${OUTPUT_ROOT}/plots
EFF_OUT=${OUTPUT_ROOT}/efficiency

cd "$REPO"

echo "============================================================"
echo "M33 vs M39 - controlled DiNAT self-attention ablation"
echo "============================================================"
echo "M33: local / exact-two-hop, no self"
echo "M39: self+local / self+exact-two-hop"
echo "Task: Flow Matching, time scale 1000"
echo "Backbone: h128 L10 H4"
echo "Training batch: 128"
echo "Seed: 42"
echo "Generation: best.pt, Heun 50, seed 5678"
echo "============================================================"

git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

echo
echo "===== Regression tests for legacy training and 2-D quality metrics ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -m pytest \
        tests/unit/test_generation_benchmark.py::test_generation_benchmark_writes_isolated_run_directory \
        tests/unit/test_generation_benchmark.py::test_standardized_wasserstein_removes_channel_scale_difference \
        tests/unit/test_dinat_self_attention_ablation.py \
        -q

for run_dir in "$M33_RUN" "$M39_RUN"; do
    for artifact in best.pt history.csv summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
        test -s "${run_dir}/${artifact}" || {
            echo "ERROR: missing training artifact: ${run_dir}/${artifact}"
            exit 1
        }
    done
done

for gen_dir in "$M33_GEN" "$M39_GEN"; do
    for artifact in generated_test.pt summary.json; do
        test -s "${gen_dir}/${artifact}" || {
            echo "ERROR: missing generation artifact: ${gen_dir}/${artifact}"
            exit 1
        }
    done
done

rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

echo
echo "===== 1. Controlled training comparison ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.compare_dinat_self_attention_ablation \
        --baseline-run "$M33_RUN" \
        --candidate-run "$M39_RUN" \
        --output-dir "$TRAIN_OUT" \
        --overwrite

echo
echo "===== 2. Rebuild both generation benchmarks with identical current code ====="
for spec in "M33:$M33_GEN:$M33_REBENCH_ROOT" "M39:$M39_GEN:$M39_REBENCH_ROOT"; do
    IFS=: read -r label gen_dir bench_root <<< "$spec"
    echo
    echo "--- Rebenchmark $label ---"
    singularity exec \
        -B /home \
        -B /scratch \
        "$IMAGE" \
        "$PYTHON" -u -m scripts.benchmark_generation \
            run_dir="$gen_dir" \
            output_root="$bench_root" \
            overwrite=true \
            energy_spectrum.enabled=true \
            energy_spectrum.lower_quantile=0.10 \
            energy_spectrum.upper_quantile=0.90 \
            plots.energy_spectrum=true
done

for bench_dir in "$M33_BENCH" "$M39_BENCH"; do
    for artifact in \
        summary.json \
        channel_metrics.csv \
        standardized_wasserstein.csv \
        sample_statistics.csv \
        correlation_matrix.csv \
        spectra.csv \
        spectral_bands.csv \
        nearest_reference.csv \
        physical_metrics.csv \
        energy_spectrum_summary.csv \
        energy_spectral_bands.csv
    do
        test -s "${bench_dir}/${artifact}" || {
            echo "ERROR: missing rebuilt benchmark artifact: ${bench_dir}/${artifact}"
            exit 1
        }
    done
done

echo
echo "===== 3. Generation-quality relative comparison ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.compare_generation_benchmarks \
        baseline_dir="$M33_BENCH" \
        candidate_dir="$M39_BENCH" \
        output_root="$OUTPUT_ROOT" \
        comparison_name=generation_quality \
        overwrite=true

echo
echo "===== 4. Direct PDF and energy-spectrum plots ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.plot_dinat_self_attention_comparison \
        --m33-generation "${M33_GEN}/generated_test.pt" \
        --m39-generation "${M39_GEN}/generated_test.pt" \
        --m33-energy-summary "${M33_BENCH}/energy_spectrum_summary.csv" \
        --m39-energy-summary "${M39_BENCH}/energy_spectrum_summary.csv" \
        --output-dir "$PLOT_OUT"

echo
echo "===== 5. Controlled CUDA efficiency comparison ====="
nvidia-smi
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_dit_efficiency \
        --run "m33_no_self=$M33_RUN" \
        --run "m39_self=$M39_RUN" \
        --output-dir "$EFF_OUT" \
        --checkpoint best.pt \
        --device cuda \
        --batch-size 128 \
        --num-workers 0 \
        --forward-warmup 10 \
        --forward-steps 50 \
        --train-warmup 5 \
        --train-steps 20 \
        --generation-warmup 1 \
        --generation-repeats 3 \
        --sampling-steps 50 \
        --sampling-seed 5678 \
        --overwrite

for artifact in \
    "${TRAIN_OUT}/training_comparison.json" \
    "${TRAIN_OUT}/training_epoch_comparison.csv" \
    "${QUALITY_OUT}/summary.json" \
    "${QUALITY_OUT}/relative_comparison.csv" \
    "${PLOT_OUT}/energy_spectrum_comparison_mean.png" \
    "${PLOT_OUT}/energy_spectrum_comparison_median.png" \
    "${EFF_OUT}/efficiency_summary.json" \
    "${EFF_OUT}/efficiency_comparison.csv"
do
    test -s "$artifact" || {
        echo "ERROR: missing comparison artifact: $artifact"
        exit 1
    }
done

echo
echo "============================================================"
echo "M33 vs M39 COMPARISON COMPLETE"
echo "============================================================"
echo "Training:"
echo "  ${TRAIN_OUT}/training_comparison.json"
echo "Generation quality:"
echo "  ${QUALITY_OUT}/relative_comparison.csv"
echo "Plots:"
echo "  $PLOT_OUT"
echo "Efficiency:"
echo "  ${EFF_OUT}/efficiency_comparison.csv"
echo
echo "===== Training summary ====="
cat "${TRAIN_OUT}/training_comparison.json"
echo
echo "===== Generation relative metrics ====="
cat "${QUALITY_OUT}/relative_comparison.csv"
echo
echo "===== Efficiency comparison ====="
cat "${EFF_OUT}/efficiency_comparison.csv"
