#!/bin/bash
#SBATCH --job-name=m43c_pos_2d
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err
set -euo pipefail

# New 2-D runs only. No writes to 3-D M43-B checkpoints.
REPO=${M43C_REPO:-/scratch/coop/theret/GraphAttention_m43c_coords}
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
OUT_ROOT=${M43C_OUTPUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m43c_coordinate_2d}
EPOCHS=${M43C_EPOCHS:-100}
VARIANT=${M43C_VARIANT:-}
STAGE=${M43C_STAGE:-tests}

case "$VARIANT" in
  absolute) POSITION=true ;;
  relative) POSITION=false ;;
  "") if [ "$STAGE" = tests ]; then POSITION=true; else
        echo "Set M43C_VARIANT=absolute or relative" >&2; exit 2; fi ;;
  *) echo "Invalid M43C_VARIANT=$VARIANT" >&2; exit 2 ;;
esac
case "$STAGE" in tests|train|generate|benchmark|evaluate) ;; *)
  echo "Set M43C_STAGE=tests,train,generate,benchmark,evaluate" >&2; exit 2 ;;
esac
if [ "$EPOCHS" -lt 1 ] || [ "$EPOCHS" -gt 1000 ]; then
  echo "M43C_EPOCHS must lie in [1,1000]" >&2; exit 2
fi
cd "$REPO"
test -z "$(git status --porcelain)" || {
  echo "Dirty worktree" >&2; exit 2;
}
export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"
RUN_NAME=m43c_${VARIANT}_s42_e${EPOCHS}
RUN_DIR="$OUT_ROOT/$RUN_NAME"
GEN_NAME=flow_heun_steps50_seed5678
GEN_DIR="$RUN_DIR/generations/$GEN_NAME"
BENCH_ROOT="$OUT_ROOT/benchmark_$VARIANT"
BENCH_DIR="$BENCH_ROOT/$GEN_NAME"
mkdir -p /scratch/coop/theret/GraphAttention
echo "M43C $(git rev-parse HEAD) stage=$STAGE variant=$VARIANT"
echo "M33-matched 2-D DiNAT-DiT H128 L10 H4 scatter, absolute=$POSITION"
echo "Training run $RUN_DIR"

singularity exec --nv -B /home -B /scratch "$IMAGE" \
  "$PYTHON" -m pytest -q \
    tests/unit/test_m43c_relative_coordinates.py \
    tests/unit/test_dinat_dit.py::test_dinat_dit_alternates_local_and_exact_two_hop_topologies
if [ "$STAGE" = tests ]; then exit 0; fi
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ "$STAGE" = train ]; then
  if [ -e "$RUN_DIR" ]; then
    echo "Refusing to overwrite $RUN_DIR" >&2; exit 2
  fi
  singularity exec --nv -B /home -B /scratch "$IMAGE" \
    "$PYTHON" -u -m scripts.train_generative \
    data=hit_slice_pt geometry=dinat_exact2 model=dinat_dit \
    task=hit_flow_matching optimizer=adamw \
    +generative=hit_slice_flow_matching \
    seed=42 run_name="$RUN_NAME" task.time_embedding_scale=1000.0 \
    model.hidden_dim=128 model.num_heads=4 model.num_layers=10 \
    model.spatial_dim=2 model.mlp_ratio=4 model.condition_embed_dim=128 \
    model.dropout=0.0 model.qkv_bias=true model.out_proj_bias=false \
    model.sparse_attention_backend=scatter model.include_self_attention=false \
    model.use_absolute_position_encoding="$POSITION" \
    generative.output_root="$OUT_ROOT" generative.batch_size=128 \
    generative.num_workers=4 generative.max_epochs="$EPOCHS" generative.device=cuda
  for f in best.pt last.pt history.csv summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "$RUN_DIR/$f" || { echo "Missing $RUN_DIR/$f" >&2; exit 3; }
  done
  exit 0
fi

# Evaluation keeps trained models and all scientific inputs immutable.
# Each stage is separately rerunnable. Existing COMPLETE stages are reused;
# partial outputs are never deleted or overwritten.
for f in best.pt summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
  test -s "$RUN_DIR/$f" || { echo "Missing trained artifact $RUN_DIR/$f" >&2; exit 3; }
done
case "$POSITION" in
 true) EXPECTED=true ;;
 false) EXPECTED=false ;;
esac
# The checkpoint is selected from the fully completed scientific run.
# Protect against accidentally evaluating two copies of the same variant.
if ! grep -Eq "^[[:space:]]*use_absolute_position_encoding: $EXPECTED([[:space:]]*$|[[:space:]]+#)" \
  "$RUN_DIR/resolved_config.yaml"; then
  echo "Wrong/missing absolute position flag in $RUN_DIR/resolved_config.yaml" >&2
  exit 3
fi

if [ "$STAGE" = generate ] || [ "$STAGE" = evaluate ]; then
  if [ -d "$GEN_DIR" ]; then
    for f in generated_test.pt summary.json generation_config.yaml resolved_config.yaml generation_wall_seconds.txt; do
      test -s "$GEN_DIR/$f" || {
        echo "Incomplete generation $GEN_DIR: missing $f; refusing overwrite" >&2
        exit 3
      }
    done
    echo "Reusing complete previous generation: $GEN_DIR"
  else
    echo "== Generate $VARIANT held-out 2-D samples: Heun-50 seed 5678 =="
    start=$(date +%s%N)
    singularity exec --nv -B /home -B /scratch "$IMAGE" \
      "$PYTHON" -u -m scripts.generate_generative \
      run_dir="$RUN_DIR" checkpoint=best.pt device=cuda batch_size=128 num_workers=4 \
      sampling.steps=50 sampling.method=flow_ode sampling.solver=heun \
      sampling.sampling_eps=0.0 sampling.final_denoise=false sampling.seed=5678 \
      output_name="$GEN_NAME" overwrite=false
    end=$(date +%s%N)
    test -s "$GEN_DIR/generated_test.pt" || exit 3
    # The generation runner writes into a new directory. Timing is stored
    # only after it returns successfully and is never fabricated on reruns.
    awk "BEGIN {printf \"%.6f\\n\", ($end-$start)/1000000000.0}" \
      > "$GEN_DIR/generation_wall_seconds.txt"
    echo "Generation finished: $GEN_DIR"
  fi
fi
if [ "$STAGE" = generate ]; then exit 0; fi

for f in generated_test.pt summary.json generation_config.yaml resolved_config.yaml generation_wall_seconds.txt; do
  test -s "$GEN_DIR/$f" || {
    echo "Missing generation artifact $GEN_DIR/$f; run stage=generate first" >&2
    exit 3
  }
done
if [ -d "$BENCH_DIR" ]; then
  for f in summary.json channel_metrics.csv standardized_wasserstein.csv \
           spectral_bands.csv physical_metrics.csv energy_spectrum_summary.csv; do
    test -s "$BENCH_DIR/$f" || {
      echo "Incomplete benchmark $BENCH_DIR: missing $f; refusing overwrite" >&2
      exit 3
    }
  done
  echo "Reusing complete previous benchmark: $BENCH_DIR"
  exit 0
fi

echo "== Benchmark $VARIANT held-out generated distribution =="
singularity exec --nv -B /home -B /scratch "$IMAGE" \
  "$PYTHON" -u -m scripts.benchmark_generation \
  run_dir="$GEN_DIR" output_root="$BENCH_ROOT" overwrite=false \
  energy_spectrum.enabled=true energy_spectrum.lower_quantile=0.10 \
  energy_spectrum.upper_quantile=0.90 plots.energy_spectrum=true
for f in summary.json channel_metrics.csv standardized_wasserstein.csv \
         spectral_bands.csv physical_metrics.csv energy_spectrum_summary.csv; do
  test -s "$BENCH_DIR/$f" || { echo "Missing $BENCH_DIR/$f" >&2; exit 3; }
done
echo "Evaluation complete for $VARIANT: $BENCH_DIR"
