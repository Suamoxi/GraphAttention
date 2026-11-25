#!/bin/bash
#SBATCH --job-name=m49_pretrain
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err
set -euo pipefail
cd "${M49_REPO:-/scratch/coop/theret/GraphAttention}"
test -z "$(git status --porcelain)" || { echo "Dirty checkout"; exit 2; }
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
OUT="$ROOT/pretraining/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUT"
run() { singularity exec --nv -B /home -B /scratch "$IMAGE" "$PYTHON" "$@"; }
git rev-parse HEAD
nvidia-smi
case "$SLURM_ARRAY_TASK_ID" in
0) run -m pytest -q -rs tests/unit/test_triton_sparse_backward.py tests/unit/test_m47_segmented_conditioning.py tests/unit/test_m47_single_graph_broadcast.py tests/unit/test_m49_full_checkpointing.py tests/unit/test_generative_training_adapters.py tests/unit/test_flow_matching_runner.py --junitxml="$OUT/tests.xml";;
1) run -u -m scripts.probe_3d_max_batch --model full --sweep --max-batch 16 --output "$OUT/sweep.json";;
2) run -u -m scripts.probe_3d_max_batch --model full --checkpointing --sweep --max-batch 16 --output "$OUT/sweep.json";;
3) run -u -m scripts.probe_3d_max_batch --model dinat --backend torch_sparse --sweep --max-batch 16 --output "$OUT/sweep.json";;
4) run -u -m scripts.probe_3d_max_batch --model dinat --backend torch_sparse_triton --sweep --max-batch 64 --output "$OUT/sweep.json";;
5) run -u -m scripts.probe_3d_max_batch --model dinat --backend torch_sparse_triton --checkpointing --sweep --max-batch 64 --output "$OUT/sweep.json";;
*) exit 2;;
esac
