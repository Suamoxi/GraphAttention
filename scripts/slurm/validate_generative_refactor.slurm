#!/bin/bash
#SBATCH --job-name=validate_gen_refactor
#SBATCH --time=01:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
BASE_SHA=6a8bef69ce9d06b76569ebb1852693873d2af4f2
CANDIDATE_SHA=$(git -C "$REPO" rev-parse HEAD)

VALID_ROOT=/scratch/coop/theret/GraphAttention_refactor_validation/${SLURM_JOB_ID}
BASE_REPO=${VALID_ROOT}/repo_base
CANDIDATE_REPO=${VALID_ROOT}/repo_candidate
BASE_RUNS=${VALID_ROOT}/runs_base
CANDIDATE_RUNS=${VALID_ROOT}/runs_candidate

mkdir -p "$VALID_ROOT" "$BASE_RUNS" "$CANDIDATE_RUNS"

cleanup() {
    git -C "$REPO" worktree remove --force "$BASE_REPO" >/dev/null 2>&1 || true
    git -C "$REPO" worktree remove --force "$CANDIDATE_REPO" >/dev/null 2>&1 || true
}
trap cleanup EXIT

printf '%s\n' "============================================================"
printf '%s\n' "Generic generative runner reproduction validation"
printf 'Baseline SHA:  %s\n' "$BASE_SHA"
printf 'Candidate SHA: %s\n' "$CANDIDATE_SHA"
printf 'Artifacts:     %s\n' "$VALID_ROOT"
printf '%s\n' "============================================================"

git -C "$REPO" worktree add --detach "$BASE_REPO" "$BASE_SHA"
git -C "$REPO" worktree add --detach "$CANDIDATE_REPO" "$CANDIDATE_SHA"

run_training() {
    local repo_dir=$1
    local output_root=$2
    local module=$3
    local task=$4
    local run_name=$5

    cd "$repo_dir"
    singularity exec --nv \
        -B /home \
        -B /scratch \
        "$IMAGE" \
        "$PYTHON" -u -m "$module" \
            data=hit_slice_pt \
            geometry=dinat_exact2 \
            model=alternating_dilated_geometric_sparse_transformer \
            task="$task" \
            optimizer=adamw \
            +generative=hit_slice_diffusion \
            run_name="$run_name" \
            model.hidden_dim=128 \
            model.num_heads=4 \
            model.num_layers=10 \
            model.spatial_dim=2 \
            model.mlp_ratio=4 \
            generative.output_root="$output_root" \
            generative.batch_size=128 \
            generative.num_workers=4 \
            generative.max_epochs=2 \
            generative.device=cuda
}

M21_RUN=refactor_check_m21_linear_ddpm
M22_RUN=refactor_check_m22_edm

echo
echo "===== M21 baseline ====="
run_training \
    "$BASE_REPO" \
    "$BASE_RUNS" \
    scripts.train_slice_linear_diffusion \
    hit_diffusion_linear \
    "$M21_RUN"

echo
echo "===== M21 candidate ====="
run_training \
    "$CANDIDATE_REPO" \
    "$CANDIDATE_RUNS" \
    scripts.train_slice_linear_diffusion \
    hit_diffusion_linear \
    "$M21_RUN"

echo
echo "===== M22 baseline ====="
run_training \
    "$BASE_REPO" \
    "$BASE_RUNS" \
    scripts.train_slice_edm \
    hit_edm_diffusion \
    "$M22_RUN"

echo
echo "===== M22 candidate ====="
run_training \
    "$CANDIDATE_REPO" \
    "$CANDIDATE_RUNS" \
    scripts.train_slice_edm \
    hit_edm_diffusion \
    "$M22_RUN"

echo
echo "===== Compare M21 ====="
cd "$CANDIDATE_REPO"
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.compare_training_reproduction \
        "${BASE_RUNS}/${M21_RUN}" \
        "${CANDIDATE_RUNS}/${M21_RUN}" \
        --rtol 1e-6 \
        --atol 1e-7

echo
echo "===== Compare M22 ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.compare_training_reproduction \
        "${BASE_RUNS}/${M22_RUN}" \
        "${CANDIDATE_RUNS}/${M22_RUN}" \
        --rtol 1e-6 \
        --atol 1e-7

echo
echo "============================================================"
echo "GENERIC GENERATIVE RUNNER REPRODUCTION CHECK COMPLETE"
echo "$VALID_ROOT"
echo "============================================================"
