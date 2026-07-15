#!/bin/bash
# Launch Libero90 multi-task DSRL training.
# Usage: run_libero_multitask.sh <multi_grad_step> <seed> [extra args...]
#
# Set task ids via DSRL_MULTI_TASK_IDS, e.g.:
#   DSRL_MULTI_TASK_IDS=28,29,30,31,32 bash examples/scripts/run_libero_multitask.sh 20 0
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <multi_grad_step> <seed> [extra args...]" >&2
    exit 1
fi

MULTI_GRAD_STEP=$1
SEED=$2
shift 2

MULTI_TASK_IDS=${DSRL_MULTI_TASK_IDS:-}
if [ -z "$MULTI_TASK_IDS" ]; then
    echo "DSRL_MULTI_TASK_IDS must be set (comma-separated libero_90 task ids)" >&2
    exit 1
fi

proj_name=${DSRL_WANDB_PROJECT:-DSRL_pi0_Libero_MultiTask}
device_id=${CUDA_VISIBLE_DEVICES:-0}

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

: "${DSRL_SCRATCH:=/global/scratch/users/$USER}"
export OPENPI_DATA_HOME=$DSRL_SCRATCH/openpi_cache
export HF_HOME=$DSRL_SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$DSRL_SCRATCH/torch_cache
export JAX_COMPILATION_CACHE_DIR=$DSRL_SCRATCH/jax_compilation_cache
export PYTHONNOUSERSITE=1
mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$JAX_COMPILATION_CACHE_DIR"

export EXP=$DSRL_SCRATCH/logs/$proj_name
export XLA_PYTHON_CLIENT_PREALLOCATE=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LIBERO_CONFIG_PATH="$REPO_ROOT/LIBERO/.libero"
mkdir -p "$LIBERO_CONFIG_PATH"

echo "[run_libero_multitask] utd=$MULTI_GRAD_STEP seed=$SEED task_ids=$MULTI_TASK_IDS"
echo "[run_libero_multitask] extra args: $*"

python3 -m examples.launch_train_sim \
    --algorithm pixel_sac \
    --prefix dsrl_pi0_libero \
    --wandb_project "${proj_name}" \
    --batch_size 256 \
    --discount 0.999 \
    --max_steps 500000 \
    --eval_interval 10000 \
    --log_interval 500 \
    --eval_episodes 10 \
    --start_online_updates 500 \
    --resize_image 64 \
    --action_magnitude 1.0 \
    --query_freq 10 \
    --hidden_dims 128 \
    --seed "$SEED" \
    --multi_grad_step "$MULTI_GRAD_STEP" \
    --task_ids "$MULTI_TASK_IDS" \
    "$@"
