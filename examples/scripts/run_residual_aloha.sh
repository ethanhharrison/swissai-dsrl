#!/bin/bash
# Single ALOHA residual-policy training job.
#
# Usage:
#   run_residual_aloha.sh <query_freq> <multi_grad_step> <seed> [extra args...]
#
# Residual edit mode (default in this script: chunk):
#   --residual_edit_mode chunk  one SAC output per query interval (query_freq x action_dim)
#   --residual_edit_mode step   one SAC output per env step (1 x action_dim)
#
# Fixed SAC temperature (alpha): set --init_temperature and keep --temp_lr 0.
# Override via extra args, e.g. --init_temperature 0.1
#
# Examples:
#   run_residual_aloha.sh 25 25 3 
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <query_freq> <multi_grad_step> <seed> [extra args...]" >&2
    exit 1
fi

QUERY_FREQ=$1
MULTI_GRAD_STEP=$2
SEED=$3
shift 3

proj_name=${DSRL_WANDB_PROJECT:-DSRL_pi0_Residual_Aloha}
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

iteration_size=${DSRL_ITERATION_SIZE:-0}

echo "[run_residual_aloha] query_freq=$QUERY_FREQ utd=$MULTI_GRAD_STEP seed=$SEED"
echo "[run_residual_aloha] extra args: $*"

suffix="qf${QUERY_FREQ}_residual"

python3 -m examples.launch_train_sim \
    --algorithm pixel_sac \
    --env aloha_cube \
    --prefix dsrl_pi0_aloha_residual \
    --suffix "$suffix" \
    --wandb_project "${proj_name}" \
    --policy_mode residual \
    --residual_edit_mode step \
    --batch_size 256 \
    --discount 0.999 \
    --max_steps 3000000 \
    --eval_interval 10000 \
    --log_interval 500 \
    --eval_episodes 10 \
    --start_online_updates 25000 \
    --resize_image 64 \
    --action_magnitude 0.05 \
    --query_freq "$QUERY_FREQ" \
    --inference_delay 0 \
    --hidden_dims 128 \
    --init_temperature 0.2 \
    --temp_lr 0 \
    --seed "$SEED" \
    --multi_grad_step "$MULTI_GRAD_STEP" \
    --iteration_size "$iteration_size" \
    "$@"
