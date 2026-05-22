#!/bin/bash
# Single training job for inference-delay experiments.
#
# Usage:
#   run_inference_delay.sh <env> <query_freq> <inference_delay> <multi_grad_step> <seed> [task_id] [extra args...]
#
# Examples:
#   run_inference_delay.sh libero 5 2 20 0 28 --max_online_trajs 5000
#   run_inference_delay.sh aloha_cube 25 10 20 1
set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <env> <query_freq> <inference_delay> <multi_grad_step> <seed> [task_id] [extra args...]" >&2
    exit 1
fi

ENV=$1
QUERY_FREQ=$2
INFERENCE_DELAY=$3
MULTI_GRAD_STEP=$4
SEED=$5
shift 5

TASK_ID=${TASK_ID:-2}
if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    TASK_ID=$1
    shift 1
fi

proj_name=${DSRL_WANDB_PROJECT:-DSRL_pi0_InferenceDelay}
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

echo "[run_inference_delay] env=$ENV query_freq=$QUERY_FREQ inference_delay=$INFERENCE_DELAY utd=$MULTI_GRAD_STEP seed=$SEED task_id=$TASK_ID"
echo "[run_inference_delay] extra args: $*"

if [ "$ENV" = "libero" ]; then
    prefix=dsrl_pi0_libero
    action_mag=1.0
    max_steps=500000
    start_online=500
    iteration_size=${DSRL_ITERATION_SIZE:-0}
    libero_extra=(--task_id "$TASK_ID")
elif [ "$ENV" = "aloha_cube" ]; then
    prefix=dsrl_pi0_aloha
    action_mag=2.0
    max_steps=3000000
    start_online=1000
    iteration_size=${DSRL_ITERATION_SIZE:-0}
    libero_extra=(--target_entropy 0.0)
else
    echo "Unsupported env: $ENV (use libero or aloha_cube)" >&2
    exit 1
fi

python3 -m examples.launch_train_sim \
    --algorithm pixel_sac \
    --env "$ENV" \
    --prefix "$prefix" \
    --suffix "qf${QUERY_FREQ}_id${INFERENCE_DELAY}" \
    --wandb_project "${proj_name}" \
    --batch_size 256 \
    --discount 0.999 \
    --max_steps "$max_steps" \
    --eval_interval 10000 \
    --log_interval 500 \
    --eval_episodes 10 \
    --start_online_updates "$start_online" \
    --resize_image 64 \
    --action_magnitude "$action_mag" \
    --query_freq "$QUERY_FREQ" \
    --inference_delay "$INFERENCE_DELAY" \
    --hidden_dims 128 \
    --seed "$SEED" \
    --multi_grad_step "$MULTI_GRAD_STEP" \
    --iteration_size "$iteration_size" \
    "${libero_extra[@]}" \
    "$@"
