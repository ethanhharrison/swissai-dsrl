#!/bin/bash
# Single training job for inference-delay experiments.
#
# Usage:
#   run_inference_delay.sh <env> <query_freq> <inference_delay> <multi_grad_step> <seed> [task_id] [extra args...]
#
# Optional env:
#   DSRL_CONDITION_ON_PREV_ACTIONS  If 1, pass --num_prev_actions equal to
#                                   inference_delay (last d played actions). Default: 0.
#   DSRL_RL_INFERENCE_DELAY         If > 0, SAC also sees s_{t-d'} with d' < inference_delay.
#   (sbatch sweeps use DSRL_RL_INFERENCE_DELAY_LIST, e.g. "0,5,10")
#
# Examples:
#   run_inference_delay.sh libero 5 2 20 0 28 --max_online_trajs 5000
#   run_inference_delay.sh aloha_cube 25 10 20 1
#   DSRL_CONDITION_ON_PREV_ACTIONS=1 run_inference_delay.sh aloha_cube 25 10 20 1
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
CONDITION_ON_PREV=${DSRL_CONDITION_ON_PREV_ACTIONS:-0}
RL_INFERENCE_DELAY=${DSRL_RL_INFERENCE_DELAY:-0}
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

NUM_PREV_ACTIONS=0
prev_action_args=()
if [ "$CONDITION_ON_PREV" -eq 1 ]; then
    if [ "$INFERENCE_DELAY" -le 0 ]; then
        echo "DSRL_CONDITION_ON_PREV_ACTIONS=1 requires inference_delay > 0" >&2
        exit 1
    fi
    NUM_PREV_ACTIONS=$INFERENCE_DELAY
    prev_action_args=(--num_prev_actions "$NUM_PREV_ACTIONS")
fi

echo "[run_inference_delay] env=$ENV query_freq=$QUERY_FREQ inference_delay=$INFERENCE_DELAY rl_inference_delay=$RL_INFERENCE_DELAY condition_on_prev=$CONDITION_ON_PREV num_prev_actions=$NUM_PREV_ACTIONS utd=$MULTI_GRAD_STEP seed=$SEED task_id=$TASK_ID"
echo "[run_inference_delay] extra args: $*"

suffix="qf${QUERY_FREQ}_id${INFERENCE_DELAY}"
if [ "$RL_INFERENCE_DELAY" -gt 0 ]; then
    suffix="${suffix}_rld${RL_INFERENCE_DELAY}"
fi
if [ "$CONDITION_ON_PREV" -eq 1 ]; then
    suffix="${suffix}_pca"
fi

rl_delay_args=()
if [ "$RL_INFERENCE_DELAY" -gt 0 ]; then
    if [ "$RL_INFERENCE_DELAY" -ge "$INFERENCE_DELAY" ]; then
        echo "DSRL_RL_INFERENCE_DELAY must be < inference_delay ($INFERENCE_DELAY)" >&2
        exit 1
    fi
    rl_delay_args=(--rl_inference_delay "$RL_INFERENCE_DELAY")
fi

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
    --suffix "$suffix" \
    --wandb_project "${proj_name}" \
    --batch_size 256 \
    --discount 0.999 \
    --max_steps "$max_steps" \
    --eval_interval 10000 \
    --log_interval 500 \
    --eval_episodes 50 \
    --start_online_updates "$start_online" \
    --resize_image 64 \
    --action_magnitude "$action_mag" \
    --query_freq "$QUERY_FREQ" \
    --inference_delay "$INFERENCE_DELAY" \
    "${rl_delay_args[@]}" \
    --hidden_dims 128 \
    --seed "$SEED" \
    --multi_grad_step "$MULTI_GRAD_STEP" \
    --iteration_size "$iteration_size" \
    "${libero_extra[@]}" \
    "$@" \
    "${prev_action_args[@]}"
