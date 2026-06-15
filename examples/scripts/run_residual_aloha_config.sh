#!/bin/bash
# Configurable ALOHA residual-policy training job.
#
# Usage:
#   run_residual_aloha_config.sh \
#       <query_freq> <multi_grad_step> <seed> \
#       <action_magnitude> <inference_delay> <rl_inference_delay> <num_prev_actions> \
#       [extra args...]
#
# Optional env:
#   DSRL_RESIDUAL_EDIT_MODE   step | chunk (default: chunk)
#   DSRL_TEMP_MODE            fixed | learned (default: fixed)
#   DSRL_INIT_TEMPERATURE     Initial alpha (default: 0.2 if fixed, 1.0 if learned)
#   DSRL_TARGET_ENTROPY       Used when DSRL_TEMP_MODE=learned: auto or a float
#                             (default: auto)
#   DSRL_CONDITION_ON_PREV_ACTIONS
#                             If 1, overrides num_prev_actions to inference_delay.
#   DSRL_WANDB_PROJECT        W&B project name
#   DSRL_ITERATION_SIZE       Passed through as --iteration_size
#
# Temperature modes:
#   fixed   --temp_lr 0 with DSRL_INIT_TEMPERATURE (static alpha)
#   learned --temp_lr 3e-4 with DSRL_TARGET_ENTROPY (adaptive alpha)
#
# Examples:
#   run_residual_aloha_config.sh 25 1 3 0.05 0 0 0
#   run_residual_aloha_config.sh 25 1 3 0.05 10 5 10
#   DSRL_TEMP_MODE=learned DSRL_TARGET_ENTROPY=0.0 \
#       run_residual_aloha_config.sh 25 1 3 0.05 0 0 0
#   DSRL_TEMP_MODE=fixed DSRL_INIT_TEMPERATURE=0.1 \
#       run_residual_aloha_config.sh 25 25 3 0.05 0 0 0
set -euo pipefail

if [ "$#" -lt 7 ]; then
    echo "Usage: $0 <query_freq> <multi_grad_step> <seed> \\" >&2
    echo "       <action_magnitude> <inference_delay> <rl_inference_delay> <num_prev_actions> \\" >&2
    echo "       [extra args...]" >&2
    exit 1
fi

QUERY_FREQ=$1
MULTI_GRAD_STEP=$2
SEED=$3
ACTION_MAGNITUDE=$4
INFERENCE_DELAY=$5
RL_INFERENCE_DELAY=$6
NUM_PREV_ACTIONS=$7
shift 7

RESIDUAL_EDIT_MODE=${DSRL_RESIDUAL_EDIT_MODE:-chunk}
TEMP_MODE=${DSRL_TEMP_MODE:-fixed}
CONDITION_ON_PREV=${DSRL_CONDITION_ON_PREV_ACTIONS:-0}

if [ "$CONDITION_ON_PREV" -eq 1 ]; then
    if [ "$INFERENCE_DELAY" -le 0 ]; then
        echo "DSRL_CONDITION_ON_PREV_ACTIONS=1 requires inference_delay > 0" >&2
        exit 1
    fi
    NUM_PREV_ACTIONS=$INFERENCE_DELAY
fi

if [ "$NUM_PREV_ACTIONS" -ne 0 ] && [ "$NUM_PREV_ACTIONS" -ne "$INFERENCE_DELAY" ]; then
    echo "num_prev_actions must be 0 or equal to inference_delay ($INFERENCE_DELAY), got $NUM_PREV_ACTIONS" >&2
    exit 1
fi

if [ "$RL_INFERENCE_DELAY" -gt 0 ]; then
    if [ "$INFERENCE_DELAY" -le 0 ]; then
        echo "rl_inference_delay > 0 requires inference_delay > 0" >&2
        exit 1
    fi
    if [ "$RL_INFERENCE_DELAY" -ge "$INFERENCE_DELAY" ]; then
        echo "rl_inference_delay must be < inference_delay ($INFERENCE_DELAY)" >&2
        exit 1
    fi
fi

case "$RESIDUAL_EDIT_MODE" in
    step|chunk) ;;
    *)
        echo "DSRL_RESIDUAL_EDIT_MODE must be 'step' or 'chunk', got '$RESIDUAL_EDIT_MODE'" >&2
        exit 1
        ;;
esac

case "$TEMP_MODE" in
    fixed)
        INIT_TEMPERATURE=${DSRL_INIT_TEMPERATURE:-0.2}
        TEMP_LR=0
        target_entropy_args=()
        ;;
    learned)
        INIT_TEMPERATURE=${DSRL_INIT_TEMPERATURE:-1.0}
        TEMP_LR=3e-4
        TARGET_ENTROPY=${DSRL_TARGET_ENTROPY:-auto}
        target_entropy_args=(--target_entropy "$TARGET_ENTROPY")
        ;;
    *)
        echo "DSRL_TEMP_MODE must be 'fixed' or 'learned', got '$TEMP_MODE'" >&2
        exit 1
        ;;
esac

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

suffix="qf${QUERY_FREQ}_residual_${RESIDUAL_EDIT_MODE}_am${ACTION_MAGNITUDE}_id${INFERENCE_DELAY}"
if [ "$RL_INFERENCE_DELAY" -gt 0 ]; then
    suffix="${suffix}_rld${RL_INFERENCE_DELAY}"
fi
if [ "$NUM_PREV_ACTIONS" -gt 0 ]; then
    suffix="${suffix}_pca${NUM_PREV_ACTIONS}"
fi
if [ "$TEMP_MODE" = "fixed" ]; then
    suffix="${suffix}_fixt${INIT_TEMPERATURE}"
else
    suffix="${suffix}_lt${TARGET_ENTROPY}"
fi

rl_delay_args=()
if [ "$RL_INFERENCE_DELAY" -gt 0 ]; then
    rl_delay_args=(--rl_inference_delay "$RL_INFERENCE_DELAY")
fi

prev_action_args=()
if [ "$NUM_PREV_ACTIONS" -gt 0 ]; then
    prev_action_args=(--num_prev_actions "$NUM_PREV_ACTIONS")
fi

echo "[run_residual_aloha_config] query_freq=$QUERY_FREQ utd=$MULTI_GRAD_STEP seed=$SEED"
echo "[run_residual_aloha_config] action_magnitude=$ACTION_MAGNITUDE inference_delay=$INFERENCE_DELAY rl_inference_delay=$RL_INFERENCE_DELAY num_prev_actions=$NUM_PREV_ACTIONS"
echo "[run_residual_aloha_config] residual_edit_mode=$RESIDUAL_EDIT_MODE temp_mode=$TEMP_MODE init_temperature=$INIT_TEMPERATURE temp_lr=$TEMP_LR"
if [ "$TEMP_MODE" = "learned" ]; then
    echo "[run_residual_aloha_config] target_entropy=$TARGET_ENTROPY"
fi
echo "[run_residual_aloha_config] suffix=$suffix"
echo "[run_residual_aloha_config] extra args: $*"

python3 -m examples.launch_train_sim \
    --algorithm pixel_sac \
    --env aloha_cube \
    --prefix dsrl_pi0_aloha_residual \
    --suffix "$suffix" \
    --wandb_project "${proj_name}" \
    --policy_mode residual \
    --residual_edit_mode "$RESIDUAL_EDIT_MODE" \
    --batch_size 256 \
    --discount 0.999 \
    --max_steps 3000000 \
    --eval_interval 10000 \
    --log_interval 500 \
    --eval_episodes 10 \
    --start_online_updates 1000 \
    --resize_image 64 \
    --action_magnitude "$ACTION_MAGNITUDE" \
    --query_freq "$QUERY_FREQ" \
    --inference_delay "$INFERENCE_DELAY" \
    "${rl_delay_args[@]}" \
    --hidden_dims 128 \
    --init_temperature "$INIT_TEMPERATURE" \
    --temp_lr "$TEMP_LR" \
    "${target_entropy_args[@]}" \
    --seed "$SEED" \
    --multi_grad_step "$MULTI_GRAD_STEP" \
    --iteration_size "$iteration_size" \
    "$@" \
    "${prev_action_args[@]}"
