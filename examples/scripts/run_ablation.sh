#!/bin/bash
# Wrapper used by launch_ablation.py for BRC sbatch jobs.
# Usage: run_ablation.sh <iteration_size> <multi_grad_step> <seed>
# Any extra arguments ($4+) are forwarded verbatim to launch_train_sim.
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <iteration_size> <multi_grad_step> <seed> [task_id] [extra args...]" >&2
    exit 1
fi

ITER_SIZE=$1
MULTI_GRAD_STEP=$2
SEED=$3
shift 3
TASK_ID=${TASK_ID:-2}
# Optional 4th positional arg overrides $TASK_ID.
if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    TASK_ID=$1
    shift 1
fi

proj_name=${DSRL_WANDB_PROJECT:-DSRL_pi0_Libero_Ablation}
device_id=${CUDA_VISIBLE_DEVICES:-0}

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

# Redirect big caches to scratch (same as run_libero.sh).
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

MULTI_TASK_IDS=${DSRL_MULTI_TASK_IDS:-}
echo "[run_ablation] iter_size=$ITER_SIZE utd=$MULTI_GRAD_STEP seed=$SEED task_id=$TASK_ID multi_task_ids=${MULTI_TASK_IDS:-<unset>}"
echo "[run_ablation] extra args: $*"

libero_extra=()
if [ -n "$MULTI_TASK_IDS" ]; then
    libero_extra+=("--multi_task" "1" "--task_ids" "$MULTI_TASK_IDS")
else
    libero_extra+=("--task_id" "$TASK_ID")
fi

python3 -m examples.launch_train_sim \
    --algorithm pixel_sac \
    --env libero \
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
    --iteration_size "$ITER_SIZE" \
    "${libero_extra[@]}" \
    "$@"
