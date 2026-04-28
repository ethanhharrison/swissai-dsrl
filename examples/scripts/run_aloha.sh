#!/bin/bash
proj_name=DSRL_pi0_Aloha
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

# See run_libero.sh for the rationale behind the scratch-redirection block.
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
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false


pip install mujoco==2.3.7

python3 -m examples.launch_train_sim \
--algorithm pixel_sac \
--env aloha_cube \
--prefix dsrl_pi0_aloha \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 3000000  \
--eval_interval 10000 \
--log_interval 500 \
--eval_episodes 10 \
--multi_grad_step 20 \
--start_online_updates 1000 \
--resize_image 64 \
--action_magnitude 2.0 \
--query_freq 50 \
--hidden_dims 128 \
--target_entropy 0.0 