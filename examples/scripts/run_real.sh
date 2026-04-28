#!/bin/bash
proj_name=DSRL_pi0_FrankaDroid
device_id=0

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

# Fill inFranka Droid camera IDs
export LEFT_CAMERA_ID=""
export RIGHT_CAMERA_ID=""
export WRIST_CAMERA_ID=""

# Fill inpi0 remote host and port
export remote_host=""
export remote_port=""


python3 -m examples.launch_train_real \
--algorithm pixel_sac \
--env franka_droid \
--prefix dsrl_pi0_real \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.99 \
--seed 0 \
--max_steps 500000  \
--eval_interval 2000 \
--log_interval 100 \
--multi_grad_step 30 \
--resize_image 128 \
--action_magnitude 2.5 \
--query_freq 10 \
--hidden_dims 1024 \
--num_qs 2 