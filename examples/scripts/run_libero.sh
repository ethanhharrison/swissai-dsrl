#!/bin/bash
proj_name=DSRL_pi0_Libero
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

# Large caches (pretrained pi0 checkpoints ~14 GB each, HF model weights,
# JAX kernel cache) must not live under $HOME on a quota'd cluster. Point
# them at scratch. Override DSRL_SCRATCH if your site uses a different path.
: "${DSRL_SCRATCH:=/global/scratch/users/$USER}"
export OPENPI_DATA_HOME=$DSRL_SCRATCH/openpi_cache
export HF_HOME=$DSRL_SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$DSRL_SCRATCH/torch_cache
export JAX_COMPILATION_CACHE_DIR=$DSRL_SCRATCH/jax_compilation_cache
# Ignore $HOME/.local/lib user-site so stale packages there can't shadow the
# conda env (this is what caused the CUDA 12.8 ↔ driver 545 cuDNN failure).
export PYTHONNOUSERSITE=1
mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$JAX_COMPILATION_CACHE_DIR"

export EXP=$DSRL_SCRATCH/logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

pip install mujoco==3.3.1

python3 -m examples.launch_train_sim \
--algorithm pixel_sac \
--env libero \
--prefix dsrl_pi0_libero \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 500000  \
--eval_interval 10000 \
--log_interval 500 \
--eval_episodes 10 \
--multi_grad_step 20 \
--start_online_updates 500 \
--resize_image 64 \
--action_magnitude 1.0 \
--query_freq 10 \
--hidden_dims 128 \
--task_id 2 \
--iteration_size 100 \