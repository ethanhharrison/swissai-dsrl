<div align="center">

# DSRL for π₀: Diffusion Steering via Reinforcement Learning

## [[website](https://diffusion-steering.github.io)]      [[paper](https://arxiv.org/abs/2506.15799)]

</div>


## Overview
This repository provides the official implementation for our paper: [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799) (CoRL 2025).

Specifically, it contains a JAX-based implementation of DSRL (Diffusion Steering via Reinforcement Learning) for steering a pre-trained generalist policy, [π₀](https://github.com/Physical-Intelligence/openpi), on **Libero90 multi-task** simulation environments.

If you find this repository useful for your research, please cite:

```
@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

## Installation
1. Create a conda environment:
```
conda create -n dsrl_pi0 python=3.11.11
conda activate dsrl_pi0
```

2. Clone this repo with all submodules
```
git clone git@github.com:nakamotoo/dsrl_pi0.git --recurse-submodules
cd dsrl_pi0
```

3. Install all packages and dependencies
```
# Install the jaxrl2 package in this repo (editable)
pip install -e .

# Install all python dependencies in one shot. `constraints.txt` pins the
# handful of packages (torch, tensorstore) that would otherwise conflict
# across dsrl_pi0, openpi and LIBERO.
pip install -r requirements.txt -c constraints.txt

# Install jax CUDA plugins (jax itself is pinned in requirements.txt).
# `constraints.txt` pins the full `nvidia-*-cu12` stack to the CUDA 12.6
# wheels, which are forward-compatible with NVIDIA driver >= 525. Without
# this pin, jax 0.5.3 pulls CUDA 12.8 wheels + cuDNN 9.10, which require
# driver >= 555 and fail with CUDNN_STATUS_INTERNAL_ERROR on older
# clusters (e.g. driver 545). If your driver is 555+ you can drop the
# cu12 pins from constraints.txt.
pip install "jax[cuda12]==0.5.3" -c constraints.txt

# Install openpi and openpi-client with --no-deps. The upstream openpi
# pyproject relies on `uv` features (local path + git sources, override
# dependencies) that pip cannot resolve, which triggers pip's
# `resolution-too-deep` error. We already installed the real runtime
# dependencies above, so --no-deps is safe here.
pip install -e openpi --no-deps
pip install -e openpi/packages/openpi-client --no-deps

# Install lerobot from the pinned git rev (same commit as openpi's
# uv.sources entry). openpi.training.data_loader imports lerobot at module
# scope, so it must be importable even for inference-only runs. Use
# --no-deps: lerobot's full dep set (gymnasium==0.29.1, mujoco<3, etc.)
# conflicts with this project's envs; requirements.txt already brings in
# the subset that fires on `import lerobot.common.datasets.lerobot_dataset`.
pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"

# Install LIBERO. `editable_mode=compat` is required because LIBERO's
# setup.py uses find_packages() but ships its sources under
# LIBERO/libero/libero/ (with no LIBERO/libero/__init__.py). The default
# PEP 660 editable install therefore registers zero packages; the
# `compat` mode falls back to a `.pth` file that puts LIBERO/ on sys.path
# so `import libero.libero` resolves via PEP 420 namespace packages.
pip install -e LIBERO --config-settings editable_mode=compat

# LIBERO has been validated against torch 2.6.0 CPU wheels. Install the
# matching torchvision CPU build too -- PyPI's torchvision 0.21 is linked
# against CUDA torch and triggers `operator torchvision::nms does not exist`
# when loaded alongside torch 2.6.0+cpu. transformers imports torchvision
# eagerly via image_utils, so this must resolve at import time.
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

## Cluster storage / disk quota
The pretrained π₀ checkpoints are ~14 GB each and HuggingFace + JAX kernel
caches grow fast, so they must not live under a quota'd `$HOME` (Savio gives
30 GB; most shared clusters are similar). The `examples/scripts/run_*.sh`
scripts redirect every cache-producing library at `$DSRL_SCRATCH`, which
defaults to `/global/scratch/users/$USER` (Savio). Override the default if
your cluster uses a different scratch path:
```bash
export DSRL_SCRATCH=/path/to/your/scratch
```
This redirects, in a single place:
- `OPENPI_DATA_HOME` — π₀ checkpoints downloaded from `gs://openpi-assets`
- `HF_HOME` / `TRANSFORMERS_CACHE` / `HF_DATASETS_CACHE` — HuggingFace models & datasets (lerobot, PaliGemma tokenizer, etc.)
- `TORCH_HOME` — torch.hub downloads
- `JAX_COMPILATION_CACHE_DIR` — JAX compiled kernels (grows to several GB)
- `EXP` — per-run training logs, checkpoints, and videos

The scripts also set `PYTHONNOUSERSITE=1` so stale packages in `~/.local/lib/python3.11/site-packages` can't shadow the conda environment (this was the
root cause of the earlier CUDA 12.8 vs. driver 545 cuDNN failure).

If you're hitting quota even before training starts, the usual offenders in
`$HOME` are:
- `~/.cache/pip` — purge and move with `pip cache purge && pip config set global.cache-dir $DSRL_SCRATCH/pip_cache`
- `~/miniconda3` — move the whole install to scratch and re-symlink, or install miniconda directly to scratch next time
- `~/.local/lib/python3.11/site-packages` — stale user-site packages; safe to archive to scratch once `PYTHONNOUSERSITE=1` is in place

## Training (Libero90 Multi-Task)
Set the task ids and launch training:
```
DSRL_MULTI_TASK_IDS=28,29,30,31,32 bash examples/scripts/run_libero_multitask.sh 20 0
```
Or submit Slurm jobs (Savio):
```
bash sbatch_multitask_28-32_all.sh
bash sbatch_multitask_28-79-44-59-43_all.sh
```
### Training Logs
We provide sample W&B runs and logs: https://wandb.ai/mitsuhiko/DSRL_pi0_public

## Credits
This repository is built upon [jaxrl2](https://github.com/ikostrikov/jaxrl2) and [PTR](https://github.com/Asap7772/PTR) repositories. 
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at nakamoto\[at\]berkeley\[dot\]edu 
