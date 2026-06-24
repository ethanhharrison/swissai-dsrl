#! /usr/bin/env python
import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs from https://github.com/huggingface/gym-aloha/tree/main?tab=readme-ov-file#-gpu-rendering-egl
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib

import jax
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
from gym.spaces import Dict, Box

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim import trajwise_alternating_training_loop
from examples.libero_multitask_env import MultiTaskLiberoEnv
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download

_jax_cache_dir = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.join(os.environ["HOME"], "jax_compilation_cache"),
)
compilation_cache.initialize_cache(_jax_cache_dir)

def _parse_task_ids(task_ids_arg):
    if task_ids_arg is None:
        return []
    if isinstance(task_ids_arg, (list, tuple)):
        return [int(t) for t in task_ids_arg]
    s = str(task_ids_arg).strip()
    if not s:
        return []
    parts = [p for p in s.replace(",", " ").split() if p]
    return [int(p) for p in parts]


class DummyEnv(gym.ObservationWrapper):

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(8, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32)


def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    multi_task_ids = _parse_task_ids(variant.task_ids)
    assert len(multi_task_ids) > 0, "Multi-task Libero90 training requires a non-empty --task_ids list"
    variant.multi_task_ids = multi_task_ids
    if len(multi_task_ids) <= 6:
        slug = "-".join(str(t) for t in multi_task_ids)
    else:
        slug = f"{len(multi_task_ids)}tasks_{multi_task_ids[0]}to{multi_task_ids[-1]}"
    variant.task_id = f"multi_{slug}"
    variant.prefix = f"{variant.prefix}_task{variant.task_id}_utd{variant.multi_grad_step}"

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
   
    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_90"]()
    env = MultiTaskLiberoEnv(
        task_suite=task_suite,
        task_ids=multi_task_ids,
        resolution=256,
        seed=variant.seed,
        rng_seed=variant.seed,
        mode="random",
    )
    eval_env = env
    first_task = task_suite.get_task(multi_task_ids[0])
    variant.task_description = first_task.language
    variant.env = 'libero'
    variant.env_max_reward = 1
    variant.max_timesteps = 400

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)

    config = openpi_config.get_config("pi05_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    variant.action_horizon = config.model.action_horizon
    variant.action_dim = config.model.action_dim
    variant.played_action_dim = 7
    assert variant.query_freq <= variant.action_horizon, (
        f"--query_freq={variant.query_freq} exceeds the model's action_horizon "
        f"({variant.action_horizon}). Lower --query_freq."
    )

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)

    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Loaded pi0 policy from %s", checkpoint_dir)
    agent = PixelSACLearner(
        variant.seed,
        sample_obs,
        sample_action,
        **kwargs,
    )

    online_buffer_size = variant.max_steps // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    trajwise_alternating_training_loop(
        variant, agent, env, eval_env,
        online_replay_buffer, replay_buffer, wandb_logger,
        shard_fn=shard_fn, agent_dp=agent_dp,
    )


def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension."""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )
