#! /usr/bin/env python
import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs from https://github.com/huggingface/gym-aloha/tree/main?tab=readme-ov-file#-gpu-rendering-egl
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib, copy

import jax
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
import gym_aloha
from gym.spaces import Dict, Box

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim import trajwise_alternating_training_loop, iteration_based_training_loop
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


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.rl_inference_delay > 0:
            obs_dict['rl_pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            if variant.env == 'libero':
                state_dim = 8
            elif variant.env == 'aloha_cube':
                state_dim = 14
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
            if variant.rl_inference_delay > 0:
                obs_dict['rl_state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        if variant.num_prev_actions > 0:
            prev_action_dim = variant.num_prev_actions * int(variant.played_action_dim)
            obs_dict['prev_action'] = Box(
                low=-np.inf,
                high=np.inf,
                shape=(prev_action_dim, 1),
                dtype=np.float32,
            )
        if variant.policy_mode == 'residual':
            exec_steps = int(variant.query_freq)
            played_dim = int(variant.played_action_dim)
            obs_dict['base_action'] = Box(low=-np.inf, high=np.inf, shape=(exec_steps, played_dim, 1), dtype=np.float32)
            if variant.residual_edit_mode != 'chunk':
                obs_dict['chunk_step'] = Box(low=0.0, high=1.0, shape=(exec_steps, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        if variant.policy_mode == 'residual':
            played_dim = int(variant.played_action_dim)
            if variant.residual_edit_mode == 'chunk':
                exec_steps = int(variant.query_freq)
                self.action_space = Box(low=-1, high=1, shape=(exec_steps, played_dim,), dtype=np.float32)
            else:
                self.action_space = Box(low=-1, high=1, shape=(1, played_dim,), dtype=np.float32)
        else:
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

    # Tag the prefix with the key ablation hyperparameters so the wandb run
    # name, group name, and output dir are self-describing. Seed is already
    # baked into expname by create_exp_name (as --s-{seed}).
    _ablation_tag_parts = []
    if variant.env == "libero":
        if variant.multi_task:
            multi_task_ids = _parse_task_ids(variant.task_ids)
            if len(multi_task_ids) == 0:
                raise ValueError(
                    "--multi_task=1 requires a non-empty --task_ids list"
                )
            variant.multi_task_ids = multi_task_ids
            if len(multi_task_ids) <= 6:
                slug = "-".join(str(t) for t in multi_task_ids)
            else:
                slug = f"{len(multi_task_ids)}tasks_{multi_task_ids[0]}to{multi_task_ids[-1]}"
            variant.task_id = f"multi_{slug}"
            _ablation_tag_parts.append(f"task{variant.task_id}")
        else:
            _ablation_tag_parts.append(f"task{variant.task_id}")
    if variant.iteration_size > 0:
        _ablation_tag_parts.append(f"iter{variant.iteration_size}")
    _ablation_tag_parts.append(f"utd{variant.multi_grad_step}")
    variant.prefix = f"{variant.prefix}_" + "_".join(_ablation_tag_parts)

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
   
    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)
    
    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["libero_90"]()
        if variant.multi_task:
            multi_task_ids = variant.multi_task_ids
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
        else:
            task_id = variant.task_id
            assert 0 <= task_id < task_suite.n_tasks, (
                f"--task_id={task_id} out of range for libero_90 "
                f"(0..{task_suite.n_tasks - 1})"
            )
            task = task_suite.get_task(task_id)
            env, task_description = _get_libero_env(task, 256, variant.seed)
            eval_env = env
            variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 400
    elif variant.env == 'aloha_cube':
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        eval_env = copy.deepcopy(env)
        variant.env_max_reward = 4
        variant.max_timesteps = 400
        

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)

    if variant.env == 'libero':
        config = openpi_config.get_config("pi05_libero")
        checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    elif variant.env == 'aloha_cube':
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    else:
        raise NotImplementedError()
    # pi05_libero has action_horizon=10; pi0_aloha_sim has action_horizon=50.
    # The noise tensor passed to pi0's infer() must match action_horizon exactly,
    # otherwise embed_suffix() produces an ar_mask that doesn't line up with the
    # action tokens and beartype raises a jaxtyping shape error. Stash the value
    # on variant so train_utils_sim can size the noise correctly.
    variant.action_horizon = config.model.action_horizon
    variant.action_dim = config.model.action_dim
    # Executed pi0 action dim after policy output transforms (not padded model dim).
    if variant.env == 'libero':
        variant.played_action_dim = 7
    elif variant.env == 'aloha_cube':
        variant.played_action_dim = 14
    else:
        raise NotImplementedError()
    # The training loop calls actions[t % query_freq] on an action chunk of
    # length action_horizon, so query_freq must not exceed action_horizon.
    # pi0_aloha_sim has horizon 50 (query_freq up to 50 OK); pi05_libero has
    # horizon 10, so `--query_freq 20` (the old pi0_libero default) silently
    # overruns the chunk and crashes ~10 steps into the first rollout.
    assert variant.query_freq <= variant.action_horizon, (
        f"--query_freq={variant.query_freq} exceeds the model's action_horizon "
        f"({variant.action_horizon}) for config '{variant.env}'. Lower --query_freq."
    )

    assert variant.inference_delay >= 0, (
        f"--inference_delay={variant.inference_delay} must be non-negative."
    )
    if variant.inference_delay > 0:
        max_chunk_index = variant.inference_delay + variant.query_freq
        assert max_chunk_index < variant.action_horizon, (
            f"--inference_delay={variant.inference_delay} with --query_freq="
            f"{variant.query_freq} requires action chunk indices up to "
            f"{max_chunk_index}, but action_horizon is only "
            f"{variant.action_horizon}. Lower --inference_delay or --query_freq."
        )

    assert variant.num_prev_actions >= 0, (
        f"--num_prev_actions={variant.num_prev_actions} must be non-negative."
    )
    if variant.inference_delay == 0:
        assert variant.num_prev_actions == 0, (
            f"--num_prev_actions={variant.num_prev_actions} must be 0 when "
            f"--inference_delay=0."
        )
    elif variant.num_prev_actions > 0:
        assert variant.num_prev_actions == variant.inference_delay, (
            f"--num_prev_actions={variant.num_prev_actions} must equal "
            f"--inference_delay={variant.inference_delay} (condition on the "
            f"last d played actions during the delay window)."
        )

    rl_inference_delay = int(variant.rl_inference_delay)
    assert rl_inference_delay >= 0, (
        f"--rl_inference_delay={rl_inference_delay} must be non-negative."
    )
    if rl_inference_delay > 0:
        assert variant.inference_delay > 0, (
            f"--rl_inference_delay={rl_inference_delay} requires --inference_delay > 0."
        )
        assert rl_inference_delay < variant.inference_delay, (
            f"--rl_inference_delay={rl_inference_delay} must be strictly less than "
            f"--inference_delay={variant.inference_delay}."
        )
    elif variant.inference_delay == 0:
        assert rl_inference_delay == 0, (
            f"--rl_inference_delay must be 0 when --inference_delay=0."
        )

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)

    assert variant.policy_mode in ('dsrl', 'residual'), (
        f"--policy_mode must be 'dsrl' or 'residual', got {variant.policy_mode!r}."
    )
    if variant.policy_mode == 'residual':
        assert variant.residual_edit_mode in ('step', 'chunk'), (
            f"--residual_edit_mode must be 'step' or 'chunk', "
            f"got {variant.residual_edit_mode!r}."
        )

    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Loaded pi0 policy from %s", checkpoint_dir)
    agent = PixelSACLearner(
        variant.seed,
        sample_obs,
        sample_action,
        policy_mode=variant.policy_mode,
        **kwargs,
    )

    online_buffer_size = variant.max_steps  // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    if variant.iteration_size > 0:
        iteration_based_training_loop(variant, agent, env, eval_env,
                                      online_replay_buffer, replay_buffer, wandb_logger,
                                      shard_fn=shard_fn, agent_dp=agent_dp)
    else:
        trajwise_alternating_training_loop(variant, agent, env, eval_env,
                                           online_replay_buffer, replay_buffer, wandb_logger,
                                           shard_fn=shard_fn, agent_dp=agent_dp)
 