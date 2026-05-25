from tqdm import tqdm
import numpy as np
import wandb
import jax
from openpi_client import image_tools
import math
import PIL

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def obs_to_img(obs, variant):
    '''
    Convert raw observation to resized image for DSRL actor/critic
    '''
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == 'aloha_cube':
        curr_image = obs["pixels"]["top"]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0: 
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image

def obs_to_pi_zero_input(obs, variant):
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        
        obs_pi_zero = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(variant.task_description),
                    }
    elif variant.env == 'aloha_cube':
        img = np.ascontiguousarray(obs["pixels"]["top"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        obs_pi_zero = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2,0,1))}
        }
    else:
        raise NotImplementedError()
    return obs_pi_zero

def obs_to_qpos(obs, variant):
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env == 'aloha_cube':
        qpos = obs["agent_pos"]
    else:
        raise NotImplementedError()
    return qpos

def _prev_action_obs(played_action_history, variant):
    """Build prev_action observation from the last d executed pi0 actions."""
    num_prev = variant.num_prev_actions
    action_dim = variant.played_action_dim
    if len(played_action_history) == 0:
        hist = np.zeros((num_prev, action_dim), dtype=np.float32)
    else:
        recent = [np.asarray(a, dtype=np.float32).reshape(-1)[:action_dim]
            for a in played_action_history[-num_prev:]]
        if len(recent) < num_prev:
            pad = [np.zeros(action_dim, dtype=np.float32)] * (num_prev - len(recent))
            recent = pad + recent
        hist = np.stack(recent, axis=0)
    hist_flat = hist.reshape(-1).astype(np.float32)
    return {
        'prev_action': hist_flat[np.newaxis, :, np.newaxis],
    }


def _conditioning_timestep(t, inference_delay):
    """Env step whose observation conditions the policy query at step ``t``."""
    if t < inference_delay:
        return t
    return t - inference_delay


def _chunk_action_index(t, inference_delay, query_freq):
    """Index into the current action chunk for env step ``t``."""
    if t < query_freq:
        return t
    if t < inference_delay:
        return t % query_freq
    return inference_delay + (t % query_freq)


def _raw_obs_to_obs_dict(raw_obs, variant):
    curr_image = obs_to_img(raw_obs, variant)
    qpos = obs_to_qpos(raw_obs, variant)
    if variant.add_states:
        return {
            'pixels': curr_image[np.newaxis, ..., np.newaxis],
            'state': qpos[np.newaxis, ..., np.newaxis],
        }
    return {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
    }


def _infer_action_chunk(variant, agent, agent_dp, rng, obs_dict, raw_obs, i, is_eval):
    """Run pi0 inference and return (actions, actions_noise, updated_rng)."""
    rng, key = jax.random.split(rng)
    obs_pi_zero = obs_to_pi_zero_input(raw_obs, variant)
    action_horizon = variant.action_horizon
    action_dim = variant.action_dim
    if i == 0 and is_eval:
        noise = jax.random.normal(rng, (1, action_horizon, action_dim))
        actions_noise = None
    elif i == 0:
        noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
        noise_repeat = jax.numpy.repeat(
            noise[:, -1:, :], action_horizon - noise.shape[1], axis=1
        )
        noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
        actions_noise = noise[0, :agent.action_chunk_shape[0], :]
    else:
        actions_noise = agent.sample_actions(obs_dict)
        actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
        noise = np.repeat(
            actions_noise[-1:, :],
            action_horizon - actions_noise.shape[0],
            axis=0,
        )
        noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]
    actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
    return actions, actions_noise, rng


def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    total_env_steps = 0
    i = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
    
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            traj = collect_traj(variant, agent, env, i, agent_dp)
            traj_id = online_replay_buffer._traj_counter
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            print('total env steps:', total_env_steps)
            
            num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step

            if len(online_replay_buffer) > variant.start_online_updates:
                for _ in range(num_gradsteps):
                    # perform first visualization before updating
                    if i == 0:
                        print('performing evaluation for initial checkpoint')
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # online perform update once we have some amount of online trajs
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1
                        

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        # wandb_logger.log({'replay_buffer_size': len(online_replay_buffer)}, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

            
def iteration_based_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                   perform_control_evals=True, shard_fn=None, agent_dp=None):
    """Collect-then-train loop with non-overlapping phases.

    Each outer iteration:
      1. Collect `variant.iteration_size` trajectories into the replay buffer.
      2. Run exactly as many gradient steps as `trajwise_alternating_training_loop`
         would have run while interleaving over those trajectories, i.e.
             `sum_over_collected_trajs(len(traj["rewards"])) * variant.multi_grad_step`.
      3. Return to (1); no env rollouts happen during the training phase.

    `len(traj["rewards"])` counts query steps (one entry per policy query),
    not raw env steps, so the total gradient-step budget matches the
    interleaved loop exactly for the same stream of trajectories.

    Termination criterion:
      * If `variant.max_online_trajs > 0`, the outer loop stops once at least
        that many trajectories have been collected. The training phase
        following the final collection still runs to completion.
      * Otherwise (default), the loop stops once `i > variant.max_steps`
        gradient updates have been performed.
    """
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    iteration_size = variant.iteration_size
    assert iteration_size >= 1, \
        "iteration_size must be >= 1 for iteration_based_training_loop"

    max_online_trajs = variant.max_online_trajs
    limit_by_trajs = max_online_trajs > 0

    total_env_steps = 0
    total_online_trajs = 0
    i = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)

    pbar_total = max_online_trajs if limit_by_trajs else variant.max_steps

    with tqdm(total=pbar_total, initial=0) as pbar:
        while True:
            if limit_by_trajs:
                if total_online_trajs >= max_online_trajs:
                    break
            else:
                if i > variant.max_steps:
                    break

            # ---- collection phase: iteration_size trajectories, no updates ----
            iter_query_steps = 0
            iter_env_steps = 0
            last_traj = None
            for _ in range(iteration_size):
                traj = collect_traj(variant, agent, env, i, agent_dp)
                add_online_data_to_buffer(variant, traj, online_replay_buffer)
                total_env_steps += traj['env_steps']
                iter_env_steps += traj['env_steps']
                iter_query_steps += len(traj['rewards'])
                total_online_trajs += 1
                last_traj = traj
                if limit_by_trajs:
                    pbar.update()
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', total_online_trajs)
            print('total env steps:', total_env_steps)
            print(f'iteration env steps: {iter_env_steps} '
                  f'(query steps: {iter_query_steps})')

            num_gradsteps = iter_query_steps * variant.multi_grad_step

            if len(online_replay_buffer) <= variant.start_online_updates:
                # Not enough data yet; keep collecting before doing any updates.
                continue

            # ---- training phase: num_gradsteps updates, no env rollouts ----
            for _ in range(num_gradsteps):
                if i == 0:
                    print('performing evaluation for initial checkpoint')
                    if perform_control_evals:
                        perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                    if hasattr(agent, 'perform_eval'):
                        agent.perform_eval(variant, i, wandb_logger, replay_buffer,
                                           replay_buffer_iterator, eval_env)

                batch = next(replay_buffer_iterator)
                update_info = agent.update(batch)

                if not limit_by_trajs:
                    pbar.update()
                i += 1

                if i % variant.log_interval == 0:
                    update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                    for k, v in update_info.items():
                        if v.ndim == 0:
                            wandb_logger.log({f'training/{k}': v}, step=i)
                        elif v.ndim <= 2:
                            wandb_logger.log_histogram(f'training/{k}', v, i)
                    wandb_logger.log({
                        'replay_buffer_size': len(online_replay_buffer),
                        'episode_return (exploration)': last_traj['episode_return'],
                        'is_success (exploration)': int(last_traj['is_success']),
                    }, i)

                if i % variant.eval_interval == 0:
                    wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                    wandb_logger.log({'num_online_trajs': total_online_trajs}, step=i)
                    wandb_logger.log({'env_steps': total_env_steps}, step=i)
                    if perform_control_evals:
                        perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                    if hasattr(agent, 'perform_eval'):
                        agent.perform_eval(variant, i, wandb_logger, replay_buffer,
                                           replay_buffer_iterator, eval_env)

                if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                    agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

                if not limit_by_trajs and i > variant.max_steps:
                    break


def add_online_data_to_buffer(variant, traj, online_replay_buffer):

    discount_horizon = variant.query_freq
    actions = np.array(traj['actions']) # (T, chunk_size, action_dim )
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        
        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()


def _maybe_sync_task_description(env, variant):
    """Copy the active task prompt from a multi-task env onto ``variant``."""
    desc = getattr(env, "task_description", None)
    if desc is not None:
        variant.task_description = desc


def _is_multitask_libero(env, variant):
    return (
        "libero" in variant.env
        and hasattr(env, "task_ids")
        and hasattr(env, "set_next_task_id")
    )


def collect_traj(variant, agent, env, i, agent_dp=None):
    query_frequency = variant.query_freq
    inference_delay = variant.inference_delay
    num_prev_actions = variant.num_prev_actions
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    agent._rng, rng = jax.random.split(agent._rng)
    
    if 'libero' in variant.env:
        obs = env.reset()
        _maybe_sync_task_description(env, variant)
    elif 'aloha' in variant.env:
        obs, _ = env.reset()
    
    image_list = [] # for visualization
    rewards = []
    action_list = []
    obs_list = []
    raw_obs_history = []
    played_action_history = []
    actions = None

    for t in tqdm(range(max_timesteps)):
        raw_obs_history.append(obs)
        curr_image = obs_to_img(obs, variant)

        if t % query_frequency == 0:
            assert agent_dp is not None
            cond_t = _conditioning_timestep(t, inference_delay)
            cond_obs = raw_obs_history[cond_t]
            obs_dict = _raw_obs_to_obs_dict(cond_obs, variant)
            if num_prev_actions > 0:
                obs_dict.update(_prev_action_obs(played_action_history, variant))
            actions, actions_noise, rng = _infer_action_chunk(
                variant, agent, agent_dp, rng, obs_dict, cond_obs, i, False
            )
            action_list.append(actions_noise)
            obs_list.append(obs_dict)

        action_idx = _chunk_action_index(t, inference_delay, query_frequency)
        action_t = actions[action_idx]
        if num_prev_actions > 0:
            played_action_history.append(np.asarray(action_t, dtype=np.float32).reshape(-1))
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated
            
        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

    # add last observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_dict = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
        'state': qpos[np.newaxis, ..., np.newaxis],
    }
    if num_prev_actions > 0:
        obs_dict.update(_prev_action_obs(played_action_history, variant))
    obs_list.append(obs_dict)
    image_list.append(curr_image)
    
    # per episode
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards!=None])
    is_success = (reward == env_max_reward)
    print(f'Rollout Done: {episode_return=}, Success: {is_success}')
    
    
    '''
    We use sparse -1/0 reward to train the SAC agent.
    '''
    if is_success:
        query_steps = len(action_list)
        rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
        masks = np.concatenate([np.ones(query_steps - 1), [0]])
    else:
        query_steps = len(action_list)
        rewards = -np.ones(query_steps)
        masks = np.ones(query_steps)

    return {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1 
    }

def _run_eval_rollout(agent, env, i, variant, agent_dp, rng):
    """Single eval episode in the env's current task."""
    query_frequency = variant.query_freq
    inference_delay = variant.inference_delay
    num_prev_actions = variant.num_prev_actions
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    if 'libero' in variant.env:
        obs = env.reset()
        _maybe_sync_task_description(env, variant)
    elif 'aloha' in variant.env:
        obs, _ = env.reset()

    image_list = []
    rewards = []
    reward = 0
    raw_obs_history = []
    played_action_history = []
    actions = None

    for t in tqdm(range(max_timesteps)):
        raw_obs_history.append(obs)
        curr_image = obs_to_img(obs, variant)

        if t % query_frequency == 0:
            assert agent_dp is not None
            cond_t = _conditioning_timestep(t, inference_delay)
            cond_obs = raw_obs_history[cond_t]
            obs_dict = _raw_obs_to_obs_dict(cond_obs, variant)
            if num_prev_actions > 0:
                obs_dict.update(_prev_action_obs(played_action_history, variant))
            actions, _, rng = _infer_action_chunk(
                variant, agent, agent_dp, rng, obs_dict, cond_obs, i, True
            )

        action_idx = _chunk_action_index(t, inference_delay, query_frequency)
        action_t = actions[action_idx]
        if num_prev_actions > 0:
            played_action_history.append(np.asarray(action_t, dtype=np.float32).reshape(-1))

        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated

        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

    rewards_arr = np.array(rewards)
    episode_return = float(np.sum(rewards_arr))
    episode_highest_reward = (
        float(np.max(rewards_arr)) if rewards_arr.size > 0 else 0.0
    )
    is_success = bool(reward == env_max_reward)
    return episode_return, episode_highest_reward, is_success, t + 1, image_list, rng


def perform_control_eval(agent, env, i, variant, wandb_logger, agent_dp=None):
    print('query frequency', variant.query_freq)
    print('inference delay', variant.inference_delay)
    print('num prev actions', variant.num_prev_actions)
    env_max_reward = variant.env_max_reward
    rng = jax.random.PRNGKey(variant.seed + 456)

    multi_task = _is_multitask_libero(env, variant)
    if multi_task:
        # Run eval_episodes rollouts per task id (pinned via set_next_task_id).
        task_plan = [(int(tid), variant.eval_episodes) for tid in env.task_ids]
        print(
            f"[eval] multi-task: {len(task_plan)} tasks x "
            f"{variant.eval_episodes} episodes each"
        )
    else:
        task_plan = [(None, variant.eval_episodes)]

    all_returns = []
    all_highest = []
    all_success = []
    all_lens = []
    summary_str = "\n"

    for task_id, n_episodes in task_plan:
        per_task_returns = []
        per_task_highest = []
        per_task_success = []
        per_task_lens = []

        if task_id is not None:
            print(f"[eval] task {task_id}: {n_episodes} rollouts")

        for rollout_id in range(n_episodes):
            if task_id is not None:
                env.set_next_task_id(task_id)
            (episode_return, episode_highest_reward, is_success,
             episode_len, image_list, rng) = _run_eval_rollout(
                agent, env, i, variant, agent_dp, rng
            )

            per_task_returns.append(episode_return)
            per_task_highest.append(episode_highest_reward)
            per_task_success.append(is_success)
            per_task_lens.append(episode_len)

            tag = f"task{task_id}_" if task_id is not None else ""
            print(
                f"Rollout {tag}{rollout_id}: {episode_return=}, "
                f"Success: {is_success}"
            )
            video = np.stack(image_list).transpose(0, 3, 1, 2)
            video_key = (
                f"eval_video/task_{task_id}/{rollout_id}"
                if task_id is not None
                else f"eval_video/{rollout_id}"
            )
            wandb_logger.log({video_key: wandb.Video(video, fps=50)}, step=i)

        per_task_returns_arr = np.array(per_task_returns)
        per_task_highest_arr = np.array(per_task_highest)
        per_task_success_arr = np.array(per_task_success)
        per_task_lens_arr = np.array(per_task_lens)

        all_returns.extend(per_task_returns)
        all_highest.extend(per_task_highest)
        all_success.extend(per_task_success)
        all_lens.extend(per_task_lens)

        if task_id is not None:
            sr = float(np.mean(per_task_success_arr))
            ret = float(np.mean(per_task_returns_arr))
            ep_len = float(np.mean(per_task_lens_arr))
            wandb_logger.log(
                {f"evaluation/task_{task_id}/success_rate": sr}, step=i
            )
            wandb_logger.log(
                {f"evaluation/task_{task_id}/avg_return": ret}, step=i
            )
            wandb_logger.log(
                {f"evaluation/task_{task_id}/avg_episode_len": ep_len}, step=i
            )
            for r in range(env_max_reward + 1):
                more = int((per_task_highest_arr >= r).sum())
                wandb_logger.log(
                    {f"evaluation/task_{task_id}/Reward >= {r}": more / n_episodes},
                    step=i,
                )
            summary_str += (
                f"Task {task_id}: success_rate={sr:.3f} "
                f"avg_return={ret:.3f} avg_episode_len={ep_len:.1f} "
                f"(n={n_episodes})\n"
            )

    all_returns_arr = np.array(all_returns)
    all_highest_arr = np.array(all_highest)
    all_success_arr = np.array(all_success)
    all_lens_arr = np.array(all_lens)
    total_episodes = int(all_success_arr.size)

    success_rate = float(np.mean(all_success_arr))
    avg_return = float(np.mean(all_returns_arr))
    avg_episode_len = float(np.mean(all_lens_arr))

    summary_str = (
        f"\nSuccess rate: {success_rate}\n"
        f"Average return: {avg_return}\n\n"
        + summary_str.lstrip()
    )
    wandb_logger.log({"evaluation/avg_return": avg_return}, step=i)
    wandb_logger.log({"evaluation/success_rate": success_rate}, step=i)
    wandb_logger.log({"evaluation/avg_episode_len": avg_episode_len}, step=i)
    if multi_task:
        wandb_logger.log({"evaluation/num_tasks": len(task_plan)}, step=i)
        wandb_logger.log(
            {"evaluation/episodes_per_task": variant.eval_episodes}, step=i
        )
    for r in range(env_max_reward + 1):
        more = int((all_highest_arr >= r).sum())
        more_rate = more / total_episodes
        wandb_logger.log({f"evaluation/Reward >= {r}": more_rate}, step=i)
        summary_str += (
            f"Reward >= {r}: {more}/{total_episodes} = {more_rate * 100}%\n"
        )

    print(summary_str)

def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger):
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(variant, trajs)
    wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)
  
