"""Utilities for residual RL: pi0 base chunks + SAC action edits."""
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze

_RESIDUAL_CRITIC_DROP_KEYS = frozenset({'base_action', 'chunk_step'})


def chunk_action_index(t, inference_delay, query_freq):
    """Index into the current action chunk for env step ``t``."""
    if t < query_freq:
        return t
    if t < inference_delay:
        return t % query_freq
    return inference_delay + (t % query_freq)


def chunk_local_step(t, query_freq):
    """Position within the current query interval (0 .. query_freq-1)."""
    return int(t % query_freq)


def executed_action_indices(query_step: int, variant) -> np.ndarray:
    """Chunk indices played during one query interval starting at env step ``query_step``."""
    d = variant.inference_delay
    qf = variant.query_freq
    return np.array(
        [chunk_action_index(query_step + k, d, qf) for k in range(qf)],
        dtype=np.int32,
    )


def extract_executed_chunk(full_chunk, variant, query_step: int) -> np.ndarray:
    """Return the executed slice from a full pi action chunk."""
    idx = executed_action_indices(query_step, variant)
    return np.asarray(full_chunk, dtype=np.float32)[idx]


def obs_with_residual_context(
        obs_dict: Dict,
        base_action,
        chunk_level: bool = False,
) -> Dict:
    """Attach pi0 base action context for residual SAC.

    Step mode (chunk_level=False): single current base action, shape (action_dim,).
    Chunk mode (chunk_level=True): executed base chunk, shape (query_freq, action_dim).
    """
    obs = {k: v for k, v in obs_dict.items() if k not in ('base_action', 'chunk_step')}
    base = np.asarray(base_action, dtype=np.float32)
    if chunk_level:
        if base.ndim != 2:
            raise ValueError(f"chunk-level base_action must be shape (query_freq, action_dim), got {base.shape}")
        obs['base_action'] = base.reshape(1, *base.shape, 1)
    else:
        base = base.reshape(-1)
        obs['base_action'] = base.reshape(1, base.shape[0], 1)
    return obs


def residual_edit_shape(variant) -> tuple:
    """SAC edit action shape (query_freq, played_action_dim) for chunk-level residual."""
    return (int(variant.query_freq), int(variant.played_action_dim))


def is_chunk_level_residual(variant) -> bool:
    return variant.policy_mode == 'residual' and variant.residual_edit_mode == 'chunk'


def obs_without_base_action(obs) -> Dict:
    """Drop residual-only keys so the critic sees the original SAC state."""
    if isinstance(obs, FrozenDict):
        return freeze({k: v for k, v in obs.items() if k not in _RESIDUAL_CRITIC_DROP_KEYS})
    return {k: v for k, v in obs.items() if k not in _RESIDUAL_CRITIC_DROP_KEYS}


def residual_resulting_action(base_action, edit_action):
    """Return executed action = base + edit for step- or chunk-level residual."""
    base = jnp.asarray(base_action)
    edit = jnp.asarray(edit_action)
    if base.ndim == 3:
        # Step mode: (B, action_dim, 1) + (B, 1, action_dim).
        base = base[..., 0]
        if edit.ndim == 3:
            return base[:, None, :] + edit
        return base + edit
    if base.ndim == 4:
        # Chunk mode: (B, query_freq, action_dim, 1) + (B, query_freq, action_dim).
        return base[..., 0] + edit
    raise ValueError(f"Unexpected base_action shape: {base.shape}")


def prepare_critic_batch(obs, edit_action, policy_mode: str) -> Tuple:
    """Map actor batch to critic inputs for residual policy."""
    if policy_mode != 'residual':
        return obs, edit_action
    return obs_without_base_action(obs), residual_resulting_action(obs['base_action'], edit_action)