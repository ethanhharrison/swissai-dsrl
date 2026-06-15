"""Utilities for residual RL: pi0 base chunks + SAC action edits."""
from typing import Dict

import numpy as np


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


def chunk_step_onehot(local_step: int, query_freq: int) -> np.ndarray:
    """One-hot vector of length ``query_freq`` for the current chunk timestep."""
    onehot = np.zeros((query_freq, 1), dtype=np.float32)
    onehot[local_step, 0] = 1.0
    return onehot


def obs_with_residual_context(
        obs_dict: Dict,
        base_executed_chunk: np.ndarray,
        query_freq: int,
        local_step: int = 0,
        chunk_level: bool = False,
) -> Dict:
    """Attach base executed chunk; optionally chunk-step one-hot (per-step edit mode)."""
    obs = {k: v for k, v in obs_dict.items() if k not in ('base_action', 'chunk_step')}
    base = np.asarray(base_executed_chunk, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError(f"base_executed_chunk must be shape (query_freq, action_dim), got {base.shape}")
    obs['base_action'] = base.reshape(1, *base.shape, 1)
    if not chunk_level:
        obs['chunk_step'] = chunk_step_onehot(local_step, query_freq).reshape(1, query_freq, 1)
    return obs


def residual_edit_shape(variant) -> tuple:
    """SAC edit action shape (query_freq, played_action_dim) for chunk-level residual."""
    return (int(variant.query_freq), int(variant.played_action_dim))


def is_chunk_level_residual(variant) -> bool:
    return variant.policy_mode == 'residual' and variant.residual_edit_mode == 'chunk'

