"""Multi-task wrapper around the LIBERO ``OffScreenRenderEnv``.

Switches the active task on every ``reset()``. ``step`` and attribute access
are forwarded to the env selected by the most recent ``reset()``.
"""
from __future__ import annotations

import pathlib
from typing import Optional, Sequence

import numpy as np

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


class MultiTaskLiberoEnv:
    """Randomly (or cyclically) picks a task from ``task_ids`` on each ``reset()``."""

    def __init__(
        self,
        task_suite,
        task_ids: Sequence[int],
        resolution: int = 256,
        seed: int = 0,
        rng_seed: Optional[int] = None,
        mode: str = "random",
    ) -> None:
        task_ids = [int(t) for t in task_ids]
        if len(task_ids) == 0:
            raise ValueError("MultiTaskLiberoEnv requires at least one task_id")
        n_tasks = task_suite.n_tasks
        for tid in task_ids:
            if not (0 <= tid < n_tasks):
                raise ValueError(
                    f"task_id {tid} out of range for benchmark with n_tasks={n_tasks}"
                )
        if mode not in ("random", "cycle"):
            raise ValueError(f"mode must be 'random' or 'cycle', got {mode!r}")

        self.task_suite = task_suite
        self.task_ids = task_ids
        self.resolution = resolution
        self.seed_val = seed
        self._rng = np.random.default_rng(
            rng_seed if rng_seed is not None else seed
        )
        self.mode = mode
        self._cycle_idx = 0

        self._cached: dict[int, tuple[OffScreenRenderEnv, str, object]] = {}

        self.current_task_id: Optional[int] = None
        self.current_task = None
        self.task_description: Optional[str] = None
        self.current_env: Optional[OffScreenRenderEnv] = None
        self._next_task_id_override: Optional[int] = None

    def _build_env(self, task_id: int):
        task = self.task_suite.get_task(task_id)
        task_description = task.language
        task_bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=self.resolution,
            camera_widths=self.resolution,
        )
        env.seed(self.seed_val)
        return env, task_description, task

    def _get_or_create(self, task_id: int):
        if task_id not in self._cached:
            self._cached[task_id] = self._build_env(task_id)
        return self._cached[task_id]

    def _pick_task_id(self) -> int:
        if self._next_task_id_override is not None:
            tid = self._next_task_id_override
            self._next_task_id_override = None
            return int(tid)
        if self.mode == "random":
            return int(self._rng.choice(self.task_ids))
        tid = self.task_ids[self._cycle_idx % len(self.task_ids)]
        self._cycle_idx += 1
        return int(tid)

    def set_next_task_id(self, task_id: int) -> None:
        """Force the next ``reset()`` to use ``task_id`` (for per-task eval)."""
        if int(task_id) not in self.task_ids:
            raise ValueError(
                f"task_id {task_id} is not in the wrapper's pool {self.task_ids}"
            )
        self._next_task_id_override = int(task_id)

    def reset(self):
        task_id = self._pick_task_id()
        env, task_description, task = self._get_or_create(task_id)
        self.current_task_id = task_id
        self.current_task = task
        self.task_description = task_description
        self.current_env = env
        return env.reset()

    def step(self, action):
        if self.current_env is None:
            raise RuntimeError("call reset() before step()")
        return self.current_env.step(action)

    def seed(self, seed: int) -> None:
        self.seed_val = seed
        for env, _desc, _task in self._cached.values():
            env.seed(seed)

    def close(self) -> None:
        for env, _desc, _task in self._cached.values():
            try:
                env.close()
            except Exception:
                pass
        self._cached.clear()
        self.current_env = None

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        env = self.__dict__.get("current_env")
        if env is None:
            raise AttributeError(
                f"MultiTaskLiberoEnv has no attribute {item!r}; call reset() first."
            )
        return getattr(env, item)
