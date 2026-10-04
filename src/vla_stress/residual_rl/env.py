"""LIBERO + frozen SmolVLA, seen from the residual agent.

Several LIBERO envs live in the same process. The VLA is queried in one batch for
all envs whose action chunk is empty, which amortises its cost (it dominates
when n_action_steps is small). The agent picks a correction for every env at
every control step.

Training uses soft resets (no MuJoCo model rebuild) because a hard reset costs
several seconds; evaluation of the trained corrector goes back through
vla_stress.evaluate with hard resets, like every other number in the report.
"""

from __future__ import annotations

import numpy as np

from vla_stress import perturbations as P
from vla_stress.env_utils import make_env, reset
from vla_stress.residual_rl.policy import features


class ResidualLibero:
    def __init__(
        self,
        vla,
        suite: str,
        task_id: int,
        perturbation: str = "none",
        intensity: float = 0.0,
        n_envs: int = 2,
        alpha: float = 0.2,
        n_action_steps: int = 50,
        init_states: list[int] = tuple(range(40)),
        seed: int = 0,
        hard_reset: bool = False,
        resolution: int = 360,
        max_episode_steps: int | None = None,
    ):
        self.vla = vla
        self.alpha = alpha
        self.n = n_envs
        self.n_action_steps = n_action_steps
        self.init_states = list(init_states)
        self.rng = np.random.default_rng(seed)
        self.envs = [make_env(suite, task_id, resolution, hard_reset=hard_reset) for _ in range(n_envs)]
        self.perts = [P.build(perturbation, intensity) for _ in range(n_envs)]
        self.instruction = self.envs[0].task_description
        self.horizon = max_episode_steps or self.envs[0]._max_episode_steps
        self.obs = [None] * n_envs
        self.queues = [[] for _ in range(n_envs)]
        self.t = np.zeros(n_envs, dtype=int)
        self.episode_ids = np.zeros(n_envs, dtype=int)

    # -- helpers -------------------------------------------------------------
    def _reset_one(self, i):
        ep = int(self.rng.choice(self.init_states))
        seed = int(self.rng.integers(1 << 30))
        obs = reset(self.envs[i], ep, seed)
        self.obs[i] = self.perts[i].on_reset(self.envs[i], ep) or obs
        self.queues[i] = []
        self.t[i] = 0
        self.episode_ids[i] = ep

    def _refill(self):
        """Query the VLA (one batch) for every env whose chunk is used up."""
        need = [i for i in range(self.n) if not self.queues[i]]
        if not need:
            return
        views = [self.perts[i].on_obs(self.obs[i]) for i in need]
        chunks = self.vla.chunk(views, [self.instruction] * len(need))
        for i, c in zip(need, chunks):
            self.queues[i] = list(c[: self.n_action_steps])

    def _features(self):
        return np.stack([features(self.obs[i], self.queues[i][0], self.t[i], self.horizon) for i in range(self.n)])

    # -- gym-like API --------------------------------------------------------
    def reset(self):
        for i in range(self.n):
            self._reset_one(i)
        self._refill()
        return self._features()

    def step(self, delta: np.ndarray):
        """delta: (n, 7) corrections in [-1, 1]. Returns obs, reward, done, infos."""
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=bool)
        infos = []
        for i in range(self.n):
            a_vla = self.queues[i].pop(0)
            action = np.clip(a_vla + self.alpha * np.clip(delta[i], -1, 1), -1, 1)
            self.obs[i], _, terminated, _, info = self.envs[i].step(action)
            self.t[i] += 1
            success = bool(info["is_success"])
            if success or terminated or self.t[i] >= self.horizon:
                rewards[i] = float(success)
                dones[i] = True
                infos.append({"episode_end": True, "success": success, "length": int(self.t[i]), "init_state": int(self.episode_ids[i])})
                self._reset_one(i)
            else:
                infos.append({})
        self._refill()
        return self._features(), rewards, dones, infos

    def close(self):
        for e in self.envs:
            e.close()
