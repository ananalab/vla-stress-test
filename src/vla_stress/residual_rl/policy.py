"""Small actor-critic for the residual correction.

Input (17 numbers, no images):
  end-effector position (3) + quaternion (4) + gripper fingers (2)   proprioception
  action proposed by the frozen VLA for this step (7)
  progress in the episode t / T (1)

The actor outputs the mean of a Gaussian over the correction in [-1, 1]^7. Its last
layer starts at exactly zero, so before any training the corrected policy is the
VLA itself (the mean correction is 0). That gives a safe starting point: RL can
only move away from the VLA if it finds something better.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

OBS_DIM = 3 + 4 + 2 + 7 + 1
ACT_DIM = 7


def features(obs: dict, vla_action: np.ndarray, t: int, horizon: int) -> np.ndarray:
    rs = obs["robot_state"]
    return np.concatenate(
        [rs["eef"]["pos"], rs["eef"]["quat"], rs["gripper"]["qpos"], vla_action, [t / horizon]]
    ).astype(np.float32)


class RunningNorm(nn.Module):
    """Running mean/std of the inputs (Welford), stored as buffers so it is saved with the model."""

    def __init__(self, dim: int, eps: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))

    def update(self, x: torch.Tensor):
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot) / tot
        self.count = tot

    def forward(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -5, 5)


def layer(i, o, std=np.sqrt(2)):
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, std)
    nn.init.zeros_(lin.bias)
    return lin


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 128, init_log_std: float = -1.0):
        super().__init__()
        self.norm = RunningNorm(OBS_DIM)
        self.critic = nn.Sequential(layer(OBS_DIM, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(), layer(hidden, 1, 1.0))
        self.actor = nn.Sequential(layer(OBS_DIM, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(), layer(hidden, ACT_DIM, 0.0))
        # std = exp(-1) ~ 0.37 on the correction, i.e. ~0.07 on the final action with alpha = 0.2
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), init_log_std))

    def value(self, x):
        return self.critic(self.norm(x)).squeeze(-1)

    def dist(self, x):
        mean = self.actor(self.norm(x))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    @torch.no_grad()
    def act(self, x: np.ndarray, deterministic: bool = True) -> np.ndarray:
        x = torch.as_tensor(x, dtype=torch.float32)
        d = self.dist(x)
        a = d.mean if deterministic else d.sample()
        return a.clamp(-1, 1).numpy()


class Residual:
    """Callable plugged into env_utils.run_episode for evaluation (deterministic mean action)."""

    def __init__(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.alpha = ckpt["args"]["alpha"]
        self.model = ActorCritic(hidden=ckpt["args"]["hidden"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def __call__(self, obs, vla_action, remaining_in_chunk, t, horizon):
        delta = self.model.act(features(obs, vla_action, t, horizon))
        return vla_action + self.alpha * delta
