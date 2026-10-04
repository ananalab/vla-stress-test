"""Thin layer over LeRobot: load SmolVLA, build a LIBERO env, run one episode.

Everything model-related (normalisation, tokenisation, camera renaming) goes
through LeRobot's own processor pipelines, loaded from the checkpoint, so that
the policy sees exactly what it would see under `lerobot-eval`.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field

# MuJoCo picks its GL backend at import time: EGL on headless Linux GPUs, CGL on macOS.
os.environ.setdefault("MUJOCO_GL", "cgl" if platform.system() == "Darwin" else "egl")

import numpy as np
import torch
import torch.utils._pytree as pytree

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv, _get_suite
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

CHECKPOINT = "lerobot/smolvla_libero"
# Hub commit of the checkpoint used for every number in results/.
CHECKPOINT_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
# The checkpoint was trained with cameras named camera1/camera2 while the LeRobot
# LIBERO env exposes image/image2. Without this map the policy refuses to load.
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class VLA:
    policy: torch.nn.Module
    pre: object
    post: object
    device: str
    env_proc: LiberoProcessorStep = field(default_factory=LiberoProcessorStep)

    def batch(self, obs: dict | list[dict], instruction: str | list[str]) -> dict:
        """Raw LIBERO observation(s) (unbatched numpy) -> model-ready batch."""
        obs_list = obs if isinstance(obs, list) else [obs]
        instructions = instruction if isinstance(instruction, list) else [instruction] * len(obs_list)
        stacked = pytree.tree_map(lambda *v: np.stack(v), *obs_list)
        b = preprocess_observation(stacked)
        b["task"] = list(instructions)
        b = self.env_proc._process_observation(b)  # 180 deg image flip + 8-d state
        return self.pre(b)

    @torch.no_grad()
    def chunk(self, obs: dict | list[dict], instruction: str | list[str]) -> np.ndarray:
        """Full action chunk in env units: (chunk_size, 7), or (B, chunk_size, 7) for a list of obs."""
        b = self.batch(obs, instruction)
        actions = self.policy.predict_action_chunk(b)  # (B, T, 7), normalised
        actions = self.post(actions).float().cpu().numpy()
        return actions if isinstance(obs, list) else actions[0]

    def decode_tokens(self, obs: dict, instruction: str) -> str:
        """What the language model actually receives. Used to check language perturbations."""
        b = self.batch(obs, instruction)
        tok = self.pre.steps[[type(s).__name__ for s in self.pre.steps].index("TokenizerProcessorStep")]
        ids = b[OBS_LANGUAGE_TOKENS][0].cpu().tolist()
        return tok.input_tokenizer.decode(ids, skip_special_tokens=True)


def load_vla(device: str | None = None, fp32: bool = False) -> VLA:
    device = pick_device(device)
    cfg = PreTrainedConfig.from_pretrained(CHECKPOINT, revision=CHECKPOINT_REVISION)
    cfg.pretrained_path = CHECKPOINT
    cfg.device = device
    policy = make_policy(cfg, env_cfg=LiberoEnvConfig(task="libero_goal"), rename_map=RENAME_MAP)
    policy.eval()
    # Weights are stored in bf16 and we keep them that way by default (same as on CUDA).
    # fp32 is ~20% faster per call on Apple MPS but doubles memory, which made an 8 GB
    # laptop swap so much that episodes got 2-3x slower overall.
    if fp32:
        policy.float()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return VLA(policy, pre, post, device)


def make_env(suite_name: str, task_id: int, resolution: int = 360, hard_reset: bool = True) -> LiberoEnv:
    # 360x360 is the LeRobot default for LIBERO and what the checkpoint's train config used.
    suite = _get_suite(suite_name)
    return LiberoEnv(
        suite,
        task_id,
        suite_name,
        obs_type="pixels_agent_pos",
        observation_width=resolution,
        observation_height=resolution,
        hard_reset=hard_reset,
    )


def task_instructions(suite_name: str) -> list[str]:
    return [t.language for t in _get_suite(suite_name).tasks]


def max_steps(suite_name: str) -> int:
    return TASK_SUITE_MAX_STEPS[suite_name]


def render_obs(env: LiberoEnv) -> dict:
    """Re-render the current sim state. Needed after we touch the model (camera, lights, qpos).

    Two traps here: camera world poses are only recomputed by mj_forward, and robosuite
    caches observables between sim steps unless asked to refresh them.
    """
    sim(env).forward()
    return env._format_raw_obs(env._env.env._get_observations(force_update=True))


def sim(env: LiberoEnv):
    return env._env.env.sim


def reset(env: LiberoEnv, episode: int, seed: int):
    """Reset to the LIBERO init state `episode` (fixed list of 50 per task)."""
    env.init_state_id = episode
    obs, _ = env.reset(seed=seed)
    return obs


def run_episode(
    env: LiberoEnv,
    vla: VLA,
    instruction: str,
    episode: int,
    seed: int,
    perturbation=None,
    n_action_steps: int = 50,
    record: bool = False,
    max_episode_steps: int | None = None,
    residual=None,
) -> dict:
    """One rollout. The perturbation (if any) is applied after the hard reset and on every frame.

    The same (episode, seed) pair gives the same initial state and the same flow-matching noise
    for every condition, so conditions can be compared episode by episode.
    """
    t0 = time.time()
    t_policy = 0.0
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs = reset(env, episode, seed)
    if perturbation is not None:
        obs = perturbation.on_reset(env, episode) or obs
    t_reset = time.time() - t0
    horizon = max_episode_steps or env._max_episode_steps
    frames, success, step, n_calls = [], False, 0, 0
    queue: list[np.ndarray] = []
    while step < horizon:
        policy_obs = perturbation.on_obs(obs) if perturbation is not None else obs
        if record:
            frames.append(np.concatenate([im[::-1, ::-1] for im in policy_obs["pixels"].values()], axis=1))
        if not queue:
            tp = time.time()
            queue = list(vla.chunk(policy_obs, instruction)[:n_action_steps])
            t_policy += time.time() - tp
            n_calls += 1
        action = queue.pop(0)
        if residual is not None:
            # Residual RL: final action = VLA action + alpha * correction(state, VLA action, time)
            action = residual(obs, action, len(queue), step, horizon)
        obs, _, terminated, _, info = env.step(np.clip(action, -1.0, 1.0))
        step += 1
        if info["is_success"]:
            success = True
            break
        if terminated:
            break
    return {
        "success": int(success),
        "steps": step,
        "n_policy_calls": n_calls,
        "duration_s": round(time.time() - t0, 2),
        "reset_s": round(t_reset, 2),
        "policy_s": round(t_policy, 2),
        "frames": frames,
    }
