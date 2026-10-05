"""Graded perturbations for LIBERO.

Every perturbation takes an `intensity` in [0, 1] that is mapped linearly to a
physical quantity (`magnitude`, in the unit given by `unit`). Intensity 0 is
always the unperturbed environment.

Two hooks:
  on_reset(env, episode) -> obs | None   modifies the simulator after a reset
  on_obs(obs) -> obs                     modifies what the policy sees, every frame

A hard reset in LIBERO rebuilds the MuJoCo model, which wipes any change we made
to it. So sim-level perturbations are re-applied after every reset. With soft
resets (used for RL) the model survives, so we keep the original values and
always start from them instead of compounding changes.
"""

from __future__ import annotations

import mujoco
import numpy as np

from vla_stress.env_utils import render_obs, sim

TABLE_HEIGHT = 0.9  # m, approx. top of the LIBERO table; used as the orbit pivot height


class Perturbation:
    name = "none"
    unit = ""
    max_magnitude = 0.0

    def __init__(self, intensity: float = 0.0, max_magnitude: float | None = None):
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {intensity}")
        self.intensity = float(intensity)
        if max_magnitude is not None:
            self.max_magnitude = float(max_magnitude)
        self._cache = {}

    @property
    def magnitude(self) -> float:
        return self.intensity * self.max_magnitude

    def on_reset(self, env, episode: int):
        return None

    def on_obs(self, obs: dict) -> dict:
        return obs

    def _original(self, model, key, getter):
        # One cache entry per MuJoCo model object; a hard reset creates a new model.
        ref = self._cache.get(key)
        if ref is None or ref[0] is not model:
            self._cache[key] = (model, getter())
        return self._cache[key][1]


def _model(env):
    m = sim(env).model
    return m._model if hasattr(m, "_model") else m


def _orbit_pose(pos: np.ndarray, quat: np.ndarray, angle_deg: float, pivot_z: float = TABLE_HEIGHT):
    """Rotate a camera about the vertical axis through the point it looks at on the table.

    MuJoCo cameras look along their local -z axis. We intersect that ray with the
    table plane to get the pivot, then rotate both position and orientation about
    world z. The camera keeps pointing at the same spot, so the scene stays in view.
    """
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    forward = -rot.reshape(3, 3)[:, 2]
    t = (pivot_z - pos[2]) / forward[2]
    pivot = pos + t * forward
    a = np.deg2rad(angle_deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    new_pos = pivot + rz @ (pos - pivot)
    qz = np.zeros(4)
    mujoco.mju_axisAngle2Quat(qz, np.array([0.0, 0.0, 1.0]), a)
    new_quat = np.zeros(4)
    mujoco.mju_mulQuat(new_quat, qz, quat)
    return new_pos, new_quat


class CameraOrbit(Perturbation):
    """Orbit the main (agentview) camera around the scene. The wrist camera is untouched.

    The sign alternates with the episode index (+ on even, - on odd) so that each
    level averages over both directions; the sign is logged so the two sides can
    be separated later.
    """

    name = "camera_orbit"
    unit = "deg"
    max_magnitude = 30.0
    camera = "agentview"

    def sign(self, episode: int) -> int:
        return 1 if episode % 2 == 0 else -1

    def on_reset(self, env, episode):
        if self.intensity == 0:
            return None
        m = _model(env)
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
        pos0, quat0 = self._original(m, "cam", lambda: (m.cam_pos[cid].copy(), m.cam_quat[cid].copy()))
        pos, quat = _orbit_pose(pos0, quat0, self.sign(episode) * self.magnitude)
        m.cam_pos[cid] = pos
        m.cam_quat[cid] = quat
        return render_obs(env)


class LightDimming(Perturbation):
    """Scale every light in the scene (two directional lights + headlight) by (1 - magnitude)."""

    name = "light_dimming"
    unit = "fraction removed"
    max_magnitude = 0.9

    def on_reset(self, env, episode):
        if self.intensity == 0:
            return None
        m = _model(env)
        orig = self._original(
            m,
            "light",
            lambda: (
                m.light_diffuse.copy(),
                m.light_ambient.copy(),
                m.light_specular.copy(),
                m.vis.headlight.diffuse.copy(),
                m.vis.headlight.ambient.copy(),
                m.vis.headlight.specular.copy(),
            ),
        )
        f = 1.0 - self.magnitude
        m.light_diffuse[:] = orig[0] * f
        m.light_ambient[:] = orig[1] * f
        m.light_specular[:] = orig[2] * f
        m.vis.headlight.diffuse[:] = orig[3] * f
        m.vis.headlight.ambient[:] = orig[4] * f
        m.vis.headlight.specular[:] = orig[5] * f
        return render_obs(env)


class RobotInitOffset(Perturbation):
    """Shift each of the 7 arm joints by +/- magnitude radians at the start of the episode.

    Signs are drawn once per episode index (so every condition sees the same
    direction for a given episode). Objects are not moved, so the task itself is
    unchanged; only the arm starts somewhere it never started in the demos.
    """

    name = "robot_init"
    unit = "rad/joint"
    max_magnitude = 0.2
    settle_steps = 5

    def signs(self, episode: int) -> np.ndarray:
        return np.random.default_rng(10_000 + episode).choice([-1.0, 1.0], size=7)

    def on_reset(self, env, episode):
        if self.intensity == 0:
            return None
        s = sim(env)
        robot = env._env.env.robots[0]
        idx = robot._ref_joint_pos_indexes
        s.data.qpos[idx] += self.magnitude * self.signs(episode)
        s.data.qvel[robot._ref_joint_vel_indexes] = 0.0
        s.forward()
        # The OSC controller in delta mode holds the current pose for a zero action.
        for _ in range(self.settle_steps):
            env._env.step([0, 0, 0, 0, 0, 0, -1])
        return render_obs(env)


class ImageNoise(Perturbation):
    """Additive Gaussian pixel noise on both cameras (std in 0-255 units). Fixed seed per episode."""

    name = "image_noise"
    unit = "std (0-255)"
    max_magnitude = 60.0

    def __init__(self, intensity=0.0, max_magnitude=None):
        super().__init__(intensity, max_magnitude)
        self.rng = np.random.default_rng(0)

    def on_reset(self, env, episode):
        self.rng = np.random.default_rng(20_000 + episode)
        return None

    def on_obs(self, obs):
        if self.intensity == 0:
            return obs
        pixels = {}
        for k, img in obs["pixels"].items():
            noisy = img.astype(np.float32) + self.rng.normal(0.0, self.magnitude, img.shape)
            pixels[k] = np.clip(noisy, 0, 255).astype(np.uint8)
        return {**obs, "pixels": pixels}


class CameraMask(Perturbation):
    """Replace one camera stream with a black image (intensity 1), to see which view the policy relies on."""

    name = "camera_mask"
    unit = "masked"
    max_magnitude = 1.0

    def __init__(self, intensity=1.0, max_magnitude=None, camera: str = "image"):
        super().__init__(intensity, max_magnitude)
        self.camera = camera  # "image" = agentview, "image2" = wrist

    def on_obs(self, obs):
        if self.intensity == 0:
            return obs
        pixels = dict(obs["pixels"])
        pixels[self.camera] = np.zeros_like(pixels[self.camera])
        return {**obs, "pixels": pixels}


REGISTRY = {
    cls.name: cls for cls in [Perturbation, CameraOrbit, LightDimming, RobotInitOffset, ImageNoise, CameraMask]
}


def build(name: str, intensity: float = 0.0, **kwargs) -> Perturbation:
    return REGISTRY[name](intensity, **kwargs)
