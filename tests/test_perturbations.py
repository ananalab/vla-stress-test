"""Checks that each perturbation really changes what the policy sees, and grows with intensity.

These need LIBERO (Linux, or macOS with the manual install described in the README).
The first version of the camera/light perturbations silently did nothing because
robosuite caches observations between sim steps; these tests catch that.
"""

import numpy as np
import pytest

pytest.importorskip("libero")

from vla_stress import perturbations as P  # noqa: E402
from vla_stress.env_utils import make_env, reset  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = make_env("libero_spatial", 0, resolution=128)
    yield e
    e.close()


def observe(env, name, intensity, episode=0, **kw):
    pert = P.build(name, intensity, **kw)
    obs = reset(env, episode, seed=episode)
    obs = pert.on_reset(env, episode) or obs
    return pert.on_obs(obs)


def img_diff(a, b, cam="image"):
    return np.abs(a["pixels"][cam].astype(float) - b["pixels"][cam].astype(float)).mean()


@pytest.mark.parametrize("name", ["camera_orbit", "light_dimming", "robot_init", "image_noise"])
def test_effect_grows_with_intensity(env, name):
    ref = observe(env, name, 0.0)
    diffs = [img_diff(ref, observe(env, name, i)) for i in (0.25, 1.0)]
    assert diffs[0] > 0.5, f"{name} at 0.25 does not change the image"
    assert diffs[1] > diffs[0]


def test_intensity_zero_is_identity(env):
    a = observe(env, "none", 0.0)
    for name in ["camera_orbit", "light_dimming", "robot_init", "image_noise"]:
        assert img_diff(a, observe(env, name, 0.0)) == 0.0


def test_camera_orbit_leaves_wrist_camera(env):
    ref, moved = observe(env, "none", 0.0), observe(env, "camera_orbit", 1.0)
    assert img_diff(ref, moved, "image") > 1.0
    assert img_diff(ref, moved, "image2") == 0.0


def test_soft_reset_does_not_accumulate():
    # Without a hard reset the MuJoCo model survives, so applying the orbit twice
    # must still give the same camera pose as applying it once.
    e = make_env("libero_spatial", 0, resolution=128, hard_reset=False)
    pert = P.build("camera_orbit", 1.0)
    first = pert.on_reset(e, 0) if reset(e, 0, 0) is not None else None
    second = pert.on_reset(e, 0) if reset(e, 0, 0) is not None else None
    assert img_diff(first, second) < 1.0
    e.close()


def test_camera_mask_blacks_out_one_view(env):
    obs = observe(env, "camera_mask", 1.0, camera="image2")
    assert obs["pixels"]["image2"].max() == 0
    assert obs["pixels"]["image"].max() > 0
