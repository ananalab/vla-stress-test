# Drafts for LeRobot (not posted yet)

## Issue 1: `lerobot-eval` ignores the rename map saved with the checkpoint

**Context.** `lerobot/smolvla_libero` names its cameras `observation.images.camera1/2/3`, while the
LIBERO env produces `observation.images.image/image2`. The checkpoint's `policy_preprocessor.json`
already contains the right `rename_observations_processor` map.

**What happens.** Running the command from the LIBERO docs without `--rename_map` fails in
`make_policy` with a feature-mismatch error. Passing `--rename_map` works, but in `eval_main` the
preprocessor override `{"rename_observations_processor": {"rename_map": cfg.rename_map}}` always
replaces the saved map, so the saved one is never used, even though it is the correct one.

**Proposal.** When `cfg.rename_map` is empty, read the map from the checkpoint's preprocessor config
and use it both for `make_policy` and for the processor (or at least do not override the saved map
with an empty dict). Happy to open a PR with a test.

**Versions.** lerobot 0.6.1, checkpoint revision `31d453f7edd7`.

## Issue 2: LIBERO runs on macOS, the `sys_platform == "linux"` marker could be relaxed or documented

`hf-libero` is restricted to Linux in the `libero` extra. The only package that does not build on
macOS is `egl-probe`, pulled by `robomimic`, which LIBERO evaluation does not import. Installing
`hf-libero`, `robosuite==1.4.0` and `bddl==1.0.1` with `--no-deps` plus their real dependencies
gives a working setup on Apple Silicon with `MUJOCO_GL=cgl` (offscreen rendering), and
`lerobot-eval` runs on MPS. I ran about 2,400 evaluation episodes this way.

Two possible changes: a short "macOS" section in the LIBERO doc page, or making `robomimic`
optional in `hf-libero`.

## Note for people perturbing LIBERO scenes

After modifying `sim.model` (camera pose, lights) you need `sim.forward()` **and**
`env._get_observations(force_update=True)`: robosuite caches observables between sim steps, so
without the second call the policy keeps seeing the unmodified image, silently.
