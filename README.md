# vla-stress-test

Robustness study of [SmolVLA](https://huggingface.co/lerobot/smolvla_libero) on the LIBERO benchmark,
and an attempt to fix one of its weaknesses with residual RL, without retraining the VLA.

Three questions:

1. **Does it listen?** In LIBERO-Goal all ten tasks share one scene. I build the env of task *j*
   but give the instruction of task *i*, and check which of the ten goals the robot actually reaches.
2. **How does it degrade?** Graded perturbations (camera orbit, arm initial pose, lighting, pixel
   noise), each with a continuous intensity, to look at the *shape* of the success curve and
   estimate a breaking point.
3. **Can a small corrector fix it?** A PPO-trained MLP adds a bounded correction to the VLA
   actions (`a = a_VLA + alpha * delta`), with its last layer initialised to zero.

Work in progress, results are added to `results/` as runs finish.

## Setup

Everything runs on a laptop (Apple M1, 8 GB, MPS) or on a Colab/Kaggle T4 with `notebooks/runner.ipynb`.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install "lerobot[smolvla,dataset]==0.6.1"
# Linux
uv pip install -e ".[libero,dev]"
# macOS: hf-libero depends on robomimic -> egl-probe, which does not build on macOS.
# LIBERO itself does not need it, so install it without dependencies:
uv pip install --no-deps hf-libero==0.1.4 robosuite==1.4.0 bddl==1.0.1
uv pip install "hydra-core>=1.2,<1.4" easydict cloudpickle future numba
uv pip install --no-deps -e . && uv pip install pytest==8.3.4
```

`MUJOCO_GL` is set automatically (`egl` on Linux, `cgl` on macOS).

## Running

```bash
python -m vla_stress.evaluate --config configs/baseline_spatial.yaml --smoke   # 1 episode, 60 steps
python -m vla_stress.evaluate --config configs/baseline_spatial.yaml           # full, resumable
bash scripts/run_queue.sh configs/dose_*.yaml                                  # several configs in a row
python scripts/perturbation_grid.py --wrist                                    # what the policy sees
python scripts/make_figures.py                                                 # figures + summary tables
```

Each episode is one CSV row with the config, seed, git commit, LeRobot version, checkpoint
revision and device. Episode *e* of a task always uses LIBERO init state *e* and seed *e*, so
conditions can be compared episode by episode.

## Layout

```
configs/              one YAML per experiment
src/vla_stress/
  env_utils.py        policy loading, LIBERO env, one rollout
  perturbations.py    graded perturbations (intensity in [0, 1] -> physical unit)
  evaluate.py         resumable evaluation loop -> CSV
  residual_rl/        env wrapper, actor-critic, PPO
  analysis/           Wilson intervals, dose-response fit, plots
scripts/              figures, videos, throughput, queue
results/              raw CSVs (one row per episode)
docs/JOURNAL.md       day-to-day notes (in French)
```

## Related work

LIBERO-plus (arXiv 2510.13626) and LIBERO-PRO (arXiv 2510.03827) already perturb LIBERO along
many axes. This project does not redo them: it looks at a few axes with a continuous intensity,
adds a goal-level listening test, and tries a residual correction.
