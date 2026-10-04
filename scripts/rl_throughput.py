"""Go/no-go measurement for residual RL: control steps per second of the full loop
(LIBERO + frozen SmolVLA + residual), for several numbers of in-process envs.

    python scripts/rl_throughput.py --n-envs 1 2 4 --steps 300
"""

import argparse
import time

import numpy as np

from vla_stress.env_utils import load_vla
from vla_stress.residual_rl.env import ResidualLibero

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--task", type=int, default=0)
ap.add_argument("--n-envs", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--n-action-steps", type=int, nargs="+", default=[50])
ap.add_argument("--steps", type=int, default=300, help="control steps per env")
args = ap.parse_args()

vla = load_vla()
print("n_envs,n_action_steps,steps_per_s,hours_per_100k")
for nas in args.n_action_steps:
    for n in args.n_envs:
        env = ResidualLibero(vla, args.suite, args.task, n_envs=n, n_action_steps=nas)
        env.reset()
        env.step(np.zeros((n, 7)))  # warm-up (first VLA call compiles kernels)
        t = time.time()
        for _ in range(args.steps):
            env.step(np.zeros((n, 7)))
        sps = n * args.steps / (time.time() - t)
        print(f"{n},{nas},{sps:.1f},{1e5 / sps / 3600:.2f}", flush=True)
        env.close()
