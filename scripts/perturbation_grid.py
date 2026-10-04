"""Render what the policy sees at each perturbation level (figure + feasibility check).

    python scripts/perturbation_grid.py --suite libero_spatial --task 0 --out media/perturbation_grid.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from vla_stress import perturbations as P
from vla_stress.env_utils import make_env, reset

FAMILIES = ["camera_orbit", "light_dimming", "robot_init", "image_noise"]
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--wrist", action="store_true", help="also show the wrist camera")
    ap.add_argument("--out", default="media/perturbation_grid.png")
    args = ap.parse_args()

    env = make_env(args.suite, args.task)
    rows = len(args.families) * (2 if args.wrist else 1)
    fig, axes = plt.subplots(rows, len(LEVELS), figsize=(2.0 * len(LEVELS), 2.1 * rows))
    axes = np.atleast_2d(axes)
    r = 0
    for fam in args.families:
        for j, level in enumerate(LEVELS):
            pert = P.build(fam, level)
            obs = reset(env, args.episode, seed=args.episode)
            obs = pert.on_reset(env, args.episode) or obs
            obs = pert.on_obs(obs)
            views = [obs["pixels"]["image"]] + ([obs["pixels"]["image2"]] if args.wrist else [])
            for k, img in enumerate(views):
                ax = axes[r + k, j]
                ax.imshow(img[::-1, ::-1])
                ax.set_xticks([])
                ax.set_yticks([])
                if r + k == 0 or k == 0:
                    ax.set_title(f"{pert.magnitude:g} {pert.unit}", fontsize=7)
                if j == 0:
                    ax.set_ylabel(fam + (" (wrist)" if k else ""), fontsize=8)
        r += len(views)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)
    env.close()


if __name__ == "__main__":
    main()
