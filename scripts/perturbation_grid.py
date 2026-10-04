"""Render what the policy sees at each perturbation level (report figure + feasibility check).

    python scripts/perturbation_grid.py --suite libero_spatial --task 0 --out report/figures/perturbation_grid.pdf
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from vla_stress import perturbations as P
from vla_stress.analysis import plots  # noqa: F401  (shared matplotlib style)
from vla_stress.analysis.plots import TITLES
from vla_stress.env_utils import make_env, reset

FAMILIES = ["camera_orbit", "robot_init", "light_dimming", "image_noise"]
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
FMT = {
    "camera_orbit": lambda m: f"{m:g}°",
    "robot_init": lambda m: f"{m:.2f} rad",
    "light_dimming": lambda m: f"{100 * m:.0f}% off",
    "image_noise": lambda m: f"σ = {m:g}",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--wrist", action="store_true", help="add a row with the wrist camera for each family")
    ap.add_argument("--out", default="report/figures/perturbation_grid.pdf")
    args = ap.parse_args()

    env = make_env(args.suite, args.task)
    per = 2 if args.wrist else 1
    rows = len(args.families) * per
    fig, axes = plt.subplots(rows, len(LEVELS), figsize=(3.25, 0.68 * rows + 0.1), gridspec_kw={"wspace": 0.04, "hspace": 0.32})
    axes = np.atleast_2d(axes)
    for f, fam in enumerate(args.families):
        for j, level in enumerate(LEVELS):
            pert = P.build(fam, level)
            obs = reset(env, args.episode, seed=args.episode)
            obs = pert.on_reset(env, args.episode) or obs
            obs = pert.on_obs(obs)
            views = [obs["pixels"]["image"]] + ([obs["pixels"]["image2"]] if args.wrist else [])
            for k, img in enumerate(views):
                ax = axes[f * per + k, j]
                ax.imshow(img[::-1, ::-1])
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if k == 0:
                    ax.set_title(FMT[fam](pert.magnitude), fontsize=6, pad=1.5)
                if j == 0:
                    ax.set_ylabel(TITLES[fam] + (" (wrist)" if k else ""), fontsize=6.5)
    fig.savefig(args.out)
    if args.out.endswith(".pdf"):
        fig.savefig(args.out[:-4] + ".png", dpi=300)
    print("saved", args.out)
    env.close()


if __name__ == "__main__":
    main()
