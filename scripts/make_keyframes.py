"""Figures made of simulation frames, from the rollouts cached by scripts/record_rollouts.py.

    python scripts/make_keyframes.py

Writes report/figures/teaser.pdf and keyframes_failures.pdf. The frames
are exactly what the policy saw from the main camera during the recorded episodes.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vla_stress.analysis import plots  # noqa: F401  (shared style)
from vla_stress.analysis.plots import GREY, INK, save

CACHE = Path("outputs/video_cache")
FIG = Path("report/figures")
GREEN, RED = "#1b8c5a", "#c83732"


def runs(name):
    return pickle.load(open(CACHE / f"{name}.pkl", "rb"))


def crop(img, frac=0.0):
    """Optional symmetric crop of the top band (the wall behind the table carries no information)."""
    h = img.shape[0]
    return img[int(frac * h):]


def badge(ax, ok, x=0.04, y=0.94, size=6):
    txt, col = ("success", GREEN) if ok else ("failure", RED)
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=size, color="white", fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.35", fc=col, ec="none"))


def tstamp(ax, t, size=5.5):
    ax.text(0.96, 0.94, f"t = {t}", transform=ax.transAxes, fontsize=size, color="white", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.35), ec="none"))


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def strip_times(n, k):
    """k frame indices spread over an episode of n frames, always including first and last."""
    return [int(round(i)) for i in np.linspace(0, n - 1, k)]


# ---------------------------------------------------------------- teaser
def teaser():
    lang = runs("language")
    pert = runs("perturbations")
    res = runs("residual")
    fig = plt.figure(figsize=(6.75, 1.36))
    outer = fig.add_gridspec(1, 3, width_ratios=[3, 4, 2], wspace=0.06, left=0.005, right=0.995, top=0.8, bottom=0.11)
    titles = ["Does it listen?", "How does it break?", "Can a small corrector fix it?"]
    subs = ["one scene, three instructions", "same episode, four conditions", "arm offset, seen init state"]

    # (a) language: final frames of three instructions in the same scene
    g = outer[0].subgridspec(1, 3, wspace=0.03)
    pick = [1, 2, 4]
    for j, i in enumerate(pick):
        frames, _, sub, ok = lang[i]
        ax = fig.add_subplot(g[0, j])
        ax.imshow(crop(frames[-1], 0.08))
        clean(ax)
        badge(ax, ok, size=5)
        ax.set_xlabel("“" + sub.replace("put the ", "").replace(" of the cabinet", "").replace("open the ", "open ") + "”", fontsize=5.6, labelpad=2, wrap=True)
    # (b) perturbations
    g = outer[1].subgridspec(1, 4, wspace=0.03)
    names = ["none", "camera 22.5°", "arm 0.1 rad", "90% light off"]
    for j, (frames, title, sub, ok) in enumerate(pert):
        ax = fig.add_subplot(g[0, j])
        k = min(len(frames) - 1, 140) if not ok else len(frames) - 1
        ax.imshow(crop(frames[k], 0.08))
        clean(ax)
        badge(ax, ok, size=5)
        ax.set_xlabel(names[j], fontsize=5.8, labelpad=2)
    # (c) residual: VLA alone vs corrected on the same episode
    g = outer[2].subgridspec(1, 2, wspace=0.03)
    for j, (frames, title, sub, ok) in enumerate(res[:2]):
        ax = fig.add_subplot(g[0, j])
        k = min(len(frames) - 1, 140) if not ok else len(frames) - 1
        ax.imshow(crop(frames[k], 0.08))
        clean(ax)
        badge(ax, ok, size=5)
        ax.set_xlabel("VLA" if j == 0 else "VLA + residual", fontsize=5.8, labelpad=2)

    xs = [0.005, 0.005 + 3 / 9.12 * 0.99 + 0.006, 0.005 + 7 / 9.12 * 0.99 + 0.012]
    for x, t, s, lab in zip(xs, titles, subs, "abc"):
        fig.text(x + 0.004, 0.985, f"{lab}  {t}", fontsize=8.5, fontweight="bold", color=INK, va="top")
        fig.text(x + 0.022, 0.885, s, fontsize=6.5, color=GREY, va="top")
    save(fig, FIG / "teaser.pdf")


# ---------------------------------------------------------------- failure keyframes
def keyframes_failures(k=5, top=0.2):
    pert = runs("perturbations")
    labels = ["No perturbation", "Camera orbit 22.5\u00b0", "Arm offset\n0.1 rad/joint", "90% of light\nremoved"]
    w = 5.45 / k
    fig, axes = plt.subplots(len(pert), k, figsize=(6.75, len(pert) * (w * (1 - top) + 0.08) + 0.05),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.06, "left": 0.19, "right": 1, "top": 1, "bottom": 0})
    for r, (frames, title, sub, ok) in enumerate(pert):
        for c, t in enumerate(strip_times(len(frames), k)):
            ax = axes[r, c]
            ax.imshow(crop(frames[t], top))
            clean(ax)
            tstamp(ax, t)
            if c == k - 1:
                badge(ax, ok)
        axes[r, 0].set_ylabel(labels[r], fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
    save(fig, FIG / "keyframes_failures.pdf")


if __name__ == "__main__":
    teaser()
    keyframes_failures()
    print("wrote teaser.pdf, keyframes_failures.pdf")
