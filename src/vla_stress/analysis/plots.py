"""Figures for the report. Every function takes dataframes read from results/ and an output path."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vla_stress.analysis.stats import bootstrap_x50, fit_dose_response, logistic, rate_table  # noqa: E402

COLORS = {"camera_orbit": "#2a78d6", "robot_init": "#eb6834", "light_dimming": "#1baf7a", "image_noise": "#eda100"}
GREY = "#52514e"
LABELS = {
    "camera_orbit": "Camera orbit (deg)",
    "robot_init": "Arm joint offset (rad/joint)",
    "light_dimming": "Light removed (fraction)",
    "image_noise": "Pixel noise (std, 0-255)",
}

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8984",
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def save(fig, path):
    fig.savefig(path)
    if str(path).endswith(".pdf"):
        fig.savefig(str(path)[:-4] + ".png", dpi=200)
    plt.close(fig)


def language_matrix(df: pd.DataFrame, instructions: list[str], path):
    """M[i, j] = success on goal j when the instruction of task i is given."""
    df = df.copy()
    df["said"] = df["variant"].str.replace("from_task_", "").astype(int)
    n = len(instructions)
    k = df.pivot_table(index="said", columns="task_id", values="success", aggfunc="sum").reindex(index=range(n), columns=range(n))
    c = df.pivot_table(index="said", columns="task_id", values="success", aggfunc="count").reindex(index=range(n), columns=range(n))
    m = k / c
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if not np.isnan(m.values[i, j]):
                v = m.values[i, j]
                ax.text(j, i, f"{int(k.values[i, j])}/{int(c.values[i, j])}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.6 else "#0b0b0b")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("goal checked by the environment (task j)")
    ax.set_ylabel("instruction given (task i)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="success rate")
    save(fig, path)
    return m, c


def dose_response(curves: dict[str, pd.DataFrame], path, n_boot: int = 500):
    """One panel per family. Points: pooled success with Wilson 95% CI. Line: logistic fit.
    Shaded band: 95% task-bootstrap interval of the breaking point x50."""
    fams = list(curves)
    fig, axes = plt.subplots(1, len(fams), figsize=(1.85 * len(fams), 1.9), sharey=True)
    axes = np.atleast_1d(axes)
    summary = []
    for ax, fam in zip(axes, fams):
        d = curves[fam]
        col = COLORS.get(fam, GREY)
        t = rate_table(d, ["magnitude"])
        ax.errorbar(t["magnitude"], t["rate"], yerr=[t["rate"] - t["lo"], t["hi"] - t["rate"]], fmt="o", ms=3.5, color=col, ecolor=col, elinewidth=1, capsize=0)
        fit = fit_dose_response(d["magnitude"].values, d["success"].values)
        xs = np.linspace(0, d["magnitude"].max(), 200)
        ax.plot(xs, logistic(xs, fit["s0"], fit["x50"], fit["w"]), color=col, lw=1.5)
        boots = bootstrap_x50(d, n_boot=n_boot)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        x50 = fit["x50"]
        xmax = d["magnitude"].max()
        if x50 <= xmax:
            ax.axvspan(max(lo, 0), min(hi, xmax), color=col, alpha=0.12, lw=0)
            ax.axvline(x50, color=col, lw=0.8, ls="--")
        ax.set_xlabel(LABELS.get(fam, fam))
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.04 * xmax, 1.04 * xmax)
        n_per = int(t["n"].median())
        ax.set_title(f"n={n_per}/point", color=GREY)
        summary.append({"family": fam, "s0": fit["s0"], "x50": x50, "x50_lo": lo, "x50_hi": hi, "w": fit["w"]})
    axes[0].set_ylabel("success rate")
    fig.tight_layout()
    save(fig, path)
    return pd.DataFrame(summary)


def bars(table: pd.DataFrame, label_col: str, path, title: str | None = None, color="#2a78d6", ref: float | None = None, figsize=(3.3, 1.8)):
    """Horizontal bars of success rate with Wilson CIs."""
    t = table.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["rate"], color=color, height=0.6)
    ax.errorbar(t["rate"], y, xerr=[t["rate"] - t["lo"], t["hi"] - t["rate"]], fmt="none", ecolor="#0b0b0b", elinewidth=0.8)
    for yi, (r, k, n) in zip(y, zip(t["rate"], t["k"], t["n"])):
        ax.text(1.02, yi, f"{k}/{n}", va="center", fontsize=6.5, color=GREY, transform=ax.get_yaxis_transform())
    if ref is not None:
        ax.axvline(ref, color=GREY, lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(t[label_col])
    ax.set_xlim(0, 1)
    ax.set_xlabel("success rate")
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title)
    save(fig, path)


def learning_curves(logs: dict[str, pd.DataFrame], path, baseline: float | None = None, window: int = 5):
    """Rollout success rate during PPO, smoothed; mean and min-max across seeds."""
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    steps = None
    ys = []
    for name, lg in logs.items():
        s = lg["success_rate"].rolling(window, min_periods=1).mean()
        ax.plot(lg["step"], s, color="#2a78d6", lw=0.7, alpha=0.35)
        ys.append(s.values)
        steps = lg["step"].values if steps is None or len(lg) < len(steps) else steps
    if ys:
        L = min(len(y) for y in ys)
        Y = np.stack([y[:L] for y in ys])
        ax.plot(steps[:L], Y.mean(0), color="#2a78d6", lw=1.8, label=f"mean of {len(ys)} seeds")
    if baseline is not None:
        ax.axhline(baseline, color=GREY, ls=":", lw=1, label="VLA alone (same condition)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"rollout success ({window}-update avg)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    save(fig, path)
