"""Figures for the report. Every function takes dataframes read from results/ and an output path.

Sizes are in inches for a two-column workshop template: 3.25 in for one column,
6.75 in for the full width. Text is set in a Times-like font to match the paper.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from vla_stress.analysis.stats import bootstrap_x50, bootstrap_x50_interp, fit_dose_response, logistic, rate_table, x50_interp  # noqa: E402

COLORS = {"camera_orbit": "#2a78d6", "robot_init": "#eb6834", "light_dimming": "#1baf7a", "image_noise": "#b07c00"}
INK, GREY, LIGHT = "#1f1f1d", "#6b6a66", "#c9c8c2"
TITLES = {
    "camera_orbit": "Camera orbit",
    "robot_init": "Arm initial pose",
    "light_dimming": "Lighting",
    "image_noise": "Pixel noise",
}
XLABELS = {
    "camera_orbit": "orbit angle (deg)",
    "robot_init": "offset per joint (rad)",
    "light_dimming": "fraction of light removed",
    "image_noise": "noise std (0-255)",
}
BLUES = LinearSegmentedColormap.from_list("blues", ["#f7f9fc", "#c6dbf2", "#6aa2df", "#2a78d6", "#123f7a"])

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8984",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.grid": True,
        "grid.color": "#ebeae5",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def save(fig, path):
    fig.savefig(path)
    if str(path).endswith(".pdf"):
        fig.savefig(str(path)[:-4] + ".png", dpi=300)
    plt.close(fig)


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


# --------------------------------------------------------------------------- language
def _matrix(df, value, n):
    k = df.pivot_table(index="said", columns="task_id", values=value, aggfunc="sum").reindex(index=range(n), columns=range(n))
    c = df.pivot_table(index="said", columns="task_id", values=value, aggfunc="count").reindex(index=range(n), columns=range(n))
    return k, c


def language_matrix(df: pd.DataFrame, short_names: list[str], path):
    """Two matrices over (instruction i, environment j):
    (a) goal of the environment reached -> high off-diagonal = the robot ignores the instruction;
    (b) goal of the instruction reached -> high off-diagonal = the robot follows the instruction."""
    df = df.copy()
    n = len(short_names)
    df["said"] = df["variant"].str.replace("from_task_", "").astype(int)
    achieved = df["goals_achieved"].astype(str).str.split(";")
    df["env_goal"] = df["success"].astype(int)
    df["said_goal"] = [str(s) in a for s, a in zip(df["said"], achieved)]
    df["said_goal"] = df["said_goal"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.25), constrained_layout=True)
    mats = {}
    for ax, value, title, lab in zip(
        axes,
        ["env_goal", "said_goal"],
        ["Goal of the environment reached", "Goal of the instruction reached"],
        ["a", "b"],
    ):
        k, c = _matrix(df, value, n)
        m = k / c
        mats[value] = (k, c)
        im = ax.imshow(m.values, cmap=BLUES, vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                if not np.isnan(m.values[i, j]):
                    v = m.values[i, j]
                    ax.text(j, i, f"{int(k.values[i, j])}/{int(c.values[i, j])}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.55 else INK)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(range(n))
        ax.set_yticklabels([f"{i}  {short_names[i]}" for i in range(n)] if lab == "a" else range(n))
        ax.set_xlabel("environment (task $j$)")
        if lab == "a":
            ax.set_ylabel("instruction given (task $i$)")
        ax.set_title(title)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        panel_label(ax, lab)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("rate")
    cb.outline.set_visible(False)
    save(fig, path)
    return mats


def bars(table: pd.DataFrame, label_col: str, path, color="#2a78d6", ref: float | None = None, figsize=(3.25, 1.4), xlabel="success rate"):
    """Horizontal bars of success rate with Wilson 95% intervals and k/n on the right."""
    t = table.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["rate"], color=color, height=0.62, zorder=2)
    ax.errorbar(t["rate"], y, xerr=[t["rate"] - t["lo"], t["hi"] - t["rate"]], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=3)
    for yi, k, n in zip(y, t["k"], t["n"]):
        ax.text(1.01, yi, f"{k}/{n}", va="center", fontsize=6.5, color=GREY, transform=ax.get_yaxis_transform())
    if ref is not None:
        ax.axvline(ref, color=GREY, lw=0.7, ls=(0, (2, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(t[label_col])
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    save(fig, path)


# --------------------------------------------------------------------------- dose-response
def dose_response(curves: dict[str, pd.DataFrame], path, n_boot: int = 300):
    """One panel per family.
    Thin grey lines: individual tasks. Dots: success pooled over tasks with Wilson 95% CI.
    Solid line: logistic fit with s0 fixed to the baseline. Dotted line: half of the baseline.
    Shaded band and dashed line: breaking point x50 and its 95% task-bootstrap interval."""
    fams = list(curves)
    fig, axes = plt.subplots(1, len(fams), figsize=(6.75, 1.95), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    summary = []
    for ax, fam, lab in zip(axes, fams, "abcdef"):
        d = curves[fam]
        col = COLORS.get(fam, GREY)
        xmax = d["magnitude"].max()
        per_task = d.groupby(["task_id", "magnitude"]).success.mean().unstack(0)
        for tid in per_task.columns:
            ax.plot(per_task.index, per_task[tid], color=LIGHT, lw=0.5, zorder=1)
        fit = fit_dose_response(d["magnitude"].values, d["success"].values)
        boots = bootstrap_x50(d, n_boot=n_boot)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        x50 = fit["x50"]
        ax.axhline(fit["s0"] / 2, color=GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        if fit["reached"]:
            ax.axvspan(max(lo, 0), min(hi, xmax), color=col, alpha=0.13, lw=0, zorder=0)
            ax.axvline(x50, color=col, lw=0.8, ls=(0, (3, 2)), zorder=2)
            txt = f"$x_{{50}}$ = {x50:.3g} [{lo:.2g}, {min(hi, 3 * xmax):.2g}]"
        else:
            txt = f"$x_{{50}}$ > {xmax:g} (not reached)"
        xs = np.linspace(0, xmax, 300)
        ax.plot(xs, logistic(xs, fit["s0"], fit["x50"], fit["w"]), color=col, lw=1.4, zorder=3)
        t = rate_table(d, ["magnitude"])
        ax.errorbar(t["magnitude"], t["rate"], yerr=[t["rate"] - t["lo"], t["hi"] - t["rate"]], fmt="o", ms=3.2, mfc=col, mec="white", mew=0.6, ecolor=col, elinewidth=0.9, capsize=0, zorder=4)
        ax.text(0.97, 0.96, txt, transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK)
        ax.set_title(TITLES.get(fam, fam), loc="left")
        ax.set_xlabel(XLABELS.get(fam, fam))
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(-0.03 * xmax, 1.03 * xmax)
        panel_label(ax, lab)
        xi = x50_interp(d["magnitude"].values, d["success"].values)
        bi = bootstrap_x50_interp(d, n_boot=n_boot)
        bi_lo, bi_hi = np.percentile(bi, [2.5, 97.5])
        summary.append(
            {"family": fam, "s0": fit["s0"], "x50": x50, "x50_lo": lo, "x50_hi": hi, "reached": fit["reached"], "w": fit["w"],
             "x50_interp": xi, "x50_interp_lo": bi_lo, "x50_interp_hi": bi_hi, "n_per_point": int(t["n"].median()), "n_boot_valid": len(boots)}
        )
    axes[0].set_ylabel("success rate")
    save(fig, path)
    return pd.DataFrame(summary)


# --------------------------------------------------------------------------- residual RL
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def rl_summary(logs: dict[str, pd.DataFrame], groups: list[dict], path, train_ref: float | None = None, window: int = 5):
    """(a) PPO training success per seed (moving average); (b) evaluation success, VLA alone vs
    VLA + residual for each seed, on init states seen in training and on held-out ones.

    groups: [{"label": str, "vla": (k, n), "res": {seed: (k, n)}}]"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.1), gridspec_kw={"width_ratios": [1, 1.25]}, constrained_layout=True)
    for i, (name, lg) in enumerate(sorted(logs.items())):
        s = lg["success_rate"].rolling(window, min_periods=1).mean()
        ax1.plot(lg["step"] / 1000, s, color=SEED_COLORS[i % 3], lw=1.3, label=f"seed {name.rsplit('_s', 1)[-1]}")
    if train_ref is not None:
        ax1.axhline(train_ref, color=GREY, ls=(0, (2, 2)), lw=0.9, label="VLA alone (states 0\u20134)")
    ax1.set_xlabel("environment steps (thousands)")
    ax1.set_ylabel(f"training success ({window}-update avg)")
    ax1.set_ylim(0, 1)
    ax1.legend(frameon=False, loc="lower right", ncol=2)
    ax1.set_title("PPO training (init states 0\u201339)", loc="left")
    panel_label(ax1, "a")

    from vla_stress.analysis.stats import wilson

    x = np.arange(len(groups))
    seeds = sorted({sd for g in groups for sd in g["res"]})
    width = 0.8 / (1 + len(seeds))
    for j, who in enumerate(["vla"] + seeds):
        xs, ys, los, his = [], [], [], []
        for i, g in enumerate(groups):
            kn = g["vla"] if who == "vla" else g["res"].get(who)
            if kn is None:
                continue
            k, n = kn
            lo, hi = wilson(k, n)
            xs.append(i - 0.4 + width * (j + 0.5))
            ys.append(k / n)
            los.append(k / n - lo)
            his.append(hi - k / n)
        col = GREY if who == "vla" else SEED_COLORS[(j - 1) % 3]
        lab = "VLA alone" if who == "vla" else f"+ residual, seed {who[1:]}"
        ax2.bar(xs, ys, width=width * 0.92, color=col, alpha=0.9 if who != "vla" else 0.55, label=lab, zorder=2)
        ax2.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([g["label"] for g in groups])
    ax2.set_ylim(0, 1.32)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_ylabel("success rate")
    ax2.grid(axis="x", visible=False)
    ax2.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=len(seeds) + 1, handlelength=1.2, columnspacing=1.0)
    ax2.set_title("Deterministic evaluation, hard resets", loc="left")
    panel_label(ax2, "b")
    save(fig, path)
