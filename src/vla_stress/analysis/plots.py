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
RAND_COLORS = ["#4a3aa7", "#e87ba4"]


def _series(name: str):
    """Colour, line style and label for a residual run key: 's0'..'s2' (fixed offset signs) or 'rand_s0'.."""
    k = int(name[-1])
    if "rand" in name:
        return RAND_COLORS[k % 2], (0, (3, 1.5)), f"random signs, seed {k}"
    return SEED_COLORS[k % 3], "-", f"seed {k}"


def rl_summary(logs: dict[str, pd.DataFrame], groups: list[dict], path, train_ref: float | None = None, window: int = 5):
    """(a) PPO training success per seed (moving average); (b) evaluation success, VLA alone vs
    VLA + residual for each seed, on init states seen in training and on held-out ones.

    groups: [{"label": str, "vla": (k, n), "res": {seed: (k, n)}}]"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.1), gridspec_kw={"width_ratios": [1, 1.25]}, constrained_layout=True)
    for name, lg in sorted(logs.items(), key=lambda kv: ("rand" in kv[0], kv[0])):
        s = lg["success_rate"].rolling(window, min_periods=1).mean()
        col, ls, lab = _series(name)
        ax1.plot(lg["step"] / 1000, s, color=col, ls=ls, lw=1.2, label=lab)
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
    seeds = sorted({sd for g in groups for sd in g["res"]}, key=lambda k: ("rand" in k, k))
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
        col, _, lab = (GREY, "-", "VLA alone") if who == "vla" else _series(who)
        ax2.bar(xs, ys, width=width * 0.92, color=col, alpha=0.9 if who != "vla" else 0.55, label=lab, zorder=2)
        ax2.errorbar(xs, ys, yerr=[np.maximum(los, 0), np.maximum(his, 0)], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([g["label"] for g in groups])
    ax2.set_ylim(0, 1.42)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_ylabel("success rate")
    ax2.grid(axis="x", visible=False)
    ax2.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=3, handlelength=1.2, columnspacing=1.0)
    ax2.set_title("Deterministic evaluation, hard resets", loc="left")
    panel_label(ax2, "b")
    save(fig, path)


def dose_per_task(curves: dict[str, pd.DataFrame], path):
    """Success per task (rows) and level (columns) for each family: shows how much of the pooled
    curve is a few tasks breaking early versus all tasks degrading together."""
    fams = list(curves)
    fig, axes = plt.subplots(1, len(fams), figsize=(6.75, 1.95), constrained_layout=True,
                             gridspec_kw={"width_ratios": [curves[f].magnitude.nunique() for f in fams]})
    for ax, fam, lab in zip(np.atleast_1d(axes), fams, "abcdef"):
        d = curves[fam]
        m = d.pivot_table(index="task_id", columns="magnitude", values="success", aggfunc="mean")
        im = ax.imshow(m.values, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(m.shape[1]))
        fmt = {"camera_orbit": "{:g}", "robot_init": "{:g}", "light_dimming": "{:.2f}", "image_noise": "{:g}"}[fam]
        ax.set_xticklabels([fmt.format(c) for c in m.columns], rotation=90, fontsize=5.5)
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels(m.index if lab == "a" else [], fontsize=6)
        ax.set_xlabel(XLABELS[fam], fontsize=7)
        if lab == "a":
            ax.set_ylabel("LIBERO-Spatial task")
        ax.set_title(TITLES[fam], loc="left")
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        panel_label(ax, lab)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("success rate")
    cb.outline.set_visible(False)
    save(fig, path)


# --------------------------------------------------------------------------- language: canonicalisation
def canonicalisation(panels: list[dict], sim: pd.DataFrame | None, path):
    """panels: [{"title", "levels": [label...], "raw": [(k, n)...], "canon": [(k, n)...], "ref": (k, n)}].
    sim: one row per paraphrase with columns similarity, rate, suite (raw paraphrases)."""
    from vla_stress.analysis.stats import wilson

    n = len(panels) + (sim is not None)
    fig, axes = plt.subplots(1, n, figsize=(6.75, 2.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [len(p["levels"]) + 0.6 for p in panels] + ([3.2] if sim is not None else [])})
    axes = np.atleast_1d(axes)
    for ax, p, lab in zip(axes, panels, "abc"):
        x = np.arange(len(p["levels"]))
        for j, (key, col, name) in enumerate([("raw", LIGHT, "paraphrase as written"), ("canon", "#2a78d6", "after canonicalisation")]):
            ks = p[key]
            r = [k / m for k, m in ks]
            ci = [wilson(k, m) for k, m in ks]
            xs = x + (j - 0.5) * 0.38
            ax.bar(xs, r, width=0.36, color=col, label=name, zorder=2)
            ax.errorbar(xs, r, yerr=[[max(a - c[0], 0) for a, c in zip(r, ci)], [max(c[1] - a, 0) for a, c in zip(r, ci)]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2, zorder=3)
        k, m = p["ref"]
        ax.axhline(k / m, color=GREY, ls=(0, (2, 2)), lw=0.9, label="original wording", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(p["levels"])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        ax.set_title(p["title"], loc="left")
        panel_label(ax, lab)
        if lab == "a":
            ax.set_ylabel("success rate")
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0.06, 1.0), ncol=3,
                       handlelength=1.4, columnspacing=1.2)
    if sim is not None:
        ax = axes[-1]
        for suite, mk, col in [("libero_goal", "o", "#2a78d6"), ("libero_spatial", "s", "#eb6834")]:
            d = sim[sim.suite == suite]
            ax.scatter(d.similarity, d.rate + np.random.default_rng(0).uniform(-0.015, 0.015, len(d)), s=12, marker=mk, color=col,
                       edgecolor="white", linewidth=0.4, label=suite.replace("libero_", "LIBERO-").title().replace("Libero", "LIBERO"), zorder=3)
        ax.set_xlabel("similarity to original instruction")
        ax.set_ylabel("success, paraphrase as written")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Is closeness enough?", loc="left")
        ax.legend(frameon=False, fontsize=6, loc="upper left")
        panel_label(ax, "abc"[len(panels)])
    save(fig, path)


# --------------------------------------------------------------------------- QD failure search
def qd_summary(rows: pd.DataFrame, validation: pd.DataFrame | None, grid: int, path):
    """(a) MAP-Elites archive: most harmful genome found in each cell. (b) Held-out validation of elites."""
    from vla_stress.analysis.stats import wilson

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.35), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.45]})
    ax = axes[0]
    m = np.full((grid, grid), np.nan)
    nf = np.zeros((grid, grid), int)
    for _, r in rows.iterrows():
        i, j = map(int, r["cell"].split(","))
        if np.isnan(m[j, i]) or r.fitness > m[j, i]:
            m[j, i], nf[j, i] = r.fitness, r.n_fail
    im = ax.imshow(m, origin="lower", cmap=BLUES, vmin=0.3, vmax=1.0, extent=(0, 1, 0, 1))
    for i in range(grid):
        for j in range(grid):
            if not np.isnan(m[j, i]):
                ax.text((i + 0.5) / grid, (j + 0.5) / grid, f"{nf[j, i]}/4", ha="center", va="center", fontsize=6,
                        color="white" if m[j, i] > 0.7 else INK)
    ax.scatter(rows.desc_geo, rows.desc_photo, s=2, color=INK, alpha=0.22, zorder=3)
    ax.set_xlabel("geometric intensity (camera, arm)")
    ax.set_ylabel("photometric intensity (light, noise)")
    ax.set_title(f"Archive ({len(rows)} evaluations)", loc="left")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("harm (1 = all fail)")
    cb.outline.set_visible(False)
    panel_label(ax, "a")

    ax = axes[1]
    if validation is not None and len(validation):
        g = validation.groupby("eval_id")
        t = g.agg(k=("success", lambda s: int((s == 0).sum())), n=("success", "size"), cam=("camera_deg", "first"), arm=("arm_rad", "first"),
                  light=("light_removed", "first"), noise=("noise_std", "first")).reset_index().sort_values("k")
        y = np.arange(len(t))
        rates = t.k / t.n
        ci = [wilson(int(a), int(b)) for a, b in zip(t.k, t.n)]
        ax.barh(y, rates, color="#eb6834", height=0.6, zorder=2)
        ax.errorbar(rates, y, xerr=[np.maximum(rates - [c[0] for c in ci], 0), np.maximum([c[1] for c in ci] - rates, 0)], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{c:.1f}°, {a:.3f} rad, {100 * l:.0f}% off, σ={s:.0f}" for c, a, l, s in zip(t.cam, t.arm, t.light, t.noise)], fontsize=6)
        for yi, k, n in zip(y, t.k, t.n):
            ax.text(1.01, yi, f"{k}/{n}", va="center", fontsize=6, color=GREY, transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1)
        ax.set_xlabel("failure rate on held-out episodes")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    ax.set_title("Validation (each perturbation alone: 0 failures)", loc="left")
    panel_label(ax, "b")
    save(fig, path)
