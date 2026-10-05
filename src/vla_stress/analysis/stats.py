"""Statistics used in the report.

- Wilson score interval for a success rate. With 10 episodes and p close to 0 or 1
  the usual p +/- 1.96 sqrt(p(1-p)/n) collapses to zero width or leaves [0, 1];
  Wilson does neither.
- Dose-response fit: success(x) = s0 / (1 + exp((x - x50) / w)), fitted by binomial
  maximum likelihood. x50 is the intensity at which success falls to half of its
  unperturbed value (the "breaking point").
- Cluster bootstrap over tasks for the uncertainty of x50: tasks differ a lot in
  difficulty, so resampling episodes alone would understate the uncertainty.
- Exact McNemar test for paired conditions (same init state and seed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by)["success"].agg(["sum", "count"]).reset_index()
    g["rate"] = g["sum"] / g["count"]
    ci = [wilson(k, n) for k, n in zip(g["sum"], g["count"])]
    g["lo"] = [c[0] for c in ci]
    g["hi"] = [c[1] for c in ci]
    return g.rename(columns={"sum": "k", "count": "n"})


def logistic(x, s0, x50, w):
    return s0 / (1.0 + np.exp((x - x50) / w))


def fit_dose_response(x: np.ndarray, y: np.ndarray, s0: float | None = None) -> dict:
    """Binomial MLE of the decreasing logistic, with s0 fixed to the unperturbed success rate.

    Leaving s0 free made the fit wander off (s0 -> 1 with a very wide slope) when the
    curve declines slowly. Fixing it to the observed rate at x = 0 makes x50 mean
    exactly what we want: the intensity at which success is half of the baseline.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xmax = x.max() if x.max() > 0 else 1.0
    if s0 is None:
        s0 = float(np.clip(y[x == 0].mean() if np.any(x == 0) else y[x == x.min()].mean(), 1e-3, 1 - 1e-3))

    def nll(theta):
        x50, w = theta[0], np.exp(theta[1])
        p = np.clip(logistic(x, s0, x50, w), 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    best = None
    for x50_init in (0.3 * xmax, 0.8 * xmax, 2.0 * xmax):
        r = optimize.minimize(nll, [x50_init, np.log(0.2 * xmax)], method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    # Beyond ~3x the tested range the estimate is pure extrapolation; cap it.
    x50 = float(np.clip(best.x[0], 0.0, 3 * xmax))
    return {"s0": s0, "x50": x50, "w": float(np.exp(best.x[1])), "nll": float(best.fun), "reached": bool(x50 <= xmax)}


def bootstrap_x50(df: pd.DataFrame, x_col: str = "magnitude", n_boot: int = 500, seed: int = 0) -> np.ndarray:
    """Resample whole tasks with replacement, refit (s0 re-estimated), return the x50 samples."""
    rng = np.random.default_rng(seed)
    tasks = df["task_id"].unique()
    by_task = {t: df[df["task_id"] == t] for t in tasks}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(tasks, size=len(tasks), replace=True)
        d = pd.concat([by_task[t] for t in pick])
        if d.loc[d[x_col] == 0, "success"].mean() == 0:
            continue  # all resampled tasks fail even without perturbation: no breaking point
        out.append(fit_dose_response(d[x_col].values, d["success"].values)["x50"])
    return np.array(out)


def x50_interp(x: np.ndarray, y: np.ndarray) -> float:
    """Model-free breaking point: pooled success per level, made non-increasing by
    pool-adjacent-violators, then linearly interpolated where it crosses half the
    unperturbed rate. Returns inf if it never gets there."""
    levels = np.unique(x)
    rates = np.array([y[x == lv].mean() for lv in levels])
    counts = np.array([np.sum(x == lv) for lv in levels], float)
    # pool adjacent violators for a non-increasing sequence
    blocks = [[r, c, 1] for r, c in zip(rates, counts)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            r1, c1, n1 = blocks[i]
            r2, c2, n2 = blocks.pop(i + 1)
            blocks[i] = [(r1 * c1 + r2 * c2) / (c1 + c2), c1 + c2, n1 + n2]
            i = max(i - 1, 0)
        else:
            i += 1
    iso = np.concatenate([[b[0]] * b[2] for b in blocks])
    target = iso[0] / 2
    below = np.where(iso <= target)[0]
    if len(below) == 0:
        return float("inf")
    k = below[0]
    if k == 0:
        return float(levels[0])
    x0, x1, y0, y1 = levels[k - 1], levels[k], iso[k - 1], iso[k]
    return float(x0 + (y0 - target) * (x1 - x0) / (y0 - y1)) if y0 != y1 else float(x1)


def bootstrap_x50_interp(df: pd.DataFrame, x_col: str = "magnitude", n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tasks = df["task_id"].unique()
    by_task = {t: df[df["task_id"] == t] for t in tasks}
    out = []
    for _ in range(n_boot):
        d = pd.concat([by_task[t] for t in rng.choice(tasks, size=len(tasks), replace=True)])
        out.append(x50_interp(d[x_col].values, d["success"].values))
    return np.array(out)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired binary outcomes a, b (same episodes). Two-sided exact test on discordant pairs."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    p = 1.0 if n == 0 else stats.binomtest(n10, n, 0.5).pvalue
    return {"a_only": n10, "b_only": n01, "p": p}
