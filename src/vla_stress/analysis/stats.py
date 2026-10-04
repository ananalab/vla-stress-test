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


def fit_dose_response(x: np.ndarray, y: np.ndarray) -> dict:
    """Binomial MLE of the 3-parameter decreasing logistic. x in physical units, y in {0, 1}."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xmax = x.max() if x.max() > 0 else 1.0

    def nll(theta):
        s0 = 1 / (1 + np.exp(-theta[0]))  # keep s0 in (0, 1)
        x50, w = theta[1], np.exp(theta[2])
        p = np.clip(logistic(x, s0, x50, w), 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    s0_init = np.clip(y[x == x.min()].mean(), 0.05, 0.95)
    best = None
    # A few starting points: the likelihood is flat when the curve never drops.
    for x50_init in np.linspace(0.2, 1.5, 6) * xmax:
        theta0 = [np.log(s0_init / (1 - s0_init)), x50_init, np.log(0.15 * xmax)]
        r = optimize.minimize(nll, theta0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    s0 = 1 / (1 + np.exp(-best.x[0]))
    return {"s0": s0, "x50": best.x[1], "w": np.exp(best.x[2]), "nll": best.fun}


def bootstrap_x50(df: pd.DataFrame, x_col: str = "magnitude", n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Resample whole tasks with replacement, refit, return the x50 samples."""
    rng = np.random.default_rng(seed)
    tasks = df["task_id"].unique()
    by_task = {t: df[df["task_id"] == t] for t in tasks}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(tasks, size=len(tasks), replace=True)
        d = pd.concat([by_task[t] for t in pick])
        out.append(fit_dose_response(d[x_col].values, d["success"].values)["x50"])
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
