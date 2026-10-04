"""Build every figure and summary table from results/*.csv.

    python scripts/make_figures.py

Writes report/figures/*.pdf (+ .png) and results/summary/*.csv. Each number quoted in
the README or the report is read from one of these summary files. Missing result
files are skipped, so this can run while experiments are still going.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vla_stress.analysis import plots
from vla_stress.analysis.stats import mcnemar_exact, rate_table

R = Path("results")
FIG = Path("report/figures")
SUM = R / "summary"
FIG.mkdir(parents=True, exist_ok=True)
SUM.mkdir(parents=True, exist_ok=True)

GOAL_SHORT = [
    "open middle drawer",
    "bowl \u2192 stove",
    "bottle \u2192 cabinet top",
    "bowl \u2192 top drawer",
    "bowl \u2192 cabinet top",
    "push plate to stove",
    "cheese \u2192 bowl",
    "turn on stove",
    "bowl \u2192 plate",
    "bottle \u2192 rack",
]
# Extra configs that refine or extend a family (same init states / seeds, more levels).
EXTRA = {"robot_init": ["dose_robot_init_fine"], "light_dimming": ["dose_light_dimming_fine"], "camera_orbit": ["dose_camera_orbit_wide"]}


def load(name: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(R / f"{name}.csv")) + glob.glob(str(R / f"{name}.shard*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
    for c in ["success", "task_id", "episode", "steps"]:
        df[c] = df[c].astype(int)
    df["magnitude"] = df["magnitude"].astype(float)
    df["intensity"] = df["intensity"].astype(float)
    return df


def wilson_str(k, n):
    from vla_stress.analysis.stats import wilson

    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} [{lo:.0%}, {hi:.0%}]"


headline = {}

# ---------------------------------------------------------------- baselines
for suite in ["goal", "spatial"]:
    b = load(f"baseline_{suite}")
    if b is None:
        continue
    t = rate_table(b, ["task_id"])
    t.to_csv(SUM / f"baseline_{suite}_per_task.csv", index=False)
    headline[f"baseline_{suite}"] = wilson_str(b.success.sum(), len(b))
    headline[f"baseline_{suite}_sec_per_episode"] = round(b.duration_s.astype(float).mean(), 1)

# ---------------------------------------------------------------- language
lm = load("language_matrix")
if lm is not None:
    mats = plots.language_matrix(lm, GOAL_SHORT, FIG / "language_matrix.pdf")
    lm["said"] = lm["variant"].str.replace("from_task_", "").astype(int)
    lm["said_goal"] = [str(s) in str(a).split(";") for s, a in zip(lm["said"], lm["goals_achieved"])]
    diag = lm[lm.said == lm.task_id]
    off = lm[lm.said != lm.task_id]
    headline["language_diagonal"] = wilson_str(diag.success.sum(), len(diag))
    headline["language_off_diag_env_goal"] = wilson_str(off.success.sum(), len(off))
    headline["language_off_diag_instructed_goal"] = wilson_str(int(off.said_goal.sum()), len(off))
    # other goals reached that are neither the instructed one nor the env one
    other = [len(set(str(a).split(";")) - {"", str(s), str(t)}) > 0 for a, s, t in zip(off.goals_achieved, off.said, off.task_id)]
    headline["language_off_diag_some_other_goal"] = wilson_str(int(np.sum(other)), len(off))
    lm.groupby(["said"]).said_goal.mean().to_csv(SUM / "language_instructed_goal_by_instruction.csv")

lv = load("language_variants")
bg = load("baseline_goal")
if lv is not None and bg is not None:
    ref = bg[bg.episode < lv.episode.max() + 1].assign(condition="original", variant="original")
    allv = pd.concat([ref, lv], ignore_index=True)
    allv["label"] = allv["condition"].replace({"paraphrase": "paraphrase"})
    t = rate_table(allv, ["condition"])
    order = ["original", "paraphrase", "empty", "absurd"]
    t["order"] = t["condition"].map({k: i for i, k in enumerate(order)})
    t = t.sort_values("order")
    t.to_csv(SUM / "language_variants.csv", index=False)
    names = {"original": "original instruction", "paraphrase": "paraphrase (3 per task)", "empty": "empty string", "absurd": '"sing a song"'}
    t["name"] = t["condition"].map(names)
    plots.bars(t, "name", FIG / "language_variants.pdf", figsize=(3.25, 1.2))
    # paired test vs original instruction, same task/episode
    tests = {}
    for cond in ["empty", "absurd"]:
        x = lv[lv.condition == cond].merge(ref, on=["task_id", "episode"], suffixes=("", "_ref"))
        tests[cond] = mcnemar_exact(x["success_ref"], x["success"])
    headline["language_variant_tests"] = tests
    pp = lv[lv.condition == "paraphrase"]
    headline["paraphrase_by_index"] = pp.groupby("variant").success.mean().round(2).to_dict()

# ---------------------------------------------------------------- dose-response
bs = load("baseline_spatial")
curves = {}
for fam in ["camera_orbit", "robot_init", "light_dimming", "image_noise"]:
    parts = [load(n) for n in [f"dose_{fam}"] + EXTRA.get(fam, [])]
    parts = [p for p in parts if p is not None]
    if not parts or bs is None:
        continue
    d = pd.concat(parts, ignore_index=True)
    # Intensity 0 = the baseline episodes with the same init states and seeds.
    zero = bs[bs.episode.isin(d.episode.unique())].copy()
    zero["magnitude"] = 0.0
    zero["perturbation"] = fam
    curves[fam] = pd.concat([zero, d], ignore_index=True)
    rate_table(curves[fam], ["magnitude"]).to_csv(SUM / f"dose_{fam}.csv", index=False)
    rate_table(curves[fam], ["task_id", "magnitude"]).to_csv(SUM / f"dose_{fam}_per_task.csv", index=False)
    if fam == "camera_orbit":
        cam = d.copy()
        cam["sign"] = cam["sign"].astype(int)
        rate_table(cam, ["magnitude", "sign"]).to_csv(SUM / "dose_camera_orbit_by_direction.csv", index=False)
if curves:
    s = plots.dose_response(curves, FIG / "dose_response.pdf")
    s.to_csv(SUM / "breaking_points.csv", index=False)
    headline["breaking_points"] = s.round(3).to_dict(orient="records")

cm = load("camera_mask")
if cm is not None and bs is not None:
    ref = bs[bs.episode.isin(cm.episode.unique())].assign(condition="both cameras")
    t = rate_table(pd.concat([ref, cm]), ["condition"])
    names = {"both cameras": "both cameras", "mask_agentview": "agentview masked", "mask_wrist": "wrist masked"}
    t["name"] = t["condition"].map(names)
    t.to_csv(SUM / "camera_mask.csv", index=False)
    plots.bars(t, "name", FIG / "camera_mask.pdf", figsize=(3.25, 1.0))

# ---------------------------------------------------------------- residual RL
runs = sorted(glob.glob("runs/*/log.csv"))
if runs:
    groups = {}
    for f in runs:
        name = Path(f).parent.name
        groups.setdefault(name.rsplit("_s", 1)[0], {})[name] = pd.read_csv(f)
    for g, logs in groups.items():
        base = json.load(open(SUM / f"rl_{g}_baseline.json"))["rate"] if (SUM / f"rl_{g}_baseline.json").exists() else None
        plots.learning_curves(logs, FIG / f"rl_curve_{g}.pdf", baseline=base)

json.dump(headline, open(SUM / "headline.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(headline, indent=2, default=str))
