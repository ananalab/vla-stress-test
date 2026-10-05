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
if lv is not None:
    # Reference with the original wording, same tasks and init states: baseline_goal if it covers
    # all tasks, otherwise the diagonal of the swap matrix (episodes 0-2).
    bg = load("baseline_goal")
    if bg is not None and bg.task_id.nunique() == 10:
        ref = bg[bg.episode < lv.episode.max() + 1].copy()
        ref_name = "original wording"
    else:
        ref = lm[lm.said == lm.task_id].copy()
        ref_name = "original wording (eps 0-2)"
    ref["label"] = ref_name
    lv = lv.copy()
    names = {"paraphrase_0": "paraphrase, close", "paraphrase_1": "paraphrase, reworded", "paraphrase_2": "paraphrase, distant", "empty": "empty string", "absurd": "\u201csing a song\u201d"}
    lv["label"] = lv["variant"].map(names)
    allv = pd.concat([ref[["label", "success", "task_id", "episode"]], lv[["label", "success", "task_id", "episode"]]])
    t = rate_table(allv, ["label"])
    order = [ref_name] + list(names.values())
    t["o"] = t["label"].map({k: i for i, k in enumerate(order)})
    t = t.sort_values("o")
    t.to_csv(SUM / "language_variants.csv", index=False)
    plots.bars(t, "label", FIG / "language_variants.pdf", figsize=(3.25, 1.45))
    pp = lv[lv.condition == "paraphrase"]
    headline["language_paraphrase"] = wilson_str(int(pp.success.sum()), len(pp))
    headline["language_paraphrase_by_index"] = pp.groupby("variant").success.sum().to_dict()
    headline["language_empty"] = wilson_str(int(lv[lv.condition == "empty"].success.sum()), int((lv.condition == "empty").sum()))
    headline["language_absurd"] = wilson_str(int(lv[lv.condition == "absurd"].success.sum()), int((lv.condition == "absurd").sum()))
    failed = lv[lv.success == 0]
    headline["language_variants_failed_with_any_goal"] = int((failed.goals_achieved.astype(str) != "").sum())

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
rl_vla = load("rl_eval_vla")
rl_tr_vla = load("rl_eval_train_vla")
rl_res = {Path(f).stem.split("_")[-1]: load(Path(f).stem) for f in sorted(glob.glob(str(R / "rl_eval_residual_s*.csv")))}
rl_tr_res = {Path(f).stem.split("_")[-1]: load(Path(f).stem) for f in sorted(glob.glob(str(R / "rl_eval_train_residual_s*.csv")))}
logs = {}
for f in sorted(glob.glob("runs/robot025_task2_s*/log.csv")):
    try:
        lg = pd.read_csv(f)
    except pd.errors.EmptyDataError:
        continue  # run just started
    if len(lg) >= 40:  # finished runs only
        logs[Path(f).parent.name] = lg
rl_rows, groups, rl_steps = [], [], {}
if rl_vla is not None and rl_res:
    def kn(d, mag):
        g = d[np.isclose(d.magnitude, mag)]
        return (int(g.success.sum()), len(g))

    def paired(v, r, mag):
        m = v[np.isclose(v.magnitude, mag)].merge(r[np.isclose(r.magnitude, mag)], on="episode", suffixes=("_v", "_r"))
        return m

    if rl_tr_vla is not None:
        groups.append({"label": "seen states\n0.05 rad", "vla": kn(rl_tr_vla, 0.05), "res": {sd: kn(d, 0.05) for sd, d in rl_tr_res.items()}})
    for mag, lab in [(0.05, "unseen states\n0.05 rad"), (0.0, "unseen states\nno offset"), (0.1, "unseen states\n0.10 rad")]:
        groups.append({"label": lab, "vla": kn(rl_vla, mag), "res": {sd: kn(d, mag) for sd, d in rl_res.items()}})
    for g in groups:
        row = {"group": g["label"].replace("\n", " "), "vla": f"{g['vla'][0]}/{g['vla'][1]}"}
        for sd, (k, n) in g["res"].items():
            row[sd] = f"{k}/{n}"
        rl_rows.append(row)
    # paired tests and episode length of successes
    for sd in rl_res:
        for split, v, r in [("train", rl_tr_vla, rl_tr_res.get(sd)), ("heldout", rl_vla, rl_res[sd])]:
            if v is None or r is None:
                continue
            m = paired(v, r, 0.05)
            rl_steps[(sd, split)] = (m[m.success_v == 1].steps_v.mean(), m[m.success_r == 1].steps_r.mean(), mcnemar_exact(m.success_v, m.success_r)["p"])
    pd.DataFrame(rl_rows).to_csv(SUM / "rl_eval.csv", index=False)
    pd.DataFrame([{"seed": k[0], "split": k[1], "mean_steps_success_vla": v[0], "mean_steps_success_res": v[1], "mcnemar_p": v[2]} for k, v in rl_steps.items()]).to_csv(SUM / "rl_paired.csv", index=False)
    headline["rl_eval"] = rl_rows
    d = load("dose_robot_init")
    ref = d[(d.task_id == 2) & np.isclose(d.magnitude, 0.05)].success.mean() if d is not None else None
    plots.rl_summary(logs, groups, FIG / "rl_residual.pdf", train_ref=ref)

json.dump(headline, open(SUM / "headline.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(headline, indent=2, default=str))

# ---------------------------------------------------------------- LaTeX macros
# Every number in the report text is one of these macros, recomputed from results/.
def pct(k, n):
    return f"{100 * k / n:.0f}\\%"


def frac(k, n):
    from vla_stress.analysis.stats import wilson

    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}\\%, CI {100 * lo:.0f}--{100 * hi:.0f})"


macros = {}
all_csv = [f for f in glob.glob(str(R / "*.csv"))]
n_ep = 0
for f in all_csv:
    try:
        d = pd.read_csv(f)
    except pd.errors.EmptyDataError:
        continue
    if "success" in d.columns:  # episode logs only
        n_ep += len(d)
macros["TotalEpisodes"] = f"{n_ep:,}".replace(",", "\\,")
for suite in ["goal", "spatial"]:
    b = load(f"baseline_{suite}")
    if b is not None:
        macros[f"Base{suite.capitalize()}"] = frac(int(b.success.sum()), len(b))
        macros[f"Base{suite.capitalize()}Pct"] = pct(int(b.success.sum()), len(b))
if lm is not None:
    macros["LangDiag"] = frac(int(diag.success.sum()), len(diag))
    macros["LangOffEnv"] = pct(int(off.success.sum()), len(off))
    macros["LangOffEnvFrac"] = f"{int(off.success.sum())}/{len(off)}"
    macros["LangOffInstructed"] = pct(int(off.said_goal.sum()), len(off))
    macros["LangOffInstructedFrac"] = frac(int(off.said_goal.sum()), len(off))
    macros["LangOther"] = f"{int(np.sum(other))}/{len(off)}"
if curves:
    names = {"camera_orbit": "Cam", "robot_init": "Robot", "light_dimming": "Light", "image_noise": "Noise"}
    for r in s.to_dict(orient="records"):
        n = names[r["family"]]
        macros[f"{n}Xfifty"] = f"{r['x50']:.2g}" if r["reached"] else f">{curves[r['family']]['magnitude'].max():g}"
        macros[f"{n}XfiftyCI"] = f"[{max(r['x50_lo'], 0):.2g}, {r['x50_hi']:.2g}]"
        xi_max = curves[r["family"]]["magnitude"].max()
        macros[f"{n}XfiftyInterp"] = f"{r['x50_interp']:.2g}" if np.isfinite(r["x50_interp"]) else f">{xi_max:g}"
        hi_txt = f"{r['x50_interp_hi']:.2g}" if np.isfinite(r["x50_interp_hi"]) else f">{xi_max:g}"
        macros[f"{n}XfiftyInterpCI"] = f"[{r['x50_interp_lo']:.2g}, {hi_txt}]"
# lerobot-eval cross-check (same tasks, unpaired seeds)
chk = R / "lerobot_eval_check.json"
if chk.exists() and bs is not None:
    info = json.load(open(chk))
    k = sum(sum(t["metrics"]["successes"]) for t in info["per_task"])
    n = sum(len(t["metrics"]["successes"]) for t in info["per_task"])
    ids = [t["task_id"] for t in info["per_task"]]
    ours = bs[bs.task_id.isin(ids)]
    macros["LerobotEvalCheck"] = f"{k}/{n} ({pct(k, n)})"
    macros["OurEvalCheck"] = f"{int(ours.success.sum())}/{len(ours)} ({pct(int(ours.success.sum()), len(ours))})"
    from scipy.stats import fisher_exact

    macros["EvalCheckP"] = f"{fisher_exact([[k, n - k], [int(ours.success.sum()), len(ours) - int(ours.success.sum())]])[1]:.2f}"
# where the lighting cliff is: last level keeping >= 75 % of baseline, first level <= 25 %
if "light_dimming" in curves:
    t = rate_table(curves["light_dimming"], ["magnitude"]).sort_values("magnitude")
    s0 = t.loc[t.magnitude == 0, "rate"].item()
    keep = t[t.rate >= 0.75 * s0].magnitude.max()
    lost = t[(t.rate <= 0.25 * s0) & (t.magnitude > keep)].magnitude.min()
    macros["LightCliffLow"] = f"{100 * keep:g}\\%"
    macros["LightCliffHigh"] = f"{100 * lost:g}\\%"
if lv is not None:
    pp = lv[lv.condition == "paraphrase"]
    macros["LangPara"] = frac(int(pp.success.sum()), len(pp))
    for k, nm in zip(["paraphrase_0", "paraphrase_1", "paraphrase_2"], ["Close", "Reworded", "Distant"]):
        g = pp[pp.variant == k]
        macros[f"LangPara{nm}"] = f"{int(g.success.sum())}/{len(g)}"
    for c in ["empty", "absurd"]:
        g = lv[lv.condition == c]
        macros[f"Lang{c.capitalize()}"] = f"{int(g.success.sum())}/{len(g)}"
    macros["LangVariantsFailedAnyGoal"] = str(int((lv[lv.success == 0].goals_achieved.astype(str) != "").sum()))
    macros["LangVariantsFailed"] = str(int((lv.success == 0).sum()))
if rl_rows:
    seeds = sorted(rl_res)
    macros["RLSeeds"] = str(len(seeds))
    names = {"seen states 0.05 rad": "Seen", "unseen states 0.05 rad": "Unseen", "unseen states no offset": "UnseenClean", "unseen states 0.10 rad": "UnseenStrong"}
    for row in rl_rows:
        tag = names[row["group"]]
        macros[f"RL{tag}Vla"] = row["vla"]
        macros[f"RL{tag}Res"] = ", ".join(row[sd] for sd in seeds if sd in row)
    tr = [rl_steps[(sd, "train")] for sd in seeds if (sd, "train") in rl_steps]
    ho = [rl_steps[(sd, "heldout")] for sd in seeds if (sd, "heldout") in rl_steps]
    if tr:
        macros["RLStepsSeenVla"] = f"{tr[0][0]:.0f}"
        macros["RLStepsSeenRes"] = "--".join(sorted({f"{t[1]:.0f}" for t in tr}))
        macros["RLPSeen"] = ", ".join(f"{t[2]:.2f}" for t in tr)
    if ho:
        macros["RLStepsUnseenVla"] = f"{ho[0][0]:.0f}"
        macros["RLStepsUnseenRes"] = "--".join(sorted({f"{t[1]:.0f}" for t in ho}))
    if logs:
        first = [lg.success_rate.head(10).mean() for lg in logs.values()]
        last = [lg.success_rate.tail(10).mean() for lg in logs.values()]
        macros["RLTrainFirst"] = ", ".join(f"{v:.2f}" for v in first)
        macros["RLTrainLast"] = ", ".join(f"{v:.2f}" for v in last)
rp = load("rl_eval_replan10")
if rp is not None and rl_vla is not None:
    for mag, tag in [(0.0, "Clean"), (0.05, "Train"), (0.1, "Strong")]:
        g = rp[np.isclose(rp.magnitude, mag)]
        macros[f"RLReplan{tag}"] = f"{int(g.success.sum())}/{len(g)}"
    macros["RLReplanSlowdown"] = f"{rp.duration_s.astype(float).mean() / rl_vla.duration_s.astype(float).mean():.1f}"
with open("report/numbers.tex", "w") as f:
    f.write("% generated by scripts/make_figures.py from results/ -- do not edit by hand\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("wrote report/numbers.tex with", len(macros), "macros")
