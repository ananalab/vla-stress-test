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
from vla_stress.analysis.results import load
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
        lang_ref = bg[bg.episode < lv.episode.max() + 1].copy()
        ref_name = "original wording"
    else:
        lang_ref = lm[lm.said == lm.task_id].copy()
        ref_name = "original wording (eps 0-2)"
    lang_ref["label"] = ref_name
    lv = lv.copy()
    names = {"paraphrase_0": "paraphrase, close", "paraphrase_1": "paraphrase, reworded", "paraphrase_2": "paraphrase, distant", "empty": "empty string", "absurd": "\u201csing a song\u201d"}
    lv["label"] = lv["variant"].map(names)
    allv = pd.concat([lang_ref[["label", "success", "task_id", "episode"]], lv[["label", "success", "task_id", "episode"]]])
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

# ---------------------------------------------------------------- canonicalisation
lc, sp, sc = load("language_canonical"), load("language_spatial_paraphrases"), load("language_spatial_canonical")
bs = load("baseline_spatial")
canon_panels, sim_df, canon_stats = [], None, {}


def kn(d):
    return (int(d.success.sum()), len(d))


if lc is not None and lv is not None:
    levels = ["close", "reworded", "distant"]
    canon_panels.append({
        "title": "LIBERO-Goal (3 paraphrases per task)", "levels": levels,
        "raw": [kn(lv[lv.variant == f"paraphrase_{i}"]) for i in range(3)],
        "canon": [kn(lc[lc.variant == f"paraphrase_{i}"]) for i in range(3)],
        "ref": kn(lang_ref),
    })
if sp is not None and sc is not None and bs is not None:
    canon_panels.append({
        "title": "LIBERO-Spatial (2 per task)", "levels": ["close", "reworded"],
        "raw": [kn(sp[sp.variant == f"paraphrase_{i}"]) for i in range(2)],
        "canon": [kn(sc[sc.variant == f"paraphrase_{i}"]) for i in range(2)],
        "ref": kn(bs[bs.episode < 5]),
    })
if canon_panels:
    from scipy.stats import spearmanr

    from vla_stress.env_utils import task_instructions
    from vla_stress.language import Canonicalizer

    cz = Canonicalizer()
    recs = []
    for suite, raw in [("libero_goal", lv[lv.condition == "paraphrase"] if lv is not None else None), ("libero_spatial", sp)]:
        if raw is None:
            continue
        orig = task_instructions(suite)
        for (t, text), g in raw.groupby(["task_id", "instruction"]):
            recs.append({"suite": suite, "task_id": t, "text": text, "similarity": cz.similarity(text, orig[t]), "rate": g.success.mean(), "k": int(g.success.sum()), "n": len(g)})
    sim_df = pd.DataFrame(recs)
    sim_df.to_csv(SUM / "paraphrase_similarity.csv", index=False)
    rho, p_rho = spearmanr(sim_df.similarity, sim_df.rate)
    canon_stats["spearman"] = (rho, p_rho)
    for name, d in [("goal", lc), ("spatial", sc)]:
        if d is None:
            continue
        suite = d.suite.iloc[0]
        orig = task_instructions(suite)
        u = d.drop_duplicates("instruction")
        canon_stats[f"retrieval_{name}"] = (int(sum(a == orig[t] for a, t in zip(u.instruction_used, u.task_id))), len(u))
    canon_stats["absurd_sim"] = cz("sing a song")[1]
    plots.canonicalisation(canon_panels, sim_df, FIG / "language_canonical.pdf")
    headline["canonicalisation"] = {k: v for k, v in canon_stats.items()}

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

# ---------------------------------------------------------------- residual RL
rl_vla = load("rl_eval_vla")
rl_tr_vla = load("rl_eval_train_vla")
rl_res = {Path(f).stem.split("_")[-1]: load(Path(f).stem) for f in sorted(glob.glob(str(R / "rl_eval_residual_s*.csv")))}
rl_tr_res = {Path(f).stem.split("_")[-1]: load(Path(f).stem) for f in sorted(glob.glob(str(R / "rl_eval_train_residual_s*.csv")))}
# residual trained with random offset signs (more diverse starts), keys rand_s0, rand_s1
rl_res_rand = {"rand_" + Path(f).stem.split("_")[-1]: load(Path(f).stem) for f in sorted(glob.glob(str(R / "rl_eval_residual_rand_s*.csv")))}
rl_tr_res_rand = {"rand_" + Path(f).stem.split("_")[-1]: load(Path(f).stem) for f in sorted(glob.glob(str(R / "rl_eval_train_residual_rand_s*.csv")))}
logs = {}
for f in sorted(glob.glob("runs/robot025_task2_s*/log.csv") + glob.glob("runs/robot025_task2_rand_s*/log.csv")):
    try:
        lg = pd.read_csv(f)
    except pd.errors.EmptyDataError:
        continue  # run just started
    if len(lg) >= 40:  # finished runs only
        logs[Path(f).parent.name.replace("robot025_task2_", "")] = lg
rl_rows, groups, rl_steps = [], [], {}
if rl_vla is not None and rl_res:
    def kn_at(d, mag):
        g = d[np.isclose(d.magnitude, mag)]
        return (int(g.success.sum()), len(g))

    def paired(v, r, mag):
        m = v[np.isclose(v.magnitude, mag)].merge(r[np.isclose(r.magnitude, mag)], on="episode", suffixes=("_v", "_r"))
        return m

    if rl_tr_vla is not None:
        groups.append({"label": "seen states\n0.05 rad", "vla": kn_at(rl_tr_vla, 0.05), "res": {sd: kn_at(d, 0.05) for sd, d in {**rl_tr_res, **rl_tr_res_rand}.items()}})
    for mag, lab in [(0.05, "unseen states\n0.05 rad"), (0.0, "unseen states\nno offset"), (0.1, "unseen states\n0.10 rad")]:
        groups.append({"label": lab, "vla": kn_at(rl_vla, mag), "res": {sd: kn_at(d, mag) for sd, d in {**rl_res, **rl_res_rand}.items()}})
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
    macros["LangOrig"] = frac(int(lang_ref.success.sum()), len(lang_ref))
    macros["LangOrigPct"] = pct(int(lang_ref.success.sum()), len(lang_ref))
    macros["LangParaPct"] = pct(int(pp.success.sum()), len(pp))
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
    rseeds = sorted(rl_res_rand)
    if rseeds:
        names_r = {"seen states 0.05 rad": "Seen", "unseen states 0.05 rad": "Unseen", "unseen states no offset": "UnseenClean", "unseen states 0.10 rad": "UnseenStrong"}
        for row in rl_rows:
            ks = [row[sd] for sd in rseeds if sd in row]
            if ks:
                macros[f"RLRand{names_r[row['group']]}Res"] = ", ".join(ks)
        macros["RLRandSeeds"] = str(len(rseeds))
    names = {"seen states 0.05 rad": "Seen", "unseen states 0.05 rad": "Unseen", "unseen states no offset": "UnseenClean", "unseen states 0.10 rad": "UnseenStrong"}
    for row in rl_rows:
        tag = names[row["group"]]
        macros[f"RL{tag}Vla"] = row["vla"]
        macros[f"RL{tag}Res"] = ", ".join(row[sd] for sd in seeds if sd in row)
    tr = [rl_steps[(sd, "train")] for sd in seeds if (sd, "train") in rl_steps]
    ho = [rl_steps[(sd, "heldout")] for sd in seeds if (sd, "heldout") in rl_steps]
    if tr:
        macros["RLStepsSeenVla"] = f"{tr[0][0]:.0f}"
        macros["RLStepsSeenRes"] = f"{min(t[1] for t in tr):.0f}--{max(t[1] for t in tr):.0f}"
        macros["RLPSeen"] = ", ".join(f"{t[2]:.2f}" for t in tr)
    if ho:
        macros["RLStepsUnseenVla"] = f"{ho[0][0]:.0f}"
        macros["RLStepsUnseenRes"] = f"{min(t[1] for t in ho):.0f}--{max(t[1] for t in ho):.0f}"
    if logs:
        first = [lg.success_rate.head(10).mean() for lg in logs.values()]
        last = [lg.success_rate.tail(10).mean() for lg in logs.values()]
        macros["RLTrainFirst"] = ", ".join(f"{v:.2f}" for v in first)
        macros["RLTrainLast"] = ", ".join(f"{v:.2f}" for v in last)
if cm is not None and bs is not None:
    both = bs[bs.episode.isin(cm.episode.unique())]
    macros["CamMaskBoth"] = f"{int(both.success.sum())}/{len(both)}"
    for c, tag in [("mask_agentview", "Agent"), ("mask_wrist", "Wrist")]:
        g = cm[cm.condition == c]
        macros[f"CamMask{tag}"] = f"{int(g.success.sum())}/{len(g)}"
for p in canon_panels:
    tag = "Goal" if "Goal" in p["title"] else "Spatial"
    rk = sum(k for k, _ in p["raw"]); rn = sum(n for _, n in p["raw"])
    ck = sum(k for k, _ in p["canon"]); cn = sum(n for _, n in p["canon"])
    macros[f"Canon{tag}Raw"] = frac(rk, rn)
    macros[f"Canon{tag}"] = frac(ck, cn)
    macros[f"Canon{tag}Ref"] = frac(*p["ref"])
for name, tag in [("retrieval_goal", "Goal"), ("retrieval_spatial", "Spatial")]:
    if name in canon_stats:
        macros[f"Retrieval{tag}"] = "%d/%d" % canon_stats[name]
if "spearman" in canon_stats:
    macros["SimRho"] = f"{canon_stats['spearman'][0]:.2f}"
    macros["SimRhoP"] = f"{canon_stats['spearman'][1]:.2g}"
    macros["AbsurdSim"] = f"{canon_stats['absurd_sim']:.2f}"
    macros["CanonThreshold"] = "0.5"

qd_path, qv_path = R / "qd_search.csv", R / "qd_validation.csv"
if qd_path.exists():
    qd_rows = pd.read_csv(qd_path)
    qv = pd.read_csv(qv_path) if qv_path.exists() else None
    plots.qd_summary(qd_rows, qv, 5, FIG / "qd_search.pdf")
    macros["QDEvals"] = str(len(qd_rows))
    macros["QDEpisodes"] = str(4 * len(qd_rows))
    best = qd_rows.groupby("cell").fitness.max()
    macros["QDCells"] = str(best.size)
    macros["QDFailCells"] = str(int((qd_rows.groupby("cell").n_fail.max() > 0).sum()))
    anchors = qd_rows[qd_rows.source == "anchor"]
    single = anchors[(anchors[["g0", "g1", "g2", "g3"]] == 1).sum(axis=1) == 1]
    corner = anchors[(anchors[["g0", "g1", "g2", "g3"]] == 1).all(axis=1)]
    macros["QDSingleFails"] = f"{int(single.n_fail.sum())}/{4 * len(single)}"
    if len(corner):
        macros["QDCornerFails"] = f"{int(corner.n_fail.iloc[0])}/4"
    macros["QDAnyFail"] = f"{int((qd_rows.n_fail > 0).sum())}/{len(qd_rows)}"
    from scipy.stats import spearmanr

    searched = qd_rows[qd_rows.source != "anchor"]
    for gname, tag in [("g0", "Cam"), ("g1", "Arm"), ("g2", "Light"), ("g3", "Noise")]:
        macros[f"QDRho{tag}"] = f"{spearmanr(searched[gname], searched.fitness)[0]:.2f}"
    if qv is not None and len(qv):
        g = qv.groupby("eval_id").success.agg(lambda x: int((x == 0).sum()))
        nper = qv.groupby("eval_id").size()
        macros["QDValBest"] = f"{int(g.max())}/{int(nper[g.idxmax()])}"
        macros["QDValPooled"] = frac(int(g.sum()), int(nper.sum()))
        macros["QDValPooledPct"] = pct(int(g.sum()), int(nper.sum()))
        macros["QDValElites"] = str(g.size)
        cq = qd_rows[qd_rows.eval_id.isin(g.index)]
        if len(corner) and corner.eval_id.iloc[0] in g.index:
            macros["QDValCorner"] = f"{int(g[corner.eval_id.iloc[0]])}/{int(nper[corner.eval_id.iloc[0]])}"
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
