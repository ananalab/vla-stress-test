"""Search for failures caused by combining individually harmless perturbations (MAP-Elites).

    python -m vla_stress.qd_search search --budget 120
    python -m vla_stress.qd_search validate --n-elites 6

Search space. A genome g in [0, 1]^4 sets four perturbations at once:
camera orbit g0 * 7.5 deg, arm offset g1 * 0.03 rad/joint, light removed g2 * 0.675, pixel noise
g3 * 30. Each maximum is a level at which the perturbation *alone* costs at most ~15% of the
baseline success (dose-response results).

Episodes. Fitness is measured on four episodes (four different tasks) that succeed without
perturbation *and* under each of the four perturbations alone at its maximum. A failure on them can
therefore only come from the combination. Because simulation and policy are deterministic, fitness
is a deterministic function of the genome. Validation re-runs the best genomes on the eleven other
episodes that satisfy the same condition.

Archive. 5 x 5 grid over (geometric intensity, photometric intensity) = (mean(g0, g1), mean(g2, g3)).
Fitness (higher = more harmful) = mean over episodes of 1 for a failure, else steps / max_steps.

Every evaluation is appended to a CSV, so the search resumes after an interruption.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from vla_stress import perturbations as P
from vla_stress.env_utils import load_vla, make_env, run_episode, task_instructions

SUITE = "libero_spatial"
MAX = {"camera_orbit": 7.5, "robot_init": 0.03, "light_dimming": 0.675, "image_noise": 30.0}
AXES = list(MAX)
SEARCH_EPISODES = [(0, 0), (2, 0), (6, 2), (7, 0)]
VALIDATION_EPISODES = [(0, 1), (0, 3), (0, 4), (2, 1), (2, 2), (2, 3), (2, 4), (4, 0), (6, 4), (7, 4), (9, 0)]
GRID = 5
HORIZON = 280


def perturbation(g: np.ndarray) -> P.Combined:
    parts = []
    for gi, name in zip(g, AXES):
        if gi > 0:
            parts.append(P.build(name, float(gi), max_magnitude=MAX[name]))
    return P.Combined(parts)


def descriptor(g: np.ndarray) -> tuple[float, float]:
    return float((g[0] + g[1]) / 2), float((g[2] + g[3]) / 2)


def cell(d: tuple[float, float]) -> tuple[int, int]:
    return tuple(min(int(x * GRID), GRID - 1) for x in d)


def evaluate(vla, g, episodes):
    instr = task_instructions(SUITE)
    out = []
    for task, ep in episodes:
        env = make_env(SUITE, task)
        r = run_episode(env, vla, instr[task], ep, ep, perturbation(g))
        env.close()
        out.append({"task": task, "episode": ep, "success": r["success"], "steps": r["steps"]})
    # A failure counts 1; a success counts its length relative to the time limit, so slower
    # successes (closer to failing) still guide the search.
    fitness = float(np.mean([1.0 if not o["success"] else o["steps"] / HORIZON for o in out]))
    return fitness, out


FIELDS = ["eval_id", "source", "parent", "g0", "g1", "g2", "g3", "camera_deg", "arm_rad", "light_removed", "noise_std",
          "desc_geo", "desc_photo", "cell", "fitness", "n_fail", "episodes", "date"]


def row(eval_id, source, parent, g, fitness, out):
    d = descriptor(g)
    return {
        "eval_id": eval_id, "source": source, "parent": parent,
        **{f"g{i}": round(float(x), 4) for i, x in enumerate(g)},
        "camera_deg": round(g[0] * MAX["camera_orbit"], 3), "arm_rad": round(g[1] * MAX["robot_init"], 4),
        "light_removed": round(g[2] * MAX["light_dimming"], 4), "noise_std": round(g[3] * MAX["image_noise"], 2),
        "desc_geo": round(d[0], 4), "desc_photo": round(d[1], 4), "cell": "%d,%d" % cell(d),
        "fitness": round(fitness, 4), "n_fail": sum(1 - o["success"] for o in out),
        "episodes": json.dumps(out), "date": dt.datetime.now().isoformat(timespec="seconds"),
    }


def load_archive(path: Path):
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    archive = {}
    for r in rows:
        c = r["cell"]
        if c not in archive or float(r["fitness"]) > float(archive[c]["fitness"]):
            archive[c] = r
    return rows, archive


def search(args):
    out = Path(args.out)
    rows, archive = load_archive(out)
    vla = load_vla()
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        # The four single-axis maxima and the full corner are evaluated first, as references.
        anchors = [np.eye(4)[i] for i in range(4)] + [np.ones(4)]
        for n in range(len(rows), args.budget):
            rng = np.random.default_rng(args.seed * 100_000 + n)  # same genome sequence after a resume
            if n < len(anchors):
                g, source, parent = anchors[n], "anchor", ""
            elif n < len(anchors) + args.n_init or not archive:
                g, source, parent = rng.random(4), "random", ""
            else:
                parent_row = archive[list(archive)[rng.integers(len(archive))]]
                pg = np.array([float(parent_row[f"g{i}"]) for i in range(4)])
                g, source, parent = np.clip(pg + rng.normal(0, args.sigma, 4), 0, 1), "mutation", parent_row["eval_id"]
            fitness, res = evaluate(vla, g, SEARCH_EPISODES)
            r = row(n, source, parent, g, fitness, res)
            w.writerow(r)
            f.flush()
            if r["cell"] not in archive or fitness > float(archive[r["cell"]]["fitness"]):
                archive[r["cell"]] = r
            print(f"[{n + 1}/{args.budget}] {source:8s} cam {r['camera_deg']:5.2f} arm {r['arm_rad']:.3f} "
                  f"light {r['light_removed']:.2f} noise {r['noise_std']:5.1f} -> fitness {fitness:.2f} fails {r['n_fail']}/4 "
                  f"| cells {len(archive)}", flush=True)


def validate(args):
    """Re-run the most harmful elites (one per cell) and the full corner on the held-out episodes."""
    rows, archive = load_archive(Path(args.archive))
    elites = sorted(archive.values(), key=lambda r: -float(r["fitness"]))
    picked = [r for r in elites if int(r["n_fail"]) > 0][: args.n_elites]
    corner = [r for r in rows if r["source"] == "anchor" and all(float(r[f"g{i}"]) == 1.0 for i in range(4))]
    picked += [r for r in corner if r["eval_id"] not in {p["eval_id"] for p in picked}]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["eval_id"], int(r["task"]), int(r["episode"])) for r in csv.DictReader(open(out))}
    vla = load_vla()
    instr = task_instructions(SUITE)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eval_id", "camera_deg", "arm_rad", "light_removed", "noise_std", "task", "episode", "success", "steps"])
        if new:
            w.writeheader()
        for r in picked:
            g = np.array([float(r[f"g{i}"]) for i in range(4)])
            for task, ep in VALIDATION_EPISODES:
                if (r["eval_id"], task, ep) in done:
                    continue
                env = make_env(SUITE, task)
                res = run_episode(env, vla, instr[task], ep, ep, perturbation(g))
                env.close()
                w.writerow({"eval_id": r["eval_id"], "camera_deg": r["camera_deg"], "arm_rad": r["arm_rad"],
                            "light_removed": r["light_removed"], "noise_std": r["noise_std"],
                            "task": task, "episode": ep, "success": res["success"], "steps": res["steps"]})
                f.flush()
                print(f"elite {r['eval_id']} task {task} ep {ep}: {'OK' if res['success'] else 'FAIL'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--budget", type=int, default=120)
    s.add_argument("--n-init", type=int, default=15)
    s.add_argument("--sigma", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="results/qd_search.csv")
    v = sub.add_parser("validate")
    v.add_argument("--archive", default="results/qd_search.csv")
    v.add_argument("--n-elites", type=int, default=6)
    v.add_argument("--out", default="results/qd_validation.csv")
    args = ap.parse_args()
    search(args) if args.cmd == "search" else validate(args)


if __name__ == "__main__":
    main()
