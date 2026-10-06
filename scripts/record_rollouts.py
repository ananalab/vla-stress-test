"""Record rollouts side by side and write a captioned GIF + MP4.

    python scripts/record_rollouts.py --spec configs/videos/perturbations.yaml

A spec lists panels; each panel is one episode:

    out: media/perturbations          # writes .gif and .mp4
    columns: 4
    panels:
      - {suite: libero_spatial, task: 2, episode: 0, title: "clean"}
      - {suite: libero_spatial, task: 2, episode: 0, perturbation: camera_orbit, intensity: 1.0, title: "camera +30 deg"}
      - {suite: libero_goal, task: 8, episode: 0, instruction: "turn on the stove", title: "told: turn on the stove"}
      - {..., residual: runs/x/final.pt}
      - {..., canonicalize: true}                  # instruction mapped to the closest training one
      - {..., qd_genome: [0.24, 0.66, 0.76, 0]}    # a combination from vla_stress.qd_search

The frames are what the policy sees from the main camera (after the perturbation),
upright. Panels that finish early hold their last frame with the outcome stamped on it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pickle
import numpy as np
import yaml
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from vla_stress import perturbations as P
from lerobot.utils.io_utils import write_video

from vla_stress.env_utils import load_vla, make_env, run_episode, suite_goals, task_instructions

PANEL = 288
BAR = 46
GREEN, RED, INK, PAPER = (27, 140, 90), (200, 55, 50), (25, 25, 25), (250, 250, 248)


def font(size, bold=False):
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


F_TITLE, F_SUB, F_TAG = font(14, True), font(11), font(13, True)


def wrap(text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if f.getlength(test) <= width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    return lines + [cur] if cur else lines


def panel_frame(img, title, subtitle, t, outcome=None):
    """One captioned panel. outcome: None while running, True/False once the episode is over."""
    canvas = Image.new("RGB", (PANEL, PANEL + BAR), PAPER)
    canvas.paste(Image.fromarray(img).resize((PANEL, PANEL), Image.BILINEAR), (0, BAR))
    d = ImageDraw.Draw(canvas)
    d.text((8, 5), title, font=F_TITLE, fill=INK)
    for k, line in enumerate(wrap(subtitle, F_SUB, PANEL - 16)[:2]):
        d.text((8, 22 + 12 * k), line, font=F_SUB, fill=(90, 90, 86))
    d.text((PANEL - 8, BAR + 6), f"t={t}", font=F_SUB, fill=(255, 255, 255), anchor="ra")
    if outcome is not None:
        tag, col = ("success", GREEN) if outcome else ("failure", RED)
        w = F_TAG.getlength(tag) + 16
        d.rounded_rectangle((8, BAR + 8, 8 + w, BAR + 30), radius=5, fill=col)
        d.text((16, BAR + 11), tag, font=F_TAG, fill=(255, 255, 255))
    return np.asarray(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--stride", type=int, default=2, help="keep one frame out of `stride`")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--from-cache", action="store_true", help="re-compose from the saved frames, no simulation")
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.spec))
    out = Path(spec["out"])
    cache = Path("outputs/video_cache") / (out.name + ".pkl")
    if args.from_cache:
        runs = pickle.load(open(cache, "rb"))
    else:
        runs = simulate(spec)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(runs, open(cache, "wb"))
    compose(runs, spec, out, args.stride, args.fps)


def simulate(spec):
    vla = load_vla()
    canon = None
    runs = []
    for p in spec["panels"]:
        suite = p["suite"]
        instruction = p.get("instruction") or task_instructions(suite)[p["task"]]
        subtitle = p.get("subtitle", instruction)
        if p.get("canonicalize"):
            if canon is None:
                from vla_stress.language import Canonicalizer

                canon = Canonicalizer()
            instruction = canon(instruction)[0]
            subtitle = p.get("subtitle", f"model receives: {instruction}")
        if p.get("qd_genome") is not None:
            from vla_stress.qd_search import perturbation as qd_perturbation

            pert = qd_perturbation(np.array(p["qd_genome"], dtype=float))
        else:
            pert = P.build(p.get("perturbation", "none"), p.get("intensity", 0.0), **p.get("params", {}))
        residual = None
        if p.get("residual"):
            from vla_stress.residual_rl.policy import Residual

            residual = Residual(p["residual"])
        env = make_env(suite, p["task"])
        goals = suite_goals(suite) if p.get("success_goal") is not None else None
        res = run_episode(env, vla, instruction, p["episode"], p.get("seed", p["episode"]), pert, record=True, residual=residual, goals=goals, stop_goal=p.get("success_goal"))
        env.close()
        ok = bool(res["success"])
        if p.get("success_goal") is not None:  # judge by another task's goal (language demo)
            ok = str(p["success_goal"]) in res["goals_achieved"].split(";")
        frames = [f[:, : f.shape[1] // 2] for f in res["frames"]]  # main camera only
        runs.append((frames, p.get("title", ""), subtitle, ok))
        print(f"{p.get('title')}: {'success' if ok else 'failure'} in {res['steps']} steps", flush=True)
    return runs


def compose(runs, spec, out, stride, fps):
    T = max(len(f) for f, *_ in runs)
    hold = fps * 2  # show the final state for two seconds
    cols = spec.get("columns", len(runs))
    rows = int(np.ceil(len(runs) / cols))
    video = []
    for t in list(range(0, T, stride)) + [T - 1] * hold:
        tiles = []
        for frames, title, sub, ok in runs:
            k = min(t, len(frames) - 1)
            tiles.append(panel_frame(frames[k], title, sub, k, ok if t >= len(frames) - 1 else None))
        blank = np.full_like(tiles[0], 255)
        tiles += [blank] * (rows * cols - len(tiles))
        gap = np.full((tiles[0].shape[0], 6, 3), 255, np.uint8)
        grid_rows = []
        for r in range(rows):
            row = []
            for c in range(cols):
                row += [tiles[r * cols + c], gap]
            grid_rows.append(np.concatenate(row[:-1], 1))
        hgap = np.full((6, grid_rows[0].shape[1], 3), 255, np.uint8)
        video.append(np.concatenate(sum([[g, hgap] for g in grid_rows], [])[:-1], 0))

    out.parent.mkdir(parents=True, exist_ok=True)
    write_video(out.with_suffix(".mp4"), video, fps=fps)
    # GIF for the README: smaller, every other frame, shared palette per frame.
    small = [Image.fromarray(f).resize((f.shape[1] * 2 // 3, f.shape[0] * 2 // 3), Image.LANCZOS) for f in video[::2]]
    small = [im.quantize(colors=192, method=Image.Quantize.MEDIANCUT) for im in small]
    small[0].save(out.with_suffix(".gif"), save_all=True, append_images=small[1:], duration=int(2000 / fps), loop=0, optimize=True)
    print("wrote", out.with_suffix(".gif"), out.with_suffix(".mp4"))


if __name__ == "__main__":
    main()
