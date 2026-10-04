"""Record rollouts side by side and write a captioned GIF + MP4.

    python scripts/record_rollouts.py --spec media/specs/perturbations.yaml

A spec lists panels; each panel is one episode:

    out: media/perturbations          # writes .gif and .mp4
    columns: 4
    panels:
      - {suite: libero_spatial, task: 2, episode: 0, title: "clean"}
      - {suite: libero_spatial, task: 2, episode: 0, perturbation: camera_orbit, intensity: 1.0, title: "camera +30 deg"}
      - {suite: libero_goal, task: 8, episode: 0, instruction: "turn on the stove", title: "told: turn on the stove"}
      - {..., residual: runs/x/final.pt}

The frames are what the policy sees from the main camera (after the perturbation),
upright. Panels that finish early hold their last frame with the outcome stamped on it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from vla_stress import perturbations as P
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
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.spec))
    vla = load_vla()

    runs = []
    for p in spec["panels"]:
        suite = p["suite"]
        instruction = p.get("instruction") or task_instructions(suite)[p["task"]]
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
        runs.append((frames, p.get("title", ""), p.get("subtitle", instruction), ok))
        print(f"{p.get('title')}: {'success' if ok else 'failure'} in {res['steps']} steps")

    T = max(len(f) for f, *_ in runs)
    hold = args.fps * 2  # show the final state for two seconds
    cols = spec.get("columns", len(runs))
    rows = int(np.ceil(len(runs) / cols))
    video = []
    for t in list(range(0, T, args.stride)) + [T - 1] * hold:
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

    out = Path(spec["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out.with_suffix(".mp4"), video, fps=args.fps, quality=8, macro_block_size=2)
    small = [np.asarray(Image.fromarray(f).resize((f.shape[1] * 2 // 3, f.shape[0] * 2 // 3), Image.LANCZOS)) for f in video[::2]]
    imageio.mimsave(out.with_suffix(".gif"), small, duration=1000 * 2 / args.fps, loop=0)
    Image.fromarray(video[-1]).save(out.with_name(out.name + "_final.png"))
    print("wrote", out.with_suffix(".gif"), out.with_suffix(".mp4"))


if __name__ == "__main__":
    main()
