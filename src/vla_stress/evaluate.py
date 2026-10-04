"""Resumable evaluation of SmolVLA on LIBERO under perturbations.

    python -m vla_stress.evaluate --config configs/baseline.yaml
    python -m vla_stress.evaluate --config configs/dose_camera.yaml --shard 0/2   # two processes
    python -m vla_stress.evaluate --config configs/language_matrix.yaml --smoke

One CSV row per episode. Rows already in the CSV are skipped, so a killed run
(Colab timeout, laptop lid...) just restarts where it stopped.

Episode e of a task always uses LIBERO init state e and seed e, whatever the
condition. Two conditions therefore differ only by the perturbation, which
makes paired comparisons possible.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import platform
import subprocess
from pathlib import Path

import torch
import yaml

from vla_stress import perturbations as P
from vla_stress.env_utils import CHECKPOINT_REVISION, load_vla, make_env, run_episode, task_instructions

KEY = ["suite", "task_id", "condition", "intensity", "variant", "episode"]
FIELDS = KEY + [
    "perturbation",
    "magnitude",
    "unit",
    "sign",
    "instruction",
    "success",
    "steps",
    "n_policy_calls",
    "duration_s",
    "seed",
    "n_action_steps",
    "resolution",
    "config",
    "git_commit",
    "lerobot_version",
    "checkpoint_revision",
    "device",
    "date",
]


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "src"]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return "mps-" + platform.machine() + "-" + subprocess.getoutput("sysctl -n machdep.cpu.brand_string").replace(" ", "_")
    return "cpu-" + platform.processor()


def expand_jobs(cfg: dict) -> list[dict]:
    """Turn the config into a flat, deterministic list of episodes to run."""
    suite = cfg["suite"]
    instructions = task_instructions(suite)
    tasks = list(range(len(instructions))) if cfg.get("tasks", "all") == "all" else list(cfg["tasks"])
    episodes = range(cfg.get("first_episode", 0), cfg.get("first_episode", 0) + cfg["episodes"])

    def variants(cond, task_id):
        """Yield (variant label, instruction) pairs for one condition and target task."""
        if "instruction_from_task" in cond:  # language matrix: say task i, score task j
            src = cond["instruction_from_task"]
            src = range(len(instructions)) if src == "all" else src
            for i in src:
                yield f"from_task_{i}", instructions[i]
        elif "instruction" in cond:
            yield cond.get("variant", "custom"), cond["instruction"]
        elif "paraphrases" in cond:
            for k, text in enumerate(cond["paraphrases"][task_id]):
                yield f"paraphrase_{k}", text
        else:
            yield "", instructions[task_id]

    jobs = []
    # Episode-major order: if a run is cut short, every task/condition has the same
    # number of episodes so far, instead of some conditions being complete and others empty.
    for ep in episodes:
        for task_id in tasks:
            for cond in cfg["conditions"]:
                for intensity in cond.get("intensities", [0.0]):
                    for variant, instruction in variants(cond, task_id):
                        jobs.append(
                            {
                                "suite": suite,
                                "task_id": task_id,
                                "condition": cond.get("label", cond.get("perturbation", "clean")),
                                "perturbation": cond.get("perturbation", "none"),
                                "params": cond.get("params", {}),
                                "intensity": float(intensity),
                                "variant": variant,
                                "instruction": instruction,
                                "episode": ep,
                            }
                        )
    return jobs


def key_of(row: dict) -> tuple:
    return (row["suite"], int(row["task_id"]), row["condition"], f"{float(row['intensity']):.4f}", row["variant"], int(row["episode"]))


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path) as f:
        return {key_of(r) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="CSV path (default: results/<config name>.csv)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    ap.add_argument("--smoke", action="store_true", help="1 task, 1 episode, 60 steps, print the tokenised instruction")
    ap.add_argument("--videos", action="store_true", help="save an mp4 of episode 0 for every task/condition")
    ap.add_argument("--video-dir", default="outputs/videos")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name = cfg.get("name", Path(args.config).stem)
    shard_i, shard_n = map(int, args.shard.split("/"))
    out = Path(args.out or f"results/{name}.csv")
    if shard_n > 1 and args.out is None:
        out = out.with_suffix(f".shard{shard_i}of{shard_n}.csv")
    if args.smoke:
        out = Path("outputs/smoke") / out.name
    out.parent.mkdir(parents=True, exist_ok=True)

    jobs = expand_jobs(cfg)
    if args.smoke:
        first = jobs[0]
        jobs = [j for j in jobs if j["task_id"] == first["task_id"] and j["episode"] == first["episode"]][:4]
    jobs = jobs[shard_i::shard_n]
    done = done_keys(out)
    todo = [j for j in jobs if key_of(j) not in done]
    print(f"[{name}] {len(jobs)} episodes in shard, {len(jobs) - len(todo)} already in {out}, {len(todo)} to run")
    if not todo:
        return

    vla = load_vla(args.device)
    import lerobot

    meta = {
        "config": name,
        "git_commit": git_commit(),
        "lerobot_version": lerobot.__version__,
        "checkpoint_revision": CHECKPOINT_REVISION[:12],
        "device": device_name(vla.device),
        "n_action_steps": cfg.get("n_action_steps", 50),
        "resolution": cfg.get("resolution", 360),
    }
    envs = {}
    new_file = not out.exists()
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for n, job in enumerate(todo):
            if job["task_id"] not in envs:
                envs[job["task_id"]] = make_env(job["suite"], job["task_id"], resolution=meta["resolution"])
            env = envs[job["task_id"]]
            pert = P.build(job["perturbation"], job["intensity"], **job["params"])
            if args.smoke:
                obs = env.reset(seed=0)[0]
                print("  instruction given :", repr(job["instruction"]))
                print("  tokens decoded    :", repr(vla.decode_tokens(obs, job["instruction"])))
            res = run_episode(
                env,
                vla,
                job["instruction"],
                episode=job["episode"],
                seed=job["episode"],
                perturbation=pert,
                n_action_steps=meta["n_action_steps"],
                record=args.videos and job["episode"] == 0,
                max_episode_steps=60 if args.smoke else None,
            )
            sign = pert.sign(job["episode"]) if hasattr(pert, "sign") else ""
            row = {
                **job,
                **meta,
                **{k: v for k, v in res.items() if k != "frames"},
                "magnitude": round(pert.magnitude, 4),
                "unit": pert.unit,
                "sign": sign,
                "seed": job["episode"],
                "intensity": f"{job['intensity']:.4f}",
                "date": dt.datetime.now().isoformat(timespec="seconds"),
            }
            writer.writerow(row)
            f.flush()
            if res["frames"]:
                from lerobot.utils.io_utils import write_video

                vdir = Path(args.video_dir) / name
                vdir.mkdir(parents=True, exist_ok=True)
                tag = f"{job['suite']}_t{job['task_id']}_{row['condition']}_{row['intensity']}_{job['variant']}_e{job['episode']}"
                write_video(vdir / f"{tag}_{'ok' if res['success'] else 'fail'}.mp4", res["frames"], fps=20)
            print(
                f"  [{n + 1}/{len(todo)}] task {job['task_id']} {row['condition']} i={row['intensity']} "
                f"{job['variant']} ep {job['episode']}: {'OK ' if res['success'] else 'FAIL'} "
                f"{res['steps']} steps {res['duration_s']}s",
                flush=True,
            )
    for env in envs.values():
        env.close()


if __name__ == "__main__":
    main()
