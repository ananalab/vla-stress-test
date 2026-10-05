"""PPO for the residual corrector, in one file (structure follows CleanRL's ppo_continuous_action).

    python -m vla_stress.residual_rl.ppo --task 0 --perturbation camera_orbit --intensity 0.5 \
        --total-steps 100000 --run-dir runs/cam05_s0 --seed 0

Reward is sparse: 1 when LIBERO reports success, 0 otherwise. Episodes end on
success or after the suite's step limit. The progress t/T is part of the
observation, so treating the time limit as terminal is consistent.

The run directory holds `last.pt` (overwritten after every update, used to resume
after a crash or a Colab disconnect) and `log.csv` (one line per update).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vla_stress.env_utils import load_vla
from vla_stress.residual_rl.env import ResidualLibero
from vla_stress.residual_rl.policy import ACT_DIM, OBS_DIM, ActorCritic


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--perturbation", default="none")
    ap.add_argument("--intensity", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.2, help="scale of the correction on the final action")
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--n-action-steps", type=int, default=50)
    ap.add_argument("--random-signs", action="store_true", help="draw new joint-offset signs at every reset (more diverse training starts)")
    ap.add_argument("--train-init-states", type=int, default=40, help="init states 0..N-1 for training; the rest are held out")
    ap.add_argument("--total-steps", type=int, default=100_000)
    ap.add_argument("--rollout-steps", type=int, default=1024, help="steps per env between updates")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--init-log-std", type=float, default=-1.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default=None, help="device for the VLA; the small MLP stays on CPU")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main():
    args = parse()
    if args.smoke:
        args.total_steps, args.rollout_steps, args.minibatch, args.epochs = 128, 64, 32, 2
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = ActorCritic(args.hidden, args.init_log_std)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    step, update, ep_log = 0, 0, []
    ckpt_path = run / "last.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, update = ck["step"], ck["update"]
        print(f"resumed from {ckpt_path} at step {step}")

    vla = load_vla(args.device)
    env = ResidualLibero(
        vla,
        args.suite,
        args.task,
        args.perturbation,
        args.intensity,
        n_envs=args.n_envs,
        alpha=args.alpha,
        n_action_steps=args.n_action_steps,
        init_states=range(args.train_init_states),
        seed=args.seed * 1000 + update,  # different episodes after a resume
        perturbation_params={"random_signs": True} if args.random_signs else None,
    )
    T, N = args.rollout_steps, args.n_envs
    obs_buf = torch.zeros(T, N, OBS_DIM)
    act_buf = torch.zeros(T, N, ACT_DIM)
    logp_buf = torch.zeros(T, N)
    rew_buf = torch.zeros(T, N)
    done_buf = torch.zeros(T, N)
    val_buf = torch.zeros(T, N)

    log_file = open(run / "log.csv", "a", newline="")
    log = csv.writer(log_file)
    if log_file.tell() == 0:
        log.writerow(["update", "step", "episodes", "success_rate", "mean_len", "pg_loss", "v_loss", "approx_kl", "clipfrac", "std", "mean_abs_delta", "sps"])

    obs = torch.as_tensor(env.reset())
    while step < args.total_steps:
        t0 = time.time()
        episodes = []
        # ---- rollout ----
        for t in range(T):
            with torch.no_grad():
                d = model.dist(obs)
                a = d.sample()
                logp_buf[t] = d.log_prob(a).sum(-1)
                val_buf[t] = model.value(obs)
            obs_buf[t], act_buf[t] = obs, a
            next_obs, r, done, infos = env.step(a.clamp(-1, 1).numpy())
            rew_buf[t] = torch.as_tensor(r)
            done_buf[t] = torch.as_tensor(done, dtype=torch.float32)
            episodes += [inf for inf in infos if inf.get("episode_end")]
            obs = torch.as_tensor(next_obs)
        step += T * N

        # ---- GAE ----
        with torch.no_grad():
            next_val = model.value(obs)
            adv = torch.zeros(T, N)
            last = torch.zeros(N)
            for t in reversed(range(T)):
                nonterm = 1.0 - done_buf[t]
                nv = next_val if t == T - 1 else val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nv * nonterm - val_buf[t]
                last = delta + args.gamma * args.gae_lambda * nonterm * last
                adv[t] = last
            ret = adv + val_buf

        # ---- PPO update ----
        b_obs, b_act, b_logp = obs_buf.reshape(-1, OBS_DIM), act_buf.reshape(-1, ACT_DIM), logp_buf.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
        idx = np.arange(T * N)
        clipfracs, kls = [], []
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), args.minibatch):
                mb = idx[s : s + args.minibatch]
                d = model.dist(b_obs[mb])
                logp = d.log_prob(b_act[mb]).sum(-1)
                ratio = (logp - b_logp[mb]).exp()
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                v_loss = 0.5 * ((model.value(b_obs[mb]) - b_ret[mb]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kls.append(((ratio - 1) - (logp - b_logp[mb])).mean().item())
                    clipfracs.append(((ratio - 1).abs() > args.clip).float().mean().item())
        update += 1
        # Update the input normaliser only between PPO updates, so that the log-probs stored
        # during the rollout and the ones recomputed in the update use the same statistics.
        model.norm.update(b_obs)

        with torch.no_grad():
            mean_delta = model.dist(b_obs).mean.clamp(-1, 1).abs().mean().item()
        succ = np.mean([e["success"] for e in episodes]) if episodes else float("nan")
        mlen = np.mean([e["length"] for e in episodes]) if episodes else float("nan")
        sps = T * N / (time.time() - t0)
        log.writerow([update, step, len(episodes), succ, mlen, pg_loss.item(), v_loss.item(), np.mean(kls), np.mean(clipfracs), model.log_std.exp().mean().item(), mean_delta, round(sps, 1)])
        log_file.flush()
        print(f"update {update} step {step} episodes {len(episodes)} success {succ:.2f} |delta| {mean_delta:.3f} sps {sps:.1f}", flush=True)
        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "update": update, "args": vars(args)}
        torch.save(ckpt, ckpt_path)
        if update % 10 == 0:
            torch.save(ckpt, run / f"step{step}.pt")
    torch.save({"model": model.state_dict(), "step": step, "update": update, "args": vars(args)}, run / "final.pt")
    env.close()


if __name__ == "__main__":
    main()
