#!/usr/bin/env bash
# Residual RL stage: plumbing checks, throughput (go/no-go), then training.
set -eu
export HF_HUB_OFFLINE=1
mkdir -p outputs/logs runs
python scripts/zero_residual.py runs/zero/final.pt
python -u -m vla_stress.evaluate --config configs/rl_zero_check.yaml > outputs/logs/rl_zero_check.log 2>&1
python -u -m vla_stress.residual_rl.ppo --task 2 --perturbation robot_init --intensity 0.25 --run-dir outputs/ppo_smoke --smoke > outputs/logs/ppo_smoke.log 2>&1
python -u scripts/rl_throughput.py --n-envs 1 2 4 --steps 200 > results/rl_throughput.csv 2> outputs/logs/rl_throughput.err
echo "RL checks done $(date)"
