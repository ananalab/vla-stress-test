#!/usr/bin/env bash
# Remaining runs after residual seed 0 (priority order, each step resumable).
set -u
export HF_HUB_OFFLINE=1
ev() { python -u -m vla_stress.evaluate "$@"; }
train_and_eval() {  # $1 = seed
  local run=runs/robot025_task2_s$1
  python -u -m vla_stress.residual_rl.ppo --task 2 --perturbation robot_init --intensity 0.25 \
    --n-envs 4 --rollout-steps 512 --total-steps 100000 --seed $1 --run-dir $run > outputs/logs/ppo_s$1.log 2>&1 || echo "ppo seed $1 failed"
  if [ -f $run/final.pt ]; then
    ev --config configs/rl_eval.yaml --residual $run/final.pt --out results/rl_eval_residual_s$1.csv > outputs/logs/rl_eval_s$1.log 2>&1
    ev --config configs/rl_eval_train.yaml --residual $run/final.pt --out results/rl_eval_train_residual_s$1.csv > outputs/logs/rl_eval_train_s$1.log 2>&1
  fi
}
echo "start $(date)"
ev --config configs/rl_eval_train.yaml --out results/rl_eval_train_vla.csv > outputs/logs/rl_eval_train_vla.log 2>&1
ev --config configs/rl_eval_train.yaml --residual runs/robot025_task2_s0/final.pt --out results/rl_eval_train_residual_s0.csv > outputs/logs/rl_eval_train_s0.log 2>&1
ev --config configs/rl_eval_replan.yaml > outputs/logs/rl_eval_replan10.log 2>&1;          echo "diag $(date)"
ev --config configs/dose_robot_init_fine.yaml >> outputs/logs/dose_robot_init_fine.log 2>&1
ev --config configs/dose_light_dimming_fine.yaml >> outputs/logs/dose_light_dimming_fine.log 2>&1; echo "fine $(date)"
train_and_eval 1;                                                                           echo "seed1 $(date)"
ev --config configs/baseline_goal.yaml >> outputs/logs/baseline_goal.log 2>&1
ev --config configs/camera_mask.yaml >> outputs/logs/camera_mask.log 2>&1
ev --config configs/dose_camera_orbit_wide.yaml >> outputs/logs/dose_camera_orbit_wide.log 2>&1
ev --config configs/dose_image_noise.yaml >> outputs/logs/dose_image_noise.log 2>&1;      echo "evals $(date)"
python -u scripts/record_rollouts.py --spec media/specs/perturbations.yaml > outputs/logs/video_pert.log 2>&1
python -u scripts/record_rollouts.py --spec media/specs/language.yaml > outputs/logs/video_lang.log 2>&1
python -u scripts/record_rollouts.py --spec media/specs/residual.yaml > outputs/logs/video_res.log 2>&1;  echo "videos $(date)"
train_and_eval 2;                                                                           echo "seed2 $(date)"
echo "all done $(date)"
