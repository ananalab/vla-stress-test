#!/usr/bin/env bash
# Every experiment of the report, in the order they were run, then figures, tables, videos and PDFs.
# Each evaluation is resumable: re-running this script skips the episodes already in results/.
# About 30 hours on an Apple M1 laptop; one process at a time (one VLA fits in 8 GB).
set -eu
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
ev() { python -u -m vla_stress.evaluate --config "$@"; }

# 1. Baselines and listening test
ev configs/baseline_spatial.yaml
ev configs/baseline_goal.yaml
ev configs/language_matrix.yaml
ev configs/language_variants.yaml

# 2. Graded perturbations and camera masking (LIBERO-Spatial)
for c in dose_camera_orbit dose_camera_orbit_wide dose_robot_init dose_robot_init_fine \
         dose_light_dimming dose_light_dimming_fine dose_image_noise camera_mask; do
  ev configs/$c.yaml
done

# 3. Residual RL: plumbing check, three seeds, evaluation on seen and held-out initial states
python scripts/zero_residual.py runs/zero/final.pt
ev configs/rl_zero_check.yaml
ev configs/rl_eval.yaml --out results/rl_eval_vla.csv
ev configs/rl_eval_train.yaml --out results/rl_eval_train_vla.csv
ev configs/rl_eval_replan.yaml
for s in 0 1 2; do
  run=runs/robot025_task2_s$s
  [ -f $run/final.pt ] || python -u -m vla_stress.residual_rl.ppo --task 2 --perturbation robot_init \
    --intensity 0.25 --n-envs 4 --rollout-steps 512 --total-steps 100000 --seed $s --run-dir $run
  ev configs/rl_eval.yaml --residual $run/final.pt --out results/rl_eval_residual_s$s.csv
  ev configs/rl_eval_train.yaml --residual $run/final.pt --out results/rl_eval_train_residual_s$s.csv
done

# 4. Videos, figures, tables, reports
for v in perturbations language residual; do python scripts/record_rollouts.py --spec configs/videos/$v.yaml; done
python scripts/perturbation_grid.py
python scripts/make_figures.py
python scripts/make_tables.py
python scripts/make_keyframes.py
(cd report && latexmk -pdf -quiet paper.tex && latexmk -pdf -quiet paper_short.tex)
