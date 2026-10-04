#!/usr/bin/env bash
# Same tasks with the official lerobot-eval, to compare with results/baseline_spatial.csv.
# (Rename map needed: the checkpoint calls its cameras camera1/camera2.)
set -eu
lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero --env.task=libero_spatial --env.task_ids="[1,3]" \
  --eval.batch_size=1 --eval.n_episodes=10 \
  --policy.device=mps --seed=0 \
  --output_dir=outputs/lerobot_eval_check \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
cp outputs/lerobot_eval_check/eval_info.json results/lerobot_eval_check.json
