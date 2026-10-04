#!/usr/bin/env bash
# Run experiment configs one after the other (an 8 GB laptop fits one VLA process).
# Every config is resumable, so killing and relaunching this script is safe.
#   bash scripts/run_queue.sh configs/baseline_spatial.yaml configs/dose_camera_orbit.yaml ...
set -u
mkdir -p outputs/logs
export HF_HUB_OFFLINE=1
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  echo "=== $(date '+%F %T') start $name"
  python -u -m vla_stress.evaluate --config "$cfg" >> "outputs/logs/$name.log" 2>&1
  echo "=== $(date '+%F %T') end $name (exit $?)"
done
