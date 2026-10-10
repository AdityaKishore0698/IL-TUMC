#!/usr/bin/env bash
# Closed-loop upper-body evaluation of an exported CL-type policy in MuJoCo (Mac or cir, CPU):
# every locomotion command x every hand, 8 s each, joint traces logged, then scored with the DTW
# measure (upper_body_metric.py). Writes <out>/<name>_*.npz and <out>/<name>.csv.
#
# Usage (from ALMI_trans/): bash ../scripts/eval_upper_body.sh <policy.pt> <name> <out dir> [extra play_trans args]
set -euo pipefail
POLICY=$1; NAME=$2; OUT=$3; shift 3
PY=${PY:-python}
REFS=${REFS:-output/arm_refs.npz}
mkdir -p "$OUT"
for loco in "go forward slowly" "go backward moderately" "go right fast" "go left fast"; do
  for hand in left right both; do
    text="Robot $loco and wave $hand."
    slug=$(echo "$loco $hand" | tr ' ' '_')
    $PY deploy/deploy_mujoco/play_trans.py --policy "$POLICY" --text "$text" --duration 8 --no_viewer \
      --log "$OUT/${NAME}_$slug.npz" "$@" > /dev/null 2>&1
  done
done
$PY upper_body_metric.py score --refs "$REFS" --csv "$OUT/$NAME.csv" "$OUT/${NAME}_"*.npz | tail -1
