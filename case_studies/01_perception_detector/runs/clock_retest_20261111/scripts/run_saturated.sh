#!/usr/bin/env bash
# Same alternating repeats with every core busy (24 background processes), then 12.
set -euo pipefail
REPO=~/github/bench2field; PY=$REPO/.venv/bin/python
PROF=$REPO/case_studies/01_perception_detector/profile_pipeline.py
FR=$REPO/data/rover_frames_720p; OUT="$(dirname "$0")/runs"
one() { uptime | sed 's/.*load average/load average/' > "$OUT/$1.load"; $PY $PROF --frames $FR --out "$OUT/$1.json" ${2:-} > "$OUT/$1.log" 2>&1; }
for n in 12 24; do
  pids=(); for k in $(seq $n); do $PY -c "while True: pass" & pids+=($!); done; sleep 2
  for i in 1 2 3; do one busy${n}_spin_$i; one busy${n}_nospin_$i --no-spin; done
  kill "${pids[@]}"; wait 2>/dev/null || true
done
