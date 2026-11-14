#!/usr/bin/env bash
# Runs on a board: the same alternating pairs with N background busy processes.
set -uo pipefail
N=$1; PAIRS=${2:-3}
REPO=~/github/bench2field; PY=$REPO/.venv/bin/python
PROF=$REPO/case_studies/01_perception_detector/profile_pipeline.py
OUT=/tmp/spin-repeat/busy${N}_480p; mkdir -p $OUT; cd $REPO
pids=(); for k in $(seq $N); do python3 -c "while True: pass" & pids+=($!); done
trap 'kill "${pids[@]}" 2>/dev/null' EXIT
sleep 2
for i in $(seq $PAIRS); do for mode in spin nospin; do
  flag=""; [ $mode = nospin ] && flag="--no-spin"
  cut -d" " -f1-3 /proc/loadavg > $OUT/${mode}_$i.load
  $PY $PROF --frames data/rover_frames_480p --out $OUT/${mode}_$i.json $flag > $OUT/${mode}_$i.log 2>&1 || echo "FAILED ${mode}_$i"
done; done
echo "done busy$N"
