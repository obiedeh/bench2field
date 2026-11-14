#!/usr/bin/env bash
# Runs on a board: alternating repeats of the frame profile, spinning on and off.
# Uses the board's existing checkout and environment unchanged; writes to /tmp.
set -uo pipefail
FRAMES_SET=$1; PAIRS=${2:-5}
REPO=~/github/bench2field; PY=$REPO/.venv/bin/python
PROF=$REPO/case_studies/01_perception_detector/profile_pipeline.py
OUT=/tmp/spin-repeat/$FRAMES_SET; mkdir -p $OUT
cd $REPO
for i in $(seq $PAIRS); do
  for mode in spin nospin; do
    flag=""; [ $mode = nospin ] && flag="--no-spin"
    cut -d" " -f1-3 /proc/loadavg > $OUT/${mode}_$i.load
    $PY $PROF --frames data/$FRAMES_SET --out $OUT/${mode}_$i.json $flag > $OUT/${mode}_$i.log 2>&1 || echo "FAILED ${mode}_$i: $(tail -1 $OUT/${mode}_$i.log)"
  done
done
echo "done $FRAMES_SET"
