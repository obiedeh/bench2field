#!/usr/bin/env bash
# Alternating repeats of the 720p frame profile with spinning on and off,
# first on a quiet machine, then with four busy processes in the background.
set -euo pipefail
REPO=~/github/bench2field; PY=$REPO/.venv/bin/python
PROF=$REPO/case_studies/01_perception_detector/profile_pipeline.py
FR=$REPO/data/rover_frames_720p; OUT="$(dirname "$0")/runs"
one() { # name, extra flags
  uptime | sed 's/.*load average/load average/' > "$OUT/$1.load"
  $PY $PROF --frames $FR --out "$OUT/$1.json" ${2:-} > "$OUT/$1.log" 2>&1
}
for i in 1 2 3 4 5; do one quiet_spin_$i; one quiet_nospin_$i --no-spin; done
for t in 24 1; do $PY "$(dirname "$0")/prep_alone.py" $FR "$OUT/prep_alone_cv$t.json" --cv-threads $t > "$OUT/prep_alone_cv$t.log" 2>&1; done
pids=(); for n in 1 2 3 4; do $PY -c "while True: pass" & pids+=($!); done
sleep 2
for i in 1 2 3; do one busy_spin_$i; one busy_nospin_$i --no-spin; done
$PY "$(dirname "$0")/prep_alone.py" $FR "$OUT/prep_alone_cv24_busy.json" --cv-threads 24 > "$OUT/prep_alone_cv24_busy.log" 2>&1
kill "${pids[@]}"
