#!/usr/bin/env bash
# Runs on a board: the frame profile (spinning off) with clocks as shipped and
# with clocks pinned at maximum through the board's monitoring service.
# Order default, pinned, pinned, default, repeated. Restores the shipped state on exit.
set -uo pipefail
ROUNDS=${1:-2}
REPO=~/github/bench2field; PY=$REPO/.venv/bin/python
PROF=$REPO/case_studies/01_perception_detector/profile_pipeline.py
OUT=/tmp/spin-repeat/pinned_480p; mkdir -p $OUT; cd $REPO
clocks() {  # on|off
python3 - "$1" <<'P'
import sys, time
from jtop import jtop
want = sys.argv[1] == "on"
with jtop() as jet:
    jet.jetson_clocks = want
    for _ in range(40):
        time.sleep(0.5)
        if bool(jet.jetson_clocks) == want and jet.jetson_clocks.get_status() in ("running", "inactive"):
            break
    print("jetson_clocks", jet.jetson_clocks.get_status())
P
}
state() {
  for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "$(basename $p) $(cat $p/scaling_governor) min=$(cat $p/scaling_min_freq) cur=$(cat $p/scaling_cur_freq)"; done
  for d in /sys/class/devfreq/*gpu* /sys/class/devfreq/bwmgr; do [ -e $d ] && echo "$(basename $d) $(cat $d/governor) min=$(cat $d/min_freq) cur=$(cat $d/cur_freq)"; done
}
sample() {  # one line every 0.5 s: cpu policy clocks, then gpu and memory bus clocks
  while :; do echo "$(cat /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq | tr '\n' ' ')| $(cat /sys/class/devfreq/*gpu*/cur_freq /sys/class/devfreq/bwmgr/cur_freq 2>/dev/null | tr '\n' ' ')"; sleep 0.5; done
}
trap 'kill $SP 2>/dev/null; clocks off' EXIT
n=0
for r in $(seq $ROUNDS); do for cond in default pinned pinned default; do
  n=$((n+1)); name=$(printf "%02d_%s" $n $cond)
  [ $cond = pinned ] && clocks on > $OUT/$name.clocks || clocks off > $OUT/$name.clocks
  sleep 3; state > $OUT/$name.state
  cut -d" " -f1-3 /proc/loadavg > $OUT/$name.load
  sample > $OUT/$name.freq & SP=$!
  $PY $PROF --frames data/rover_frames_480p --out $OUT/$name.json --no-spin > $OUT/$name.log 2>&1 || echo "FAILED $name"
  kill $SP 2>/dev/null
done; done
clocks off; state > $OUT/final.state
echo "done pinned"
