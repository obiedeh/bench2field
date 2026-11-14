#!/usr/bin/env bash
# Runs on a board: power with clocks as shipped and pinned, order shipped, pinned, pinned, shipped.
set -uo pipefail
REPO=~/github/bench2field; PY=$REPO/.venv/bin/python
OUT=/tmp/spin-repeat/power; mkdir -p $OUT; cd $REPO
clocks() {
python3 - "$1" <<'P'
import sys, time
from jtop import jtop
want = sys.argv[1] == "on"
with jtop() as jet:
    jet.jetson_clocks = want
    for _ in range(40):
        time.sleep(0.5)
        if bool(jet.jetson_clocks) == want and jet.jetson_clocks.get_status() in ("running", "inactive"): break
    print("jetson_clocks", jet.jetson_clocks.get_status())
P
}
trap 'kill $TP 2>/dev/null; pkill -u $USER tegrastats 2>/dev/null; clocks off' EXIT
n=0
for cond in default pinned pinned default; do
  n=$((n+1)); name=$(printf "%02d_%s" $n $cond)
  [ $cond = pinned ] && clocks on > $OUT/$name.clocks || clocks off > $OUT/$name.clocks
  sleep 20
  (tegrastats --interval 1000 | while IFS= read -r l; do echo "$(date +%s.%N) $l"; done) > $OUT/$name.tegra & TP=$!
  $PY /tmp/power_driver.py $OUT/$name.json 45 60 15,30 > $OUT/$name.log 2>&1 || echo "FAILED $name"
  kill $TP 2>/dev/null; pkill -u $USER tegrastats 2>/dev/null
done
clocks off
echo "done power"
