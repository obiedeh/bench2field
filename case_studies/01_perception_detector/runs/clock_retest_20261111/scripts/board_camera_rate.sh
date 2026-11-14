#!/usr/bin/env bash
# Runs on the rover's Orin NX: start only the camera driver (no base, no motors, no mapping),
# and read the colour topic's delivered rate for 60 s with clocks shipped, pinned, shipped.
set -o pipefail
source /opt/ros/humble/setup.bash
for w in ~/yahboomcar_ws ~/M3Pro_ws; do [ -f $w/install/setup.bash ] && source $w/install/setup.bash; done
export ROS_DOMAIN_ID=30
OUT=/tmp/spin-repeat/camera_rate; mkdir -p $OUT
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
setsid ros2 launch m3pro_bringup dabai_dcw2.launch.py > $OUT/camera.log 2>&1 & CAM=$!
trap 'kill -INT -- -$CAM 2>/dev/null; sleep 4; kill -KILL -- -$CAM 2>/dev/null; clocks off' EXIT
sleep 20
ros2 topic list > $OUT/topics.txt 2>&1
n=0
for cond in default pinned default; do
  n=$((n+1)); name=$(printf "%02d_%s" $n $cond)
  [ $cond = pinned ] && clocks on > $OUT/$name.clocks || clocks off > $OUT/$name.clocks
  sleep 8
  echo "$(cat /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq | tr '\n' ' ')" > $OUT/$name.freq
  timeout -s INT 62 ros2 topic hz /camera/color/image_raw --window 2000 > $OUT/$name.hz 2>&1
  top -b -n 1 -o %CPU | sed -n 7,12p > $OUT/$name.top
  echo "$name: $(grep 'average rate' $OUT/$name.hz | tail -1) | $(grep -c 'average rate' $OUT/$name.hz) reports"
done
echo "done camera"
