#!/bin/bash
# Host-side wrapper on the Jetson AGX Thor: samples tegrastats for the whole run and
# executes the case study 04 quickstart runner inside the owner-built container.
set -u
ROOT=$HOME/isaac-ros-20261030
LABEL=${LABEL:-thor-agx-jetpack71}
RATE=${RATE:-1}; LOOP=${LOOP:-1}; SECONDS_MEASURE=${SECONDS_MEASURE:-60}
OUT=$ROOT/logs/$LABEL
mkdir -p "$OUT"
echo "== host start $(date -Is) label=$LABEL rate=$RATE loop=$LOOP window=${SECONDS_MEASURE}s"
nvpmodel -q 2>/dev/null | head -2 > "$OUT/nvpmodel.txt"
tegrastats --interval 1000 --logfile "$OUT/tegrastats_host.log" &
TS=$!
docker run --rm --runtime nvidia --network host --ipc host \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ROS_DOMAIN_ID=77 -e ROS_LOCALHOST_ONLY=1 \
  -e ASSETS=/work/assets/extracted -e OUT=/work/logs/$LABEL -e LABEL=$LABEL \
  -e RATE=$RATE -e LOOP=$LOOP -e SECONDS_MEASURE=$SECONDS_MEASURE \
  -v "$ROOT":/work isaac-ros-4.5:thor bash /work/run_quickstart.sh
RC=$?
kill $TS 2>/dev/null; wait $TS 2>/dev/null
echo "== container exit $RC $(date -Is)"
echo "tegrastats lines: $(wc -l < "$OUT/tegrastats_host.log")"
ls "$OUT"
