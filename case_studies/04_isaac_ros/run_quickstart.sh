#!/usr/bin/env bash
# Runs the Isaac ROS YOLOv8 quickstart inside the container built from
# docker/Dockerfile and measures it. Expects:
#   ASSETS  directory with isaac_ros_yolov8/quickstart.bag, quickstart_interface_specs.json and models/yolov8n.onnx
#   OUT     directory for the engine build log, launch log and the JSON record
#   LABEL   board label for the record (e.g. x86_64-rtx5090-shared)
#   RATE    bag playback rate multiplier (default 1)
#   LOOP    1 loops the 18-frame bag for the whole window (default), 0 plays it once
#   SECONDS_MEASURE  measurement window (default 60)
set -eo pipefail
source /opt/ros/jazzy/setup.bash
ASSETS=${ASSETS:?}; OUT=${OUT:?}; LABEL=${LABEL:-unlabelled}; RATE=${RATE:-1}; SECONDS_MEASURE=${SECONDS_MEASURE:-60}; LOOP=${LOOP:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT" "$ASSETS/models"
ONNX=$ASSETS/models/yolov8n.onnx
PLAN=$ASSETS/models/yolov8n.plan
# The engine is built by the isaac_ros_tensor_rt node itself from the ONNX
# file, so it matches the TensorRT library the node links (the container's
# trtexec comes from a newer TensorRT release and would write an engine the
# node cannot load). The build happens once; later runs load the file.
BUILD_NOTE="engine built by isaac_ros_tensor_rt from $ONNX"
[ -f "$PLAN" ] && BUILD_NOTE="engine loaded from $PLAN (built by an earlier run)"
echo "== launch start $(date -Is); $BUILD_NOTE"
T_LAUNCH=$(date +%s)
ros2 launch isaac_ros_examples isaac_ros_examples.launch.py launch_fragments:=yolov8 \
    interface_specs_file:="$ASSETS/isaac_ros_yolov8/quickstart_interface_specs.json" \
    model_file_path:="$ONNX" engine_file_path:="$PLAN" force_engine_update:=False > "$OUT/launch.log" 2>&1 &
LAUNCH=$!
# wait for the engine file (first run builds it) and for the node to come up
for i in $(seq 1 180); do
  [ -f "$PLAN" ] && break
  if ! kill -0 $LAUNCH 2>/dev/null; then echo "launch exited early"; tail -30 "$OUT/launch.log"; exit 4; fi
  sleep 5
done
echo "== engine ready after $(( $(date +%s) - T_LAUNCH )) s ($(stat -c %s "$PLAN" 2>/dev/null || echo 0) bytes) $(date -Is)"
sleep 20
LOOP_FLAG=""; [ "$LOOP" = "1" ] && LOOP_FLAG="-l"
# the meter starts first so a single pass is counted from its first frame
python3 "$HERE/measure_quickstart.py" --seconds "$SECONDS_MEASURE" --label "$LABEL" --out "$OUT/isaac_ros_yolov8_quickstart-$LABEL.json" &
METER=$!
sleep 3
ros2 bag play $LOOP_FLAG --rate "$RATE" "$ASSETS/isaac_ros_yolov8/quickstart.bag" > "$OUT/bag_play.log" 2>&1 &
PLAY=$!
sleep 5
ros2 topic list > "$OUT/topics.txt" 2>&1 || true
ros2 node list > "$OUT/nodes.txt" 2>&1 || true
wait $METER
kill $PLAY $LAUNCH 2>/dev/null || true
sleep 3
dpkg-query -W -f='${Package} ${Version}\n' ros-jazzy-isaac-ros-yolov8 ros-jazzy-isaac-ros-tensor-rt ros-jazzy-isaac-ros-dnn-image-encoder ros-jazzy-isaac-ros-nitros tensorrt libnvinfer10 cuda-cudart-13-0 2>/dev/null > "$OUT/packages.txt" || true
cat "$OUT/packages.txt"
echo "== done $(date -Is)"
