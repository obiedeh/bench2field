#!/bin/bash
# Second pass: build the YOLOv8n engine standalone with trtexec and a bounded workspace (the in-pipeline build ran out of memory on the 8 GB board), then the measured runs.
set -u
W=~/ds-yolo-20261029
LOG=$W/orin_ds2_20261029.log
while pgrep -f orin_ds.sh > /dev/null; do sleep 5; done
cd $W/run
echo "start $(date -Is)" > $LOG
free -m | head -2 >> $LOG
rm -f $W/run/yolov8n_b1_gpu0_fp16.engine
echo "== trtexec engine build ==" >> $LOG
T0=$(date +%s)
/usr/src/tensorrt/bin/trtexec --onnx=$W/yolov8n.onnx --saveEngine=$W/run/yolov8n_b1_gpu0_fp16.engine --fp16 --memPoolSize=workspace:768M --skipInference > trtexec_build.log 2>&1
echo "trtexec exit $? in $(( $(date +%s) - T0 )) s $(date -Is)" >> $LOG
ls -la $W/run/*.engine >> $LOG 2>&1
if [ ! -f $W/run/yolov8n_b1_gpu0_fp16.engine ]; then
  echo "retry with 256M workspace" >> $LOG
  /usr/src/tensorrt/bin/trtexec --onnx=$W/yolov8n.onnx --saveEngine=$W/run/yolov8n_b1_gpu0_fp16.engine --fp16 --memPoolSize=workspace:256M --skipInference > trtexec_build_256.log 2>&1
  echo "trtexec retry exit $? $(date -Is)" >> $LOG
fi
echo "== trtexec timing of the engine alone ==" >> $LOG
/usr/src/tensorrt/bin/trtexec --loadEngine=$W/run/yolov8n_b1_gpu0_fp16.engine --iterations=200 --warmUp=500 --duration=0 > trtexec_timing.log 2>&1
echo "trtexec timing exit $? $(date -Is)" >> $LOG
for name in file camera; do
  echo "== measured run: $name ==" >> $LOG
  tegrastats --interval 1000 > tegrastats_$name.log 2>&1 &
  TS=$!
  sleep 5
  timeout 75 /opt/nvidia/deepstream/deepstream-7.1/bin/deepstream-app -c app_$name.txt > app_$name.log 2>&1
  echo "app $name exit $? $(date -Is)" >> $LOG
  kill $TS 2>/dev/null; wait $TS 2>/dev/null
  echo "PERF lines: $(grep -c PERF app_$name.log)" >> $LOG
  grep PERF app_$name.log | tail -3 >> $LOG
done
echo "== idle tegrastats ==" >> $LOG
tegrastats --interval 1000 > tegrastats_idle.log 2>&1 & TS=$!; sleep 30; kill $TS; wait $TS 2>/dev/null
cat /etc/nv_tegra_release > versions.txt; dpkg -l | grep -E "deepstream|tensorrt |cuda-toolkit-12" | awk '{print $2, $3}' >> versions.txt; nvpmodel -q >> versions.txt 2>&1; python3 -c "import ultralytics, torch, onnx; print('ultralytics', ultralytics.__version__, 'torch', torch.__version__, 'onnx', onnx.__version__)" >> versions.txt 2>&1
echo "done $(date -Is)" >> $LOG
