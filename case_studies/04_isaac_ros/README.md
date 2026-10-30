# Case study 04: Isaac ROS 4.5, YOLOv8 through the TensorRT node, in a container

**Question:** what does it take to run NVIDIA's Isaac ROS object detection
pipeline (image encoder, TensorRT node, YOLOv8 decoder, all NITROS-connected)
without changing the host, and what does the pipeline deliver on the
reference quickstart data?

Status: measured on the x86_64 workstation (RTX 5090) with the GPU shared
with an unrelated simulation process the whole time, so every rate below is
a lower bound. The Jetson AGX Thor is the board this pipeline is meant for;
its run waits for a storage decision (below) and is not measured yet.

## What ran

- **Isaac ROS 4.5** (July 2026, the last release line for JetPack 7.1; 5.0
  moved to ROS 2 Lyrical and JetPack 7.2), installed from NVIDIA's signed apt
  repository into a plain Ubuntu 24.04 container by
  [`docker/Dockerfile`](docker/Dockerfile): `ros-jazzy-ros-base`,
  `ros-jazzy-isaac-ros-yolov8`, `-tensor-rt`, `-dnn-image-encoder`,
  `-examples`, plus rosbag2. The documented route installs `isaac-ros-cli`
  on the host and lets it build the dev container; this study keeps the host
  untouched and performs the same apt install inside the container. Image
  size on the workstation: 30.2 GB (18.6 GB of it the apt layer with ROS 2,
  Isaac ROS, CUDA 13.0 runtime and TensorRT). Versions as installed are in
  each run's `packages.txt`: `libnvinfer10 10.16.1.11+cuda13.2`,
  `cuda-cudart-13-0 13.0.96`, host driver 580.196.07.
- **Model.** The YOLOv8n ONNX export from [case study 03](../03_deepstream_orin/)
  (Ultralytics, static 1x3x640x640, SHA-256 `e88ac5f680c118ce…`). The
  engine is built by `isaac_ros_tensor_rt` itself from the ONNX at first
  launch (25 s, 15.3 MB), so it matches the TensorRT library the node links;
  the container's `trtexec` comes from a newer TensorRT package (11.4) and
  would write an engine the node cannot load.
- **Data.** The quickstart assets NVIDIA publishes for this package
  (`isaac_ros_yolov8_assets` 4.0.0 on NGC): an 18-frame bag of 640x640
  frames of people cycling, 1.69 s long, looped by `ros2 bag play -l`.
- **Measurement.** [`measure_quickstart.py`](measure_quickstart.py) runs
  inside the container next to the graph: it counts input images (reliable
  subscription, the bag publishes reliably) and `/detections_output`
  messages over a 60 s window, keeps detections per frame, class ids and
  scores, and matches each detections message to its image by header stamp
  for an end-to-end latency. [`run_quickstart.sh`](run_quickstart.sh) does
  the launch, the playback and the measurement; the host samples
  `nvidia-smi` beside it. [`table.py`](table.py) prints the table below from
  the records under `logs/`.

## Results (x86_64, RTX 5090, GPU shared)

Output of `table.py`, from the records under `logs/`:

| run | bag rate | input frames (Hz) | detections out (Hz) | out / in | per frame | latency p50 / p95 (ms) | host GPU during the window |
|---|---|---|---|---|---|---|---|
| `x86_64-rtx5090-shared` | 1 | 606 (10.09) | 415 (6.91) | 0.68 | 9 | 6.21 / 893.69 | 94% util, 11.0 GiB used (host, all processes) |
| `x86_64-rtx5090-shared-rate0.4` | 0.4 | 250 (4.16) | 172 (2.86) | 0.69 | 9 | 6.77 / 2241.16 | 94% util, 11.0 GiB used (host, all processes) |
| `x86_64-rtx5090-shared-single-pass` | single pass | 18 (0.9) | 2 (0.1) | 0.11 | 9 | 2114.1 / 3293.64 | - |

Reading it:

- The pipeline runs end to end from the published packages: every detections
  message carries nine boxes (classes 0 and 1, person and bicycle, median
  score 0.81), which is what the quickstart screenshot shows.
- At the bag's native 10.1 Hz the graph emits 6.9 Hz of detections; the
  stamp-matched latency is 6.2 ms at the median. The GPU was at 92 to 94
  percent utilisation from another process before this graph started, so
  neither number is the hardware's.
- The output to input ratio is 0.68 at 10 Hz and 0.69 at 4 Hz. A drop that
  does not change with load is not load. A single pass of the bag (cold
  graph) yields two detections messages, two to three seconds after their
  frames. The missing frames are not explained by these runs; the candidates
  are the looped bag's stamps jumping back every 1.69 s (nodes that drop
  out-of-order stamps) and the first frames after launch never being
  processed. It is recorded as an open item rather than smoothed over.
- Latency p95 values (0.9 s, 2.2 s) come from the same loop-restart effect:
  a detections message that matches a stamp from the previous loop.

## The Thor

The same Dockerfile builds the Jetson image with
`--build-arg ARCH=sbsa --build-arg SUITE=noble-jetpack --build-arg JETSON_REPO=common`
(the Isaac ROS `noble-jetpack` suite and the L4T r38.4 user-space
repository, matching JetPack 7.1 on the board). Docker on the Thor stores
images under `/var/lib/docker` on the board's root disk, which has 37 GB
free; the x86 image is 30 GB, and the Jetson one will be of the same order.
That is a fixed-path write on a nearly full disk, so it is a decision for the
board's owner, not for this study. Until it runs, this case study has no
Jetson number, and it says so.

## Layout

| Path | What |
|---|---|
| `docker/Dockerfile` | the container recipe, both architectures |
| `run_quickstart.sh` | launch, playback, measurement, package inventory (runs inside the container) |
| `measure_quickstart.py` | the rclpy meter |
| `table.py` | the results table from `logs/*/isaac_ros_yolov8_quickstart-*.json` |
| `logs/<run>/` | the JSON record, `launch.log`, `bag_play.log`, `nodes.txt`, `topics.txt`, `packages.txt`, host `nvidia_smi_host.csv` |

Not in the repository: the quickstart bag and the ONNX and engine files
(fetched or built as described above).
