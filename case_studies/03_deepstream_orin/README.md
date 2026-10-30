# Case study 03: DeepStream 7.1 on the rover's Jetson Orin NX, YOLOv8n through TensorRT

**Question:** what does NVIDIA's DeepStream pipeline deliver on the rover's
8 GB Orin NX with a small detector in TensorRT fp16, from the rover's own
camera and from a recorded rover clip, in frames per second and board power,
and what does it take to get it running on a board that is also the rover's
ROS 2 computer?

Measured 2026-10-29 on `field-orin` (Jetson Orin NX 8 GB, JetPack 6.2,
L4T R36.4.7, DeepStream 7.1.0, TensorRT 10.7.0, CUDA 12.6, `MAXN_SUPER`),
nothing changed at system level: no package install, no power-mode change,
everything under the home directory. Every number below comes from a record
under `runs/` built by `summarize_run.py` from the logs under `logs/`.

## What ran

- **Model.** Ultralytics YOLOv8n, exported to ONNX on the board with the
  Ultralytics package already installed there (`yolo export`, opset 12,
  static 1x3x640x640, no simplifier; SHA-256s in the records), then built
  into a TensorRT fp16 engine with `trtexec` (`logs/trtexec_build.log`):
  434 s, 9.0 MB engine. The first attempt let `nvinfer` build the engine
  inside the running pipeline and ran out of GPU memory on the 8 GB board
  with the desktop, a notebook server and the rover's own services
  resident (`logs/engine_build_run.log`); building standalone with a
  768 MB workspace is the recorded path.
- **Parser.** DeepStream's `nvinfer` needs a bounding-box parser for the
  export's `[84, 8400]` output. `parser/nvdsparsebbox_yolov8.cpp` (about 60
  lines, written for this study, no third-party code) takes the best class
  per column, applies the per-class threshold, converts centre and size to a
  clipped rectangle and leaves clustering to `nvinfer` (cluster mode 2, IoU
  0.45, pre-cluster threshold 0.25). Built with the toolkit's `g++` against
  the DeepStream headers (`logs/parser_build.log`).
- **Pipeline.** `deepstream-app` with the configs under `configs/`: one
  source, `nvstreammux` at 640x480, `nvinfer` batch 1 every frame, a fake
  sink (no display, no encode), the app's own one-second performance
  counter (`enable-perf-measurement=1`). Two sources:
  `file`, the rover's 480p clip (300 frames from case study 01's
  `rover_frames_480p`, H.264, looped, not paced to real time, so the
  pipeline runs as fast as it can), and `camera`, the rover's RGB camera
  over V4L2 at 640x480 MJPEG, live.
- **Telemetry.** `tegrastats --interval 1000` beside each run, 30 s of
  idle after the runs; the engine alone timed with `trtexec --loadEngine`.

## Results

From `runs/deepstream_field-orin_file.json` and
`runs/deepstream_field-orin_camera.json` (per-second samples after the
first 10 s of each 70 s run; VDD_IN is the board rail):

| | File source, looped, unpaced | Live camera, 640x480 MJPEG |
|---|---|---|
| Frames per second, median | 249.8 | 25.1 |
| Frames per second, p5 to p95 | 249.2 to 250.4 | 25.0 to 25.1 |
| Board power VDD_IN, median | 21.5 W | 6.6 W |
| Board power VDD_IN, max | 21.7 W | 6.7 W |
| GPU load (GR3D), median | 94 % | 29 % |
| Junction temperature, max | 70.0 C | 59.9 C |
| Samples | 64 | 64 |

Idle board power over 30 s after the runs: 5.44 W median. Engine alone
under `trtexec` (`logs/trtexec_timing.log`): 3.99 ms median latency per
image, 283 inferences per second, GPU compute 3.51 ms.

## Reading the numbers

- **The engine is the ceiling, and the pipeline reaches it.** The file
  source runs the whole DeepStream pipeline (decode, scale, batch, infer,
  parse, cluster) at 250 frames per second at 94 percent GPU load, against
  283 inferences per second for the bare engine: the pipeline costs about
  12 percent over the engine alone and nothing else in it is a bottleneck
  at this resolution. 21.5 W for 250 detections per second is 86 mJ per
  frame, whole board.
- **The camera is the limit when it is live.** With the rover's camera the
  pipeline delivers 25.1 frames per second at 29 percent GPU load and 6.6 W,
  1.2 W above idle: the detector waits for frames. Case study 01 measured the
  same camera's topic at 25.9 Hz through ROS 2 and found the camera delivers
  under its nominal 30; this is the same camera-bound figure through a
  different stack. Under the 33 ms budget case study 01 set for the rover,
  DeepStream has about 29 ms of headroom per frame at this load.
- **Against the ONNX Runtime path of case study 01.** On the same board,
  case study 01's YOLOX-s fp32 through ONNX Runtime's TensorRT provider
  took 24.8 ms of inference inside a 43.9 ms frame and missed the deadline;
  here a smaller fp16 model inside DeepStream's native pipeline takes 4 ms
  and the frame is camera-bound. Different model, different precision,
  different stack: this is not a like for like comparison, and the point is
  what DeepStream makes cheap (decode to inference without leaving the GPU)
  rather than which detector is better.
- **What is not measured.** Detection quality on the rover's scenes (no
  labels exist; case study 01 carries the labelled COCO baseline for its
  model, this model has none here), the clip paced to real time, the ROS 2
  stack running at the same time (the rover's services were resident but
  idle; see the host state in the tegrastats logs), and any display or
  encode sink.

## Friction

- The in-pipeline engine build ran out of GPU memory on the 8 GB board
  (`logs/engine_build_run.log`): TensorRT's builder and the decoder shared
  the memory left by the desktop and the rover's services (about 3.3 GB
  available). `trtexec` with a 768 MB workspace built the engine in 434 s.
- `deepstream-app`'s multi-URI source type (3) reported the clip as not
  found; the plain URI source type (2) played it. The committed file config
  uses type 2.
- `nvpmodel -q` prints errors for CPU cores 6 and 7 on this module
  (`logs/versions.txt`): the Orin NX 8 GB exposes six cores and the tool
  queries eight. Harmless, recorded.

## Layout

```text
README.md              this file
summarize_run.py       PERF and tegrastats logs to one JSON record per run
run_orin.sh            the run script as executed on the board (paths generalised)
configs/               nvinfer and deepstream-app configs for the file and camera runs
parser/                the YOLOv8 bounding-box parser (C++) and the COCO labels
logs/                  app, tegrastats, trtexec, export, parser build and version logs, sanitized
runs/                  the two measured records
```

Raw logs, the clip, the ONNX export, the engine and the weights are kept on
the lab's backup volume under `nvidia-gaps-20261029/orin/deepstream/`.
