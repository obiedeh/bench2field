# Case study 01, phase 1: baseline and end-to-end profile

Every number here comes from a file under `runs/` in this folder; `summarize_phase1.py` regenerates the tables. Model: `models/yolox_s.onnx` (provenance in `exports.json`), FP32, one frame per request. Dates: 2026-10-02.

> **Corrected in v1.0.1.** v1.0 attributed the Orin's 23.1 ms against 26.9 ms difference between repeats at 26 Hz to temperature. A later sweep does not support that; the text now states it as a one-off fast first repeat whose cause is not established. No measured number changed.

## 0. The rover's camera rate: 26 Hz, not 30

Before trusting any tier, the rate the rover's camera topic actually delivers was measured on the Orin with the rover's stack running (micro-ROS agent, `m3pro_bringup car_base.launch.py` with the camera, `slam_stack.sh`). The launch configures the Orbbec DaBai DCW2 colour stream at 640x480 MJPG, 30 fps (`ros2 param get` confirms). `ros2 topic hz /camera/color/image_raw` over 60 s gave a **mean of 25.9 Hz** (window averages 23.7 to 27.0), as 640x480 `rgb8`: the driver decodes MJPEG on the CPU (32% of a core) and does not keep up with the camera's 30 fps. The camera itself offers MJPEG only, in 16 sizes up to 1920x1080 at up to 30 fps. Captures: `bringup/orin/ros2_topic_hz_color.txt`, `bringup/orin/camera_v4l2_formats.txt`.

**26 Hz is therefore the headline tier.** The 33.3 ms deadline is kept as the budget's frame period.

### Headlines at 26 Hz, YOLOX-s FP32 through TensorRT, `--no-spin`

| machine | role | response p95 (median of 3, range) | p50 | misses | power | runs |
|---|---|---|---|---|---|---|
| RTX 5090 | bench | **2.04 ms** (2.00–2.14) | 1.84 ms | 0 / 4,680 | 73 W GPU | `runs/bench_5090_nospin/` |
| Jetson AGX Thor | bench | **11.04 ms** (11.02–11.05) | 10.46 ms | 0 / 4,680 | 21.1 W board | `runs/bench_thor_nospin/` |
| Jetson Orin NX (rover, idle) | field board | **26.9 ms** (23.1–26.9) | 26.6 ms | 0 / 4,680 | 9.2 W board | `runs/field_orin/` |

The Orin row is the field headline for **the model alone**: `b2f run` times inference plus the input and output copies, nothing else. On that measure the detector takes 26.9 ms of the 33.3 ms frame at the rover's rate. **The full frame on the Orin does not fit**: section 2 measures decode, preprocess, copies, inference and postprocess together on the rover's own 480p frames and gets 47.8 ms p95. On the 5090 the host stages had already taken about 4x the inference time; on the Orin they take about 0.7x, but inference itself is slower inside the frame than back to back. The bench rows are what the same file does on the two bench machines.

## 1. FP32 baseline, idle machines

`b2f sweep sweeps/phase1_baseline.yaml`: TensorRT fp32 (the baseline every optimization will be measured against) alternated with CUDA fp32 (a reference for what TensorRT itself is worth), 10/26/30/100 Hz, 60 s per tier, three repeats, onnxruntime threads sleeping between runs (`--no-spin`, recorded as `platform.allow_spinning = false`). Latencies in ms; the median across the three repeats with the range in brackets.

### RTX 5090, baseline (`runs/bench_5090_nospin/`)

onnxruntime 1.30.0, TensorRT 10.16.1, cuDNN 9.19.0, CUDA 13.0. Background at run start: a `pgvector` container and an idle `uvicorn` API of another project holding 762 MiB on the GPU; nothing was stopped. No deadline missed.

| variant | Hz | response p50 | response p95 | response p99 | service p95 | GPU power p50 |
|---|---|---|---|---|---|---|
| trt_fp32 | 10 | 2.056 (2.046–2.193) | 2.290 (2.224–3.461) | 2.405 | 2.133 | 68 W |
| **trt_fp32** | **26** | **1.835 (1.769–1.956)** | **2.038 (2.003–2.135)** | **2.181** | **1.853** | **73 W** |
| trt_fp32 | 30 | 1.809 (1.756–1.895) | 2.017 (2.010–2.033) | 2.153 | 1.858 | 75 W |
| trt_fp32 | 100 | 1.754 (1.715–1.878) | 1.963 (1.906–2.093) | 2.068 | 1.754 | 98 W |
| cuda_fp32 | 10 | 2.960 (2.914–2.987) | 4.025 (3.435–4.055) | 4.358 | 3.878 | 69 W |
| cuda_fp32 | 26 | 2.754 (2.692–2.830) | 2.953 (2.823–2.954) | 3.048 | 2.771 | 74 W |
| cuda_fp32 | 30 | 2.767 (2.748–2.846) | 2.944 (2.853–3.173) | 3.003 | 2.754 | 76 W |
| cuda_fp32 | 100 | 2.738 (2.722–2.782) | 2.904 (2.838–3.047) | 2.963 | 2.724 | 103 W |

Spin-on reference (`runs/bench_5090_rerun/`, same sweep with onnxruntime's default spin-waiting): TensorRT 26 Hz response p95 1.939 (1.922–2.192), p50 1.772; 10 Hz 2.707; 30 Hz 1.946; 100 Hz 1.952. **`--no-spin` costs about 0.1 ms in this inference-only sweep (2.04 vs 1.94 ms at 26 Hz) but cuts the end-to-end frame's p95 from 44.7 to 6.6 ms (section 2), because spinning only hurts when host-side stages compete for the CPU.** The first three-tier sweep (`runs/bench_5090/`, spin-on) agrees with the spin-on reference.

### Jetson AGX Thor, baseline (`runs/bench_thor_nospin/`)

onnxruntime 1.24.0, TensorRT 10.13.3, cuDNN 9.12.0, CUDA 13.2, power mode `120W`. Run with the Safety Observability unit stopped and its vLLM container removed; every report's `platform.background` shows no container and no vLLM process. No deadline missed.

| variant | Hz | response p50 | response p95 | response p99 | service p95 | board power p50 |
|---|---|---|---|---|---|---|
| trt_fp32 | 10 | 10.753 (10.605–10.811) | 11.301 (11.164–11.351) | 11.560 | 11.103 | 20.9 W |
| **trt_fp32** | **26** | **10.464 (10.461–10.479)** | **11.043 (11.023–11.050)** | **11.374** | **10.979** | **21.1 W** |
| trt_fp32 | 30 | 10.408 (10.319–10.429) | 10.979 (10.970–10.987) | 11.334 | 10.914 | 21.2 W |
| trt_fp32 | 100 | 3.912 (3.563–4.126) | 4.247 (4.116–4.308) | 4.517 | 4.186 | 34.7 W |
| cuda_fp32 | 10 | 11.359 (11.351–11.386) | 24.502 (24.373–24.869) | 24.921 | 24.256 | 21.9 W |
| cuda_fp32 | 26 | 14.353 (14.322–14.632) | 15.947 (15.846–15.961) | 16.477 | 15.882 | 21.8 W |
| cuda_fp32 | 30 | 13.263 (13.103–13.362) | 14.306 (14.174–14.392) | 14.693 | 14.241 | 22.1 W |
| cuda_fp32 | 100 | 6.568 (6.555–6.767) | 7.192 (7.165–7.707) | 7.631 | 7.126 | 48.5 W |

On the Thor `--no-spin` changed nothing measurable (26 Hz TensorRT 11.04 vs 11.05 ms spin-on): the inference-only sweep has no host stage for the spinning threads to fight.

Spin-on reference, same clean state (`runs/bench_thor_4tier_spin/`): TensorRT 26 Hz response p95 11.049 (11.035–11.073), p50 10.449; 10 Hz 11.307; 30 Hz 10.990; 100 Hz 4.373. CUDA 26 Hz 16.038. (That sweep was meant to be `--no-spin` and ran spin-on because the Thor's checkout was two commits stale; the provenance guard, `b2f sweep --expect-commit`, exists because of it.)

**The idle vLLM container had no measurable effect on the Thor's TensorRT latency.** The first Thor sweep (`runs/bench_thor/`, container up and idle, spinning on) and the clean spin-on sweep agree within noise at every tier: 10 Hz 11.311 vs 11.307 ms, 30 Hz 10.942 vs 10.990 ms, 100 Hz 4.400 vs 4.373 ms (p95). The container held memory, not the GPU.

### Jetson Orin NX, the rover's board, idle (`runs/field_orin/`)

onnxruntime 1.24.0, TensorRT 10.7.0, cuDNN 9.3.0, CUDA 12.6, power mode `MAXN_SUPER`. The rover's stack was off; the OLED script and `openclaw-gateway` were left running and appear in every report's background. Labelled `bench-idle` because the board was idle; the runs with the stack up, labelled `field`, are phase 5.

| variant | Hz | response p50 | response p95 | response p99 | service p95 | misses / scheduled | board power p50 |
|---|---|---|---|---|---|---|---|
| trt_fp32 | 10 | 33.680 (33.577–33.687) | 33.958 (33.837–34.054) | 34.103 | 33.845 | **1679 / 1800** | 7.1 W |
| **trt_fp32** | **26** | **26.590 (22.847–26.644)** | **26.854 (23.104–26.881)** | **26.965** | **26.757** | **0 / 4680** | **9.2 W** |
| trt_fp32 | 30 | 22.767 (22.743–22.818) | 23.022 (22.972–23.053) | 23.191 | 22.929 | 0 / 5400 | 9.7 W |
| trt_fp32 | 100 | 5891.678 | 11185.062 | 11657.404 | 12.103 | 17965 / 18000 | 23.8 W |
| cuda_fp32 | 10 | 47.729 (47.353–47.791) | 60.982 (59.626–61.238) | 61.772 | 60.883 | 1767 / 1800 | 8.2 W |
| cuda_fp32 | 26 | 26.373 (26.368–26.390) | 29.143 (29.073–29.221) | 29.368 | 29.068 | 0 / 4680 | 13.6 W |
| cuda_fp32 | 30 | 24.227 (24.223–24.232) | 24.463 (24.443–24.465) | 24.609 | 24.374 | 0 / 5400 | 15.2 W |
| cuda_fp32 | 100 | 30383.296 | 57737.702 | 60144.322 | 20.443 | 17994 / 18000 | 23.1 W |

What the Orin shows:

- **At 26 Hz the model alone takes 26.9 ms of the 33.3 ms frame** (response p95, inference plus copies), with no misses in this inference-only sweep. The full frame, measured in section 2, is 47.8 ms and misses. The three repeats split 23.1 / 26.9 / 26.9 ms. The first repeat was a one-off fast run whose cause is not established. It ran with the junction at 59.7 °C and GPU load at 53%, the other two at 63.5 °C and 65–69% load, and v1.0 of this document attributed the difference to temperature. That attribution was wrong. A later sweep on the same board (phase 2, `runs/phase2_orin_fp16/` on branch `phase2`) measured the same FP32 variant at 26.84–26.90 ms in all three repeats with the junction at 58.0–60.3 °C, including a repeat cooler than the fast one here, so junction temperature does not explain it. Treat 26.9 ms as the board's 26 Hz figure; the repeats and alternation are what exposed the outlier.
- **At 10 Hz the same model misses the deadline almost every frame** (service p95 33.8 ms, 1679 of 1800 late). The Orin clocks its GPU down hard between sparse frames; the 100 Hz tier, where the GPU never idles, serves a frame in 12.1 ms. So on this board the rate dependence is 2.8x between 10 and 100 Hz, versus 2.5x on the Thor and 1.2x on the 5090. A perception node that runs at 10 Hz to "save work" would be slower per frame than one running at 30.
- **100 Hz cannot be served:** the board sustains 83.5 Hz with TensorRT and the open-loop queue grows for the whole tier (response p95 11.2 s). This is the no-`--drop-late` policy doing what it says; the tier is there to show where saturation is, and the field runs will use the rover's rate.
- **TensorRT over CUDA:** 1.09x at 26 Hz, 1.06x at 30 Hz, 1.8x at 10 Hz, 1.7x at 100 Hz (p95). Far less than on the 5090 and the Thor at the rover's rate.
- **Power:** 9.2 W board at 26 Hz against the budget's 15 W; 23.8 W when saturated.
- **Cross-board comparison needs care.** The Thor and the Orin share onnxruntime 1.24.0 but not TensorRT (10.13.3 vs 10.7.0) or cuDNN (9.12.0 vs 9.3.0), because they run JetPack 7 and 6; `b2f retention` will warn on both when the phase 5 field runs are compared with the Thor bench runs. Idle Orin runs cannot be fed to retention as the field side at all (both sides would be `bench-idle`), which is as intended.

### What the baseline shows across machines

- **The request rate changes the latency, on every machine, with nothing else changing.** TensorRT fp32 at 26 Hz against 100 Hz: 5090 2.04 vs 1.96 ms; Thor 11.04 vs 4.25 ms (2.6x; the GPU rail reads 3.5 W at 26 Hz and 12.2 W at 100 Hz); Orin 26.9 vs (saturated; service 12.1 ms). Between sparse frames the GPU drops into a lower power state and pays to come back; on the 5090 the reports show it directly (SM clock p50 2445 MHz at 10 Hz, 2467 at 26 Hz, 2557 at 100 Hz; GPU power 68 to 98 W), on the Jetsons it is inferred from the GPU rail and GR3D load. **A benchmark run flat out reports a number the 26 Hz pipeline never sees**, and every later comparison must be made at the tier the rover uses.
- **TensorRT over the CUDA provider, same ONNX file, fp32, at 26 Hz:** 1.45x on the 5090, 1.44x on the Thor, 1.09x on the Orin (p95). This is the gain the "baseline" already contains, which is why both are recorded.
- **Repeat spread** is small except at the 5090's 10 Hz tier, where power state is in play (TensorRT p95 2.22–3.46 ms), and for one unexplained fast repeat on the Orin at 26 Hz (23.1 ms against 26.9 ms twice). Those tiers need the repeats.
- **"fp32" on the 5090 is TF32 inside TensorRT.** The nsys kernel summary (`runs/nsys/phase1_5090_rover720p_cuda_gpu_kern_sum.csv`) attributes 76% of GPU kernel time to `tf32` implicit-GEMM kernels. TensorRT allows TF32 tensor-core math for fp32 networks by default. The baseline is what TensorRT does by default with an fp32 graph, which is the honest baseline for a deployment, but phase 2's fp16 result must be read against TF32, not true fp32. Whether to also build a TF32-disabled engine is a decision for the owner (see HANDOFF.md). The Jetsons' kernels were not profiled.

### Caveat on the first Thor sweep

`runs/bench_thor/` was launched with `docker stop physical-ai-vllm` and its reports carry a `--stopped` note, but that container runs with `--rm`, so the stop destroyed it, and the Safety Observability API re-created it three minutes later, during the first run's engine build; every report's `platform.background` shows it up (idle) during the measured tiers. The `stopped_for_this_run` note in those six reports is wrong and the snapshots are right. As shown above, it made no measurable difference.

## 2. End-to-end pipeline profile, RTX 5090

`profile_pipeline.py`: one frame at a time through decode (`cv2.imdecode`), preprocess (letterbox to 640, HWC to CHW, float32: the unfused baseline), host-to-device copy, inference (TensorRT fp32, inputs and outputs bound on the GPU), device-to-host copy, and postprocess (score threshold 0.3, per-class NMS 0.45, NumPy, YOLOX's own `multiclass_nms` code). 300 frames after 30 of warmup; p50 and p95 per stage in ms. The 5090 columns are spin-on unless marked; the Thor and Orin columns are `--no-spin` on the same 480p frame set (same manifest hash) as the 5090's 480p column. The Thor ran with the Safety Observability API and its idle vLLM container up, recorded in the JSON; the Orin with its rover stack off.

Frame sets: 300 frames from the rover's own camera (Orbbec DaBai DCW2, MJPEG as the camera encodes it) at 1280x720 and 640x480, captured with `tools/capture_frames.py` while the camera happened to be looking at a blank wall (no detections, small JPEGs, so decode and NMS are at their cheapest), and 300 frames of a busy indoor scene at 640x640 cut from a video with `tools/frames_from_video.py` (1.1 detections per frame).

| stage | 5090, rover 720p | 5090, rover 720p, no spin | 5090, rover 480p | 5090, scene 640 | **Thor, rover 480p, no spin** | **Orin NX, rover 480p, no spin** |
|---|---|---|---|---|---|---|
| decode | 1.579 / 1.771 | 1.565 / 1.634 | 0.562 / 0.623 | 0.449 / 0.470 | 3.415 / 3.443 | 4.320 / 5.189 |
| preprocess | 3.286 / **39.985** | 1.974 / 2.230 | 1.508 / 1.998 | 1.421 / 1.627 | 1.689 / 1.722 | 3.001 / 3.486 |
| h2d copy | 0.295 / 0.419 | 0.258 / 0.297 | 0.265 / 0.333 | 0.246 / 0.278 | 0.629 / 0.663 | 1.227 / 1.366 |
| inference | 1.122 / 1.393 | 1.110 / 1.224 | 1.118 / 1.310 | 1.106 / 1.274 | 5.469 / 5.681 | 24.828 / 25.104 |
| d2h copy | 0.248 / 0.297 | 0.229 / 0.268 | 0.229 / 0.302 | 0.222 / 0.256 | 0.252 / 0.316 | 1.195 / 1.322 |
| postprocess | 0.970 / 1.199 | 0.962 / 1.031 | 0.992 / 1.297 | 0.976 / 1.052 | 4.465 / 4.505 | 9.533 / 11.676 |
| **total per frame** | 7.536 / 44.748 | 6.088 / 6.584 | 4.686 / 5.667 | 4.409 / 4.829 | **15.899 / 16.217** | **43.851 / 47.826** |
| share of total p50 that is inference | 15% | 18% | 24% | 25% | 34% | 56% |

Files: `runs/profile_5090_rover_frames_720p.json`, `..._720p_nospin.json`, `..._480p.json`, `runs/profile_5090_scene_frames_640.json`, `runs/profile_thor_rover_frames_480p.json`, `runs/profile_orin_rover_frames_480p.json`.

### The full frame on the Thor: 15.9 ms p50, 16.2 ms p95

Same frames, TensorRT fp32, `--no-spin`, with the Safety Observability API and its idle vLLM container up (`VLLM::EngineCore` is the busiest process in the snapshot, at idle). The frame fits the 33.3 ms deadline twice over, but **host-side work is 9.6 ms, 60% of the frame**: decode 3.4 ms, postprocess 4.5 ms (the same NumPy NMS that costs 1.0 ms on the 5090 and 9.5 ms on the Orin), preprocess 1.7 ms. Inference is 5.5 ms inside the frame against 3.2 ms back to back and 10.5 ms p50 in the 26 Hz sweep: as on the Orin, the GPU's clock state follows how busy the host keeps it, and here the host's 10 ms between inferences leaves the GPU clocked higher than the sweep's 38 ms gaps do. Copies are 0.9 ms. Board power 22.4 W p50, junction 37.6 °C. These are the blank-wall frames too, so decode and NMS are at their cheapest.

### The full frame on the Orin does not fit the 33.3 ms deadline

On the rover's board, with its own 640x480 camera frames, the whole frame costs **43.9 ms p50 and 47.8 ms p95** (max 49.4 ms), a sequential throughput of 22.9 frames per second against a camera delivering 26. Every frame would miss the 33.3 ms deadline, and a pipeline built this way would fall behind the camera by about three frames a second, one in eight. Where it goes:

- **Inference is 24.8 ms inside the frame, 22.8 ms in the 26 Hz sweep, 16.8 ms back to back.** The same `session.run` costs 16.8 ms p50 in a tight loop (the profiler's second pass), 22.8 ms when frames arrive 38 ms apart (the sweep), and 24.8 ms with 19 ms of host work between calls: the GPU drops its clocks whenever the host is busy or waiting, then pays to come back for every frame. A flat-out benchmark understates the model's cost on this board by a third; the 26 Hz sweep by about 8%.
- **Postprocess is 9.5 ms**, ten times the 5090's 1.0 ms: NumPy NMS over all 8,400 candidates on six A78 cores. It is 22% of the frame for zero detections.
- **Decode is 4.3 ms** for the camera's 38 KB JPEGs (the 5090: 0.56 ms).
- **Preprocess is 3.0 ms** (the 5090: 1.5 ms) and the two copies 2.4 ms together: the unified memory the phase 3 kernel can write into directly is worth 2.4 ms here before the kernel saves anything on the resize.
- Host stages together (decode, preprocess, postprocess): 16.9 ms, about 0.7x inference on the Orin against about 4x on the 5090 at 720p.

The board drew 9.1 W during the profile, GPU load 71% p50, junction 56.7 °C.

These frames show a blank wall: zero detections in all 300, small JPEGs (38 KB). Decode and NMS were therefore at their cheapest. A busy scene gives the decoder more to do and NMS real candidates to suppress, so the Orin frame will be slower than 47.8 ms, not faster.

### Where the time goes on the 5090

1. **Inference is 15–25% of the frame.** At the rover's 720p, the host-side stages (decode, preprocess, postprocess) take 4.5 ms with spinning off (5.8 ms with it) against 1.1 ms of inference. Optimizing the model alone cannot make this pipeline faster than about 5 ms per frame on the 5090; the host work has to shrink too.
2. **Preprocessing rivals, and at 720p exceeds, inference.** The separate-pass letterbox, transpose and cast cost 2.0 ms at 720p and 1.4–1.5 ms at 480p and 640, against 1.1 ms of inference. That is the case for the phase 3 fused preprocessing kernel: a single pass that reads each source pixel once and writes the tensor straight into device memory removes both the preprocess stage and the host-to-device copy.
3. **ONNX Runtime's spin-waiting thread pool steals CPU from the host stages.** With the default `allow_spinning`, preprocess at 720p had a p50 of 3.3 ms and a p95 of 40 ms, with frames up to 83 ms. The same preprocess run alone, with no session in the process, takes 1.5 ms p50 with no spikes (24 or 1 OpenCV threads). With `--no-spin` (intra-op threads sleep between runs) preprocess drops to 2.0 ms p50 and 2.2 ms p95, inference is unchanged, and the frame's p95 goes from 44.7 ms to 6.6 ms. The jetson-edge-ai-security case study found the same knob worth 29 W on the Thor; here it is worth a 7x reduction in tail latency on the host. **The phase 2 ladder should run with spinning off on every machine**, and `b2f run --no-spin` exists for it.
4. **Host-device copies are small on the 5090 but not nothing: 0.5 ms per frame, a third of inference.** nsys shows the actual transfers: `cudaMemcpy` host-to-device averages 229 us for the 4.9 MB input and device-to-host 142 us for the 2.9 MB output, which is about 21 GB/s, pageable-memory speed. Pinned host memory would roughly halve this; on the Jetson the input could be written into unified memory and not copied at all. That belongs with the phase 3 kernel (its spec already names zero-copy on Jetson and pinned memory on the 5090).
5. **JPEG decode scales with resolution and content:** 1.6 ms for the camera's 113 KB 720p frames, 0.56 ms at 480p, 0.45 ms for the 640x640 scene frames. On the rover the camera delivers MJPEG, so this stage is real; on the Thor it will be slower and is a candidate for the hardware JPEG decoder (`nvjpeg`), not planned in this study.
6. **Postprocess is a flat 1.0 ms** regardless of content, because the NumPy NMS filters 8400 candidates every frame. With 0–2 detections it is pure overhead and would be near zero if the score threshold were applied before box decoding. Not changed in phase 1; it is part of the baseline the phase 2 variants are measured against.

### What `b2f run` measures versus the whole frame

`session.run` with NumPy feeds, which is what `b2f run` times, is 1.57 ms p50 for this model: inference plus both copies. The sweep's 30 Hz TensorRT p50 of 1.86 ms agrees once request scheduling is added. The rover's frame, though, costs 6.1 ms on the 5090 with spinning off, so **field retention measured on `b2f run` alone over-credits model optimizations at 720p**: a model made 2x faster takes 0.55 ms off a 6.1 ms frame, a 9% shorter frame for a 50% shorter inference. Phase 2 must report both the model's own latency (for retention) and the frame's.

## 3. nsys capture

`nsys profile --trace cuda,nvtx,osrt` over 100 rover 720p frames (20 warmup), default spinning, on the 5090. The `.nsys-rep` is at `~/bench2field_profiles/phase1_5090_rover720p.nsys-rep` outside git; the four `nsys stats` summaries are in `runs/nsys/`, and the profiler's own JSON from the same run is `runs/nsys/profile_under_nsys_rover720p.json`.

- GPU kernel time is 1.08 ms per inference (238 ms over 220 inferences, warmup included), 54 distinct kernels; TensorRT's `enqueueV3` averages 464 us of host time. The GPU is busy for about the time the host measures as "inference", so the model is GPU-bound at this batch size and nothing is hiding in launch overhead.
- Memory copies: 229 us host-to-device and 142 us device-to-host per frame on average, 0.37 ms per frame, as in finding 4.
- `cudaStreamSynchronize` is 44% of CUDA API time: the pipeline is synchronous by construction (one frame at a time, `.item()`-style waits in every stage), which is the shape of a control loop, not a throughput benchmark.
- The NVTX `postprocess` range has a 867 ms maximum: the first call imports `yolox.utils.demo_utils`. It is inside the warmup and does not reach the timed frames.

## 4. Not done in phase 1

- Accuracy (COCO subset and labelled rover frames) starts in phase 2 with the first optimized variant, per the methodology.
- The Orin NX was baselined idle (section 1); the runs with the rover's stack up, the replay profile and attribution are phase 5.
- Thermal state was not controlled (no `--cooldown-c`). On the 5090 (45–50 °C) and the Thor (38–44 °C) it made no visible difference across the sweeps. On the Orin the first 26 Hz repeat was faster than the other two (23.1 against 26.9 ms) and also cooler (59.7 against 63.5 °C); v1.0 called that a 16% thermal swing, which the evidence does not support. A later sweep on the same board (phase 2, `runs/phase2_orin_fp16/` on branch `phase2`) measured the same FP32 variant at 26.84–26.90 ms in all three repeats with the junction at 58.0–60.3 °C, including a repeat cooler than the fast one here. The cause of the fast repeat is not established.
- Provenance: reports record the git commit they ran from since `0ac4fbf`. The `bench_5090_nospin` and `field_orin` sweeps predate that; their manifests carry the commit (`ac95271`) recorded afterwards from the session log, marked as such. The Orin profile ran at `86d0490`, likewise noted in its JSON.
