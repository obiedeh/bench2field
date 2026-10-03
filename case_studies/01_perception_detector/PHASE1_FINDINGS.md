# Case study 01, phase 1: baseline and end-to-end profile

Every number here comes from a file under `runs/` in this folder; `summarize_phase1.py` regenerates the tables. Model: `models/yolox_s.onnx` (provenance in `exports.json`), FP32, one frame per request. Dates: 2026-10-02.

## 1. FP32 baseline, idle bench

`b2f sweep sweeps/phase1_baseline.yaml`: TensorRT fp32 (the baseline every optimization will be measured against) alternated with CUDA fp32 (a reference for what TensorRT itself is worth), 10/30/100 Hz, 60 s per tier, three repeats. Latencies in ms; the median across the three repeats with the range in brackets. No deadline (33.3 ms) was missed in any run.

### RTX 5090 (`runs/bench_5090/`)

onnxruntime 1.30.0, TensorRT 10.16.1, cuDNN 9.19.0, CUDA 13.0. Background at run start: a `pgvector` container and an idle `uvicorn` API of another project, load average 0.25; nothing was stopped.

| variant | Hz | response p50 | response p95 | response p99 | service p95 | GPU power p50 |
|---|---|---|---|---|---|---|
| trt_fp32 | 10 | 2.148 (1.937–2.179) | 2.515 (2.446–2.694) | 2.955 | 2.251 | 69 W |
| trt_fp32 | 30 | 1.864 (1.753–2.005) | 2.093 (1.869–2.219) | 2.220 | 1.920 | 75 W |
| trt_fp32 | 100 | 1.731 (1.730–1.754) | 1.893 (1.802–1.953) | 2.027 | 1.711 | 98 W |
| cuda_fp32 | 10 | 3.009 (2.915–3.048) | 3.565 (3.400–4.109) | 4.349 | 3.396 | 69 W |
| cuda_fp32 | 30 | 2.803 (2.762–2.947) | 3.055 (2.881–3.251) | 3.293 | 2.894 | 76 W |
| cuda_fp32 | 100 | 2.739 (2.724–2.742) | 2.867 (2.835–3.016) | 3.088 | 2.685 | 102 W |

### Jetson AGX Thor (`runs/bench_thor/`)

onnxruntime 1.24.0, TensorRT 10.13.3, cuDNN 9.12.0, CUDA 13.2, power mode `120W`. See the caveat on the container below.

| variant | Hz | response p50 | response p95 | response p99 | service p95 | board power p50 |
|---|---|---|---|---|---|---|
| trt_fp32 | 10 | 10.590 (10.484–11.065) | 11.311 (11.196–11.528) | 11.714 | 11.187 | 20.9 W |
| trt_fp32 | 30 | 10.378 (10.289–10.424) | 10.942 (10.937–10.951) | 11.276 | 10.877 | 21.2 W |
| trt_fp32 | 100 | 4.079 (4.074–4.114) | 4.400 (4.373–4.497) | 4.758 | 4.337 | 34.5 W |
| cuda_fp32 | 10 | 14.064 (12.233–14.551) | 24.516 (24.265–24.706) | 24.748 | 24.307 | 21.8 W |
| cuda_fp32 | 30 | 12.964 (12.824–12.987) | 14.079 (14.050–14.141) | 14.644 | 14.014 | 22.4 W |
| cuda_fp32 | 100 | 6.526 (6.513–6.562) | 6.943 (6.935–7.010) | 7.122 | 6.878 | 46.2 W |

### What the baseline shows

- **The request rate changes the latency, on both machines, with nothing else changing.** On the Thor, TensorRT fp32 responds in 4.4 ms (p95) at 100 Hz and 10.9 ms at 30 Hz: the same model is 2.5x slower at the rover's camera rate than at a rate that keeps the GPU busy. The 5090 shows the same shape, smaller: 1.89 ms at 100 Hz, 2.52 ms at 10 Hz. The reports show why on the 5090: the SM clock p50 is 2467 MHz at 10 Hz and 2550 MHz at 100 Hz, and GPU power rises from 69 W to 98 W; between sparse frames the GPU drops into a lower power state and pays to come back. The Thor's tegrastats has no GPU clock or load channel, so the same explanation there is inferred from the power rail (`VDD_GPU` 2.8 W at 10 Hz, 11.0 W at 100 Hz) rather than measured directly. For a robot this means: **a benchmark run flat out, or at 100 Hz, reports a number the 30 Hz pipeline will never see.** Every later comparison must be made at the tier the rover uses.
- **TensorRT over CUDA provider, same ONNX file, fp32:** 1.4–1.5x on the 5090, 1.3–2.2x on the Thor (p95). This is the gain the "baseline" already contains, which is why both are recorded.
- **Repeat spread is small** except where the clock effect is in play: the 5090 10 Hz TensorRT p95 ranges 2.45–2.69 ms over three repeats, the Thor 10 Hz CUDA p50 ranges 12.2–14.6 ms. Those tiers need the repeats; at 100 Hz the p50s agree to within 2% (p95s within 8%).
- **"fp32" on the 5090 is TF32 inside TensorRT.** The nsys kernel summary (`runs/nsys/phase1_5090_rover720p_cuda_gpu_kern_sum.csv`) attributes 76% of GPU kernel time to `tf32` implicit-GEMM kernels. TensorRT allows TF32 tensor-core math for fp32 networks by default. The baseline is what TensorRT does by default with an fp32 graph, which is the honest baseline for a deployment, but phase 2's fp16 result must be read against TF32, not true fp32. Whether to also build a TF32-disabled engine is a decision for the owner (see HANDOFF.md).

### Caveat: the Thor was not container-free

The sweep was launched with `docker stop physical-ai-vllm` and the reports carry `--stopped "docker container physical-ai-vllm ..."`. That container runs with `--rm`, so the stop destroyed it, and the Safety Observability API on the Thor re-created it three minutes later, while the first run was still building its TensorRT engine. Every report's `platform.background.containers_running` shows the container up during the measured tiers, with `VLLM::EngineCore` among the busiest processes at 0.3% CPU. It was idle (no requests) but resident, holding 25% of GPU memory by its own setting. The numbers above are therefore "idle bench with an idle vLLM server loaded", and the `stopped_for_this_run` note in those six reports is wrong; the snapshots are right.

## 2. End-to-end pipeline profile, RTX 5090

`profile_pipeline.py`: one frame at a time through decode (`cv2.imdecode`), preprocess (letterbox to 640, HWC to CHW, float32: the unfused baseline), host-to-device copy, inference (TensorRT fp32, inputs and outputs bound on the GPU), device-to-host copy, and postprocess (score threshold 0.3, per-class NMS 0.45, NumPy). 300 frames after 30 of warmup; p50 and p95 per stage in ms.

Frame sets: 300 frames from the rover's own camera (Orbbec DaBai DCW2, MJPEG as the camera encodes it) at 1280x720 and 640x480, captured with `tools/capture_frames.py` while the camera happened to be looking at a blank wall (no detections, small JPEGs, so decode and NMS are at their cheapest), and 300 frames of a busy indoor scene at 640x640 cut from a video with `tools/frames_from_video.py` (1.1 detections per frame).

| stage | rover 720p | rover 720p, no spin | rover 480p | scene 640 |
|---|---|---|---|---|
| decode | 1.579 / 1.771 | 1.565 / 1.634 | 0.562 / 0.623 | 0.449 / 0.470 |
| preprocess | 3.286 / **39.985** | 1.974 / 2.230 | 1.508 / 1.998 | 1.421 / 1.627 |
| h2d copy | 0.295 / 0.419 | 0.258 / 0.297 | 0.265 / 0.333 | 0.246 / 0.278 |
| inference | 1.122 / 1.393 | 1.110 / 1.224 | 1.118 / 1.310 | 1.106 / 1.274 |
| d2h copy | 0.248 / 0.297 | 0.229 / 0.268 | 0.229 / 0.302 | 0.222 / 0.256 |
| postprocess | 0.970 / 1.199 | 0.962 / 1.031 | 0.992 / 1.297 | 0.976 / 1.052 |
| **total per frame** | 7.536 / 44.748 | 6.088 / 6.584 | 4.686 / 5.667 | 4.409 / 4.829 |
| share of total p50 that is inference | 15% | 18% | 24% | 25% |

Files: `runs/profile_5090_rover_frames_720p.json`, `..._720p_nospin.json`, `..._480p.json`, `runs/profile_5090_scene_frames_640.json`.

### Where the time goes

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
- The Orin NX (the field machine) was brought up but not baselined; phase 5 does that.
- Thermal state was not controlled (no `--cooldown-c`); temperatures stayed within 45–50 °C on the 5090 and 38–44 °C on the Thor across the sweeps, so it did not matter here.
