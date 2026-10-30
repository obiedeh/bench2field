# Changelog

What changed in each Bench2Field release, newest first. Versions follow
`pyproject.toml` and the git tags. A result named here came from a committed
run file on named hardware, as the project rules require.

## 1.1 (release notes drafted 2026-10-29, not yet tagged)

Alpamayo 1 on the bench and the edge, with a custom CUDA preprocessing kernel.

- Case study 02: NVIDIA's open Alpamayo 1 (`nvidia/Alpamayo-R1-10B`) measured on the RTX 5090 (sdpa and flash-attn) and the Jetson AGX Thor (sdpa, board power) on the same eight PhysicalAI-AV clips under a protocol fixed before the first run: 0.89 s against 4.38 s per decision, 377 W GPU rail against 64 W whole board, 316 J against 290 J per decision, 16 to 25 percent retention on every stage. `torch.profiler` stage split and kernel tables on both boards, six plots, a per-clip trajectory error table, and an Orin NX feasibility note.
- Phase 3b, a CUDA preprocessing kernel for the camera frames: a fused one-pass version and a separable two-pass version of the processor's resize, normalize and patch layout, a C++ host side (PyTorch binding with input checks, stream and launch check) and a standalone C++ driver built with CMake and the CUDA toolkit alone. Both variants agree with PyTorch's GPU path within one uint8 step on every committed record, which a test enforces; the standalone driver matches the extension bit for bit. Measured against PyTorch's own ops on both boards, with the kernel time under the profiler recorded.
- Tutorial: `docs/tutorials/run-alpamayo-1-on-jetson-agx-thor.md`, a numbered walk with the numbers to expect at each step.
- Case study 03, DeepStream 7.1 on the rover's Orin NX: YOLOv8n in TensorRT fp16 through `deepstream-app` with an owner-written bounding-box parser, 249.8 frames per second unpaced from a rover clip at 21.5 W and 25.1 frames per second from the live camera at 6.6 W (camera-bound), engine alone 3.99 ms; configs, parser, logs and records committed.
- Case study 01: labelled accuracy of the fp32 YOLOX-s baseline on a 500-image COCO val2017 subset (mean AP 0.652 at IoU 0.5, person AP 0.796; CUDA and TensorRT providers agree to four decimals) and the diagnosis of the earlier recall-0 result on rendered frames.
- Case study 01, phase 2 precision ladder: the same YOLOX-s export at fp32, fp16 and int8 (min-max PTQ, 100 disjoint COCO images) through TensorRT on the Jetson AGX Thor and the Orin NX, accuracy and paced speed measured together. fp16 keeps the fp32 accuracy (mean AP 0.653) and halves the response time on both boards; int8 gives up 1.5 AP points and is faster than fp16 only on the Orin NX (15.6 against 19.8 ms at 10 Hz, 5.9 against 6.3 W board), not on the Thor (7.3 against 6.8 ms), where it only lowers the GPU rail. fp32 on the Orin NX misses the 33.3 ms deadline at 10 Hz and saturates at 100 Hz. `b2f run` and `b2f sweep` gained `--trt-cache-dir`, `--calibration-table` and `--trt-workspace-mb`.
- Case study 04, Isaac ROS 4.5 in a container: the published object detection packages (image encoder, TensorRT node, YOLOv8 decoder) installed from NVIDIA's apt repository into a plain Ubuntu 24.04 container by an owner-written Dockerfile, run on the quickstart bag with the case study 03 YOLOv8n export and measured by an rclpy meter: 6.9 Hz of nine-box detections at 10.1 Hz input, 6.2 ms median stamp-to-detections latency, on a GPU shared with another process, and on the Jetson AGX Thor (JetPack 7.1, 25.4 GB image built on the board): 5.25 Hz of detections at 9.6 Hz input, 15.5 ms median latency, 23.5 W whole board and 2.2 W on the GPU rail during the window, engine built by the node in 50 s. A constant frame drop (31 percent on the workstation, 45 percent on the Thor) is recorded as an open item. Finding: a plain Ubuntu image gets no GPU from the NVIDIA runtime on the Thor unless NVIDIA_VISIBLE_DEVICES is set.
- Findings worth reading in the case study friction log: upstream's flash-attention default does not run under its own pins, and forcing it crashes the trajectory head; torch 2.8.0+cu128 runs sm80 kernels on the Blackwell 5090 while the Thor's torch 2.11 runs native sm110 GEMMs; the 5090 host's only CUDA 12 nvcc wheel contains no compiler.
- Repository: CHANGELOG, `AGENTS.md`, issue templates, the EmbodiedEdge Labs report theme, `docs/COMMIT_MAP.md`, and the `phase2` handoff branch archived as a tag.

## Unreleased

- Case study 02, Alpamayo 1 on the bench and the edge: measured decision latency, memory and power on the RTX 5090 (sdpa and flash-attn) and the Jetson AGX Thor (sdpa, board power), a torch.profiler stage split with plots, and an Orin NX feasibility note. Findings in the case study README and friction log, including that upstream's flash-attention default does not run under its own pins.
- Branch `phase2` (the 2026-10-03 handoff state) archived as tag `archive/phase2-handoff-20261003`; its HANDOFF.md conflicts with master and its tests were superseded by v1.
- Report theme: the EmbodiedEdge Labs evidence theme (dark default, light and print variants), with readable power-card tables and the case study 01 report regenerated with it.
- `AGENTS.md`: the working rules for anyone changing the repository.
- GitHub issue templates (bug report, share a result) and a manual trigger for the tests workflow.
- README: the case study 01 report is linked through GitHub Pages so it opens as a page.
- Tests: "self-contained" for the report means nothing is fetched from the network; navigation links are allowed.
- Publication: `docs/COMMIT_MAP.md` maps the commit IDs recorded in run files to the rewritten history.

## 1.0.1 (2026-10-03)

- Corrected the Orin NX thermal claim: the fast first repeat was a one-off and its cause is not established. The repeats panel in the report no longer implies a thermal cause. The correction stays in the record (rule 6).
- Version 1.0.1; report regenerated.

## 1.0 (2026-10-03)

Measure and diagnose: bring-up on three boards and case study 01, phase 1.

- Bring-up with real-hardware fixtures on the RTX 5090, the Jetson AGX Thor and the rover's Jetson Orin NX: NVML and tegrastats samplers, board power rails (`VIN` on the Thor, `VDD_IN` on the Orin NX) reported as `power_board_w`, the ONNX Runtime wheel that works on each board recorded in `bringup/`.
- Run contract (schema 1.1): response-time statistics per tier, `--drop-late` with dropped frames counted apart from deadline misses, TensorRT, cuDNN and CUDA runtime versions recorded from the loaded libraries, and a snapshot of what else the machine was doing.
- Repeats and `b2f sweep`: alternated repeats of several variants, one process and one report per run, with a manifest of the order.
- Field-load replay calibrates stressor duty with the model idle, then freezes it; the GPU stressor runs in its own process.
- Case study 01 (YOLOX-s on the rover): FP32 baselines at the camera's real rates on all three boards, the end-to-end pipeline profile, an Nsight Systems capture and `PHASE1_FINDINGS.md`.
- Report (`b2f report`) built from the committed run files only; clean-clone test; history scrubbed of hostnames and addresses before publication.

## 0.1 (2026-10-02)

- First commit: runner, report contract, field retention, gap attribution, load replay and verdicts.
- Python 3.10 support (JetPack 6 ships it); `runs/` anchored to the repository root so case-study runs can be committed.
