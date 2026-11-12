# Changelog

What changed in each Bench2Field release, newest first. Versions follow
`pyproject.toml` and the git tags. A result named here came from a committed
run file on named hardware, as the project rules require.

## Unreleased

- 2026-11-12: Case study 07 report: the stage breakdown and the energy chart are vertical columns (the build, clip and trajectory charts stay horizontal for their long labels); the title keeps "Jetson AGX Thor" on one line.
- 2026-11-12: Case study 07 report redesigned as an instrument cluster: a speedometer for decisions per second against the 10 Hz deadline (PyTorch, NVIDIA and Embedl marked), a tachometer for reasoning tokens per second, an energy gauge, and six tell-tale lamps for the findings; wider page, more space between sections, larger cards and tables. Same numbers, same sources.
- 2026-11-12: Case study 07 report (`report/index.html`, `tools/make_report.py`): one self-contained page in the Bench2Field theme with every build's median and tail against the deadline, the stage breakdown, the nine clips, the trajectory comparison against the candidate spread, energy, what each change bought, the seven issues and provenance; every number read from `runs/`. Vision-cache line corrected (stage 98 to 27 ms, p95 613 to 525; the median moved 9 ms).
- 2026-11-12: Case study 07, Alpamayo 2 Edge with NVIDIA's TensorRT Edge-LLM recipe on the Jetson AGX Thor (JetPack 7.1, TensorRT 10.13.3.9): FP8 engines as written at 367 ms median and 427 ms p95 per decision on the nine clips (one candidate; six candidates 413 and 493 ms), the same chain of causation as NVIDIA's example; a repeated-decision runner with tegrastats energy (`tools/thor_stream.py`); NVFP4 W4A16 on the language model measured slower (462 ms median) and W4A4 not buildable on this TensorRT; three Euler steps 353 ms; trajectories compared across builds with the recipe's decoder (0.01 to 0.33 m between FP8 and NVFP4, inside the 0.7 to 1.7 m spread among candidates). Seven issues with causes and fixes in the friction log (recursive submodules, nvcc pinned to CUDA 13.0, one-candidate action config, quantization config renames, CuTe DSL kernel group, TensorRT 10.13 and W4A4, a withdrawn probe number). Case study 06's issue 7 corrected: the recipe is public.
- 2026-11-10: Case study 06 scaffold, NVIDIA Alpamayo 2 Edge (the 2B-class student released 2026-11-10) on the same eight clips as case study 02: plan and budgets fixed before any timed run (0.1 s decision deadline), navigation text derived by a fixed rule from the logged path, a runner that times the upstream call without changing it, and a friction log started at setup. Results pending.
- 2026-11-10: Case study 06, BF16 eager results on both boards (24 decisions each): 0.344 s median per decision on the RTX 5090 and 2.150 s on the Jetson AGX Thor, 135 J and 130 J per decision in a sustained block, 6.0 GiB peak; the 0.1 s deadline is missed on both, as NVIDIA states for BF16 PyTorch. Energy method amended after the first run (0.1 s sampling, NVML energy counter, 30 s sustained block). Issues logged, including the missing Edge TensorRT recipe.
- 2026-11-10: Case study 06, optimization runs on the RTX 5090: upstream acceleration (0.186 s median, 0.249 s p95, 93 J), five diffusion steps, input preparation with the processor built once (0.80 s to 0.04 s per clip, identical inputs), torchao FP8 (worse in eager mode, best under compilation: 0.172 s median, 0.219 s p95, 80 J). Runner options for steps, preparation mode, FP8 and a local sample cache.
- 2026-11-10: Case study 06, vision reuse on a 10 Hz stream (`run_stream.py`): the vision tower runs only on frames not seen before, cache hits checked by fingerprint, outputs within 3.2 cm of full re-encoding. On the RTX 5090 it saves about 55 ms per decision; with upstream acceleration, FP8 and five steps the stream runs at 0.100 s median and 0.161 s p95.
- 2026-11-10: Case study 06 on the Jetson AGX Thor: upstream acceleration works once Triton is pointed at the JetPack `ptxas` (`TRITON_PTXAS_BLACKWELL_PATH`; the bundled one does not know sm_110a) and flash-attn is built from source with ninja. Thor p95 from 2.70 s as shipped to 1.21 s with acceleration, and 0.61 s on a 10 Hz stream with acceleration, FP8, five steps and vision reuse (bit-identical waypoints). Energy per decision 130 J to 66 to 74 J. README results, optimizations and issues written from the records.
- 2026-11-10: Correction to the next entry: the Orin NX was tried in fifteen load attempts, not ten; two loaded. Recounted from the logs.
- 2026-11-10: Case study 06, Jetson AGX Thor at MAXN (owner switched it): faster and less energy per decision in every variant; best stream p95 0.544 s. Rover's Jetson Orin NX 8 GB tried in ten load attempts (BF16, 8-bit, 4-bit, 4-bit on every layer, direct-to-GPU loading, page cache dropped, caching allocator off): loaded twice, never completed a decision; does not run reliably as the rover is configured. Python 3.10 workarounds recorded (virtualenv, SciPy 1.15.3, a runtime replacement for one 3.11-only format string). `run_decisions.py` gains `field-orin` and `--lowbit`.

## 1.1 (release notes drafted 2026-10-29, not yet tagged)

Alpamayo 1 on the bench and the edge, with a custom CUDA preprocessing kernel.

- Case study 02: NVIDIA's open Alpamayo 1 (`nvidia/Alpamayo-R1-10B`) measured on the RTX 5090 (sdpa and flash-attn) and the Jetson AGX Thor (sdpa, board power) on the same eight PhysicalAI-AV clips under a protocol fixed before the first run: 0.89 s against 4.38 s per decision, 377 W GPU rail against 64 W whole board, 316 J against 290 J per decision, 16 to 25 percent retention on every stage. `torch.profiler` stage split and kernel tables on both boards, six plots, a per-clip trajectory error table, and an Orin NX feasibility note.
- Phase 3b, a CUDA preprocessing kernel for the camera frames: a fused one-pass version and a separable two-pass version of the processor's resize, normalize and patch layout, a C++ host side (PyTorch binding with input checks, stream and launch check) and a standalone C++ driver built with CMake and the CUDA toolkit alone. Both variants agree with PyTorch's GPU path within one uint8 step on every committed record, which a test enforces; the standalone driver matches the extension bit for bit. Measured against PyTorch's own ops on both boards, with the kernel time under the profiler recorded.
- Tutorial: `docs/tutorials/run-alpamayo-1-on-jetson-agx-thor.md`, a numbered walk with the numbers to expect at each step.
- Case study 03, DeepStream 7.1 on the rover's Orin NX: YOLOv8n in TensorRT fp16 through `deepstream-app` with an owner-written bounding-box parser, 249.8 frames per second unpaced from a rover clip at 21.5 W and 25.1 frames per second from the live camera at 6.6 W (camera-bound), engine alone 3.99 ms; configs, parser, logs and records committed.
- Case study 01: labelled accuracy of the fp32 YOLOX-s baseline on a 500-image COCO val2017 subset (mean AP 0.652 at IoU 0.5, person AP 0.796; CUDA and TensorRT providers agree to four decimals) and the diagnosis of the earlier recall-0 result on rendered frames.
- Case study 01, phase 2 precision ladder: the same YOLOX-s export at fp32, fp16 and int8 (min-max PTQ, 100 disjoint COCO images) through TensorRT on the Jetson AGX Thor and the Orin NX, accuracy and paced speed measured together. fp16 keeps the fp32 accuracy (mean AP 0.653) and halves the response time on both boards; int8 gives up 1.5 AP points and is faster than fp16 only on the Orin NX (15.6 against 19.8 ms at 10 Hz, 5.9 against 6.3 W board), not on the Thor (7.3 against 6.8 ms), where it only lowers the GPU rail. fp32 on the Orin NX misses the 33.3 ms deadline at 10 Hz and saturates at 100 Hz. `b2f run` and `b2f sweep` gained `--trt-cache-dir`, `--calibration-table` and `--trt-workspace-mb`.
- Case study 04, Isaac ROS 4.5 in a container: the published object detection packages (image encoder, TensorRT node, YOLOv8 decoder) installed from NVIDIA's apt repository into a plain Ubuntu 24.04 container by an owner-written Dockerfile, run on the quickstart bag with the case study 03 YOLOv8n export and measured by an rclpy meter: 6.9 Hz of nine-box detections at 10.1 Hz input, 6.2 ms median stamp-to-detections latency, on a GPU shared with another process, and on the Jetson AGX Thor (JetPack 7.1, 25.4 GB image built on the board): 5.25 Hz of detections at 9.6 Hz input, 15.5 ms median latency, 23.5 W whole board and 2.2 W on the GPU rail during the window, engine built by the node in 50 s. A constant frame drop (31 percent on the workstation, 45 percent on the Thor) is recorded as an open item. Finding: a plain Ubuntu image gets no GPU from the NVIDIA runtime on the Thor unless NVIDIA_VISIBLE_DEVICES is set.
- Case study 05, Cosmos Predict 2.5: results page filled from measured records on the two boards (one 5090 generation: 773.9 s wall, 33,185,333,248-byte peak, 93 frames; the Thor admission finding that CUDA free memory tracks MemFree, so page cache blocks a run the board has room for; one Thor attempt after a page-cache drop was admitted and failed before generation, no video), with the source records pointed to by hash. The plan's own repeats remain open.
- Findings worth reading in the case study friction log: upstream's flash-attention default does not run under its own pins, and forcing it crashes the trajectory head; torch 2.8.0+cu128 runs sm80 kernels on the Blackwell 5090 while the Thor's torch 2.11 runs native sm110 GEMMs; the 5090 host's only CUDA 12 nvcc wheel contains no compiler.
- Repository: CHANGELOG, `AGENTS.md`, issue templates, the EmbodiedEdge Labs report theme, `docs/COMMIT_MAP.md`, and the `phase2` handoff branch archived as a tag.
- Private identifiers: reports now carry a machine label (`B2F_HOST_LABEL`, default `unlabelled-host`) instead of the host name, recorded process command lines have home directories replaced by `<home>`, and a repository test fails on any home path or private-range address in a tracked text file (AGENTS.md rule 9). The history was rewritten on 2026-10-30 to remove the workstation host name, one LAN address and home paths from every revision; `docs/COMMIT_MAP.md` maps the changed commit IDs.

## Unreleased

- 2026-11-03: The report header band reads "EmbodiedEdge Evidence"; the retired Labs label is removed from the renderer and the case study 01 report.
- 2026-11-03: Evidence pages and their theme name the lab EmbodiedEdge; the retired EmbodiedEdge Labs name is removed.
- README: case studies 02 to 04 linked from Read the results, the measured baseline accuracy and later case studies in Status, and "EmbodiedEdge project" in place of the retired EmbodiedEdge Labs name.
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
