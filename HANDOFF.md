# Handoff

State of Bench2Field at the v1.0 release. Phase 1 is v1.0 (measure and diagnose); phases 2 to 5 continue on `master` afterwards as v2.

The GitHub repo is `github.com/obiedeh/bench2field`, **private** until the owner publishes it. PR #1 (`hardware-bringup`) merged into `master`; the v1.0 release work is on `v1-release` with its own PR. CI runs pytest on CPU for Python 3.10 and 3.12.

**Identifiers:** machine hostnames, the rover's LAN address and home-directory paths were replaced in every tracked file with the board labels `bench-5090`, `bench-thor` and `field-orin` Where this file shows `bench-thor:` or `field-orin:` in an `rsync`/`ssh` command, that is the operator's SSH alias for the board. The history was rewritten before publication so no commit contains them; `docs/COMMIT_MAP.md` maps the original commit IDs recorded in run files to the current ones.

Last updated 2026-10-03, v1.0 release candidate. Steps 1 to 4 are done: bring-up on all three machines, publish (private repo, PR #1 open), and case study 01 phase 1: `--no-spin` FP32 baselines at 10/26/30/100 Hz on the 5090, the Thor (clean of vLLM) and the idle Orin NX, the end-to-end profile, the nsys capture, and `PHASE1_FINDINGS.md` with 26 Hz (the rover camera's delivered rate) as the headline tier. The owner reviews the findings against the reports before the PR merges. Phase 2 has not started.

## Done

**Fixes requested before bring-up** (one commit each, with tests)

| Commit | Change |
|---|---|
| `02c3ded` | Python 3.10: `timezone.utc` instead of `datetime.UTC`. Suite verified under CPython 3.10.20. |
| `2f685d9` | `.gitignore`: `runs/` anchored to the repo root, so `case_studies/*/runs/` can be committed. |
| `9b60aaf` | Schema 1.1: response-time stats per tier (`TierResult.response`), `--drop-late` with `dropped` counted apart from `deadline_misses`. Response p95 is the default stat for retention, attribution and validity. Verdicts gate response p99 (the budget's percentile) and count late plus dropped frames against the miss budget; both accepted by the owner. |
| `c7c5d1c`, `38a3d3b` | Replay calibrates stressor duty with the model idle, then freezes it. Duty, achieved utilisation and the sampler that measured it are recorded as `platform.replay_calibration`. |
| `3dedb45` | Loud `TelemetryWarning` when a sampler is missing, reason recorded as `platform.telemetry_missing`; samplers closed after a run. |
| `16c462b` | GPU stressor in its own spawned process with its own CUDA context; all stressors use spawn. |
| `67eb6df` | Second half of the telemetry fix: Jetson total power is reported as `power_board_w` (mapped from `VIN` on the Thor), separate from NVML's `power_gpu_w`. |

**Step 1, RTX 5090** (evidence and exact versions in `bringup/rtx5090/`)

- Environment: `.venv`, Python 3.12.3, `pip install -e ".[dev,tools,nvml]"`, `onnxruntime-gpu[cuda,cudnn]==1.30.0`, `tensorrt-cu13==10.16.1.11`, and `torch==2.11.0+cu130` for the GPU stressor. Installing torch moved cuDNN from 9.27.0.42 to 9.19.0.56; the CUDA Conv test was re-run against 9.19 and passes.
- ONNX Runtime wheel: 1.30.0 is the current PyPI release and is built for CUDA 13.0 and cuDNN 9, which is what driver 580.178.04 supports.
- 81 tests pass in the venv, including four that only run on real hardware (CUDA Conv, TensorRT Conv in fp16, NVML channels, GPU stressor process).
- NVML sampler validated: all six channels populated and plausible in `run_cuda_fp32_30hz.json`.
- TensorRT fp16 through ONNX Runtime works: `run_trt_fp16_30hz.json`.
- GPU stressor at a 50% target, open loop: NVML measured a median of 50% (range 49 to 51) over 30 s, in `gpu_stressor_50pct.json`.

**Step 2, Jetson AGX Thor** (evidence in `bringup/thor/`, details in `bringup/README.md`)

- Installed at `~/github/bench2field` on `bench-thor` (a copy of this branch, synced with rsync), with its own `.venv`. 80 tests pass there, 1 skipped (PyTorch is not installed on the Thor).
- ONNX Runtime wheel: `onnxruntime-gpu==1.24.0` from the Jetson AI Lab index. The PyPI 1.30.0 aarch64 wheel does not work on the Thor.
- tegrastats parser: every one of 60 real lines parses completely.
- Board power rail: `VIN`, reported as `power_board_w`. The rover budget's `power_channel` now names `power_board_w`.
- `nvpmodel` mode is recorded correctly (`NV Power Mode: 120W`). Power mode, fan and clocks were not touched.
- CUDA and TensorRT fp16 runs of the tiny model both work.
- Replay steering, 60 s, CPU target 40%: duty frozen at 0.342, 42.4% measured with the model idle, 41.9% median during the run. Memory-bandwidth target 30%: not measurable on the Thor, ran open-loop.

**Step 2, Jetson Orin NX on the rover** (evidence in `bringup/orin/`, details in `bringup/README.md`)

- L4T R36.4.7 (JetPack 6), **Python 3.10.12**: the suite runs there, 110 passed, 2 skipped (no PyTorch, NVML unusable).
- ONNX Runtime wheel: `onnxruntime-gpu==1.24.0` from the Jetson AI Lab `jp6/cu126` index, the same version as the Thor, as the owner asked. TensorRT 10.7.0, CUDA 12.6.
- The cuDNN that actually loads is 9.3.0 (from the `cross-aarch64` package `ldconfig` points at), not the installed 9.11.0.98. Recorded in every report; not changed.
- Board power rail: `VDD_IN`, now mapped to `power_board_w`, so the rover budget's power gate resolves there.
- tegrastats has `GR3D_FREQ` but no `EMC_FREQ`; NVML answers no telemetry query at all.
- Replay steering, CPU target 40%: converged at duty 0.118 (the rover's own services were already using CPU), 41.7% measured idle, 35.5% median during the run.

**Owner's decisions applied after bring-up**

- Every report records the TensorRT, cuDNN and CUDA runtime versions the providers actually loaded (`platform.tensorrt`, `.cudnn`, `.cuda_runtime`), asked of the libraries themselves.
- Retention treats the onnxruntime version like power mode: repeats and baseline/optimized pairs must match, bench vs field differing is a warning. 5090 stays on 1.30.0, Jetsons on 1.24.0.
- Every report records what else the machine was doing (`platform.background`: running containers, the five busiest processes, load average) and what was stopped for the run (`b2f run --stopped ...`, or `stopped:` in a sweep config).
- Synthetic inputs follow each input's dtype, so RT-DETR's int64 input would work.

**Step 3**: `LICENSE` (Apache-2.0), `CONTRIBUTING.md`, `.github/workflows/ci.yml` (pytest on CPU, Python 3.10 and 3.12), the private GitHub repo, and a PR from `hardware-bringup` to `master` so CI runs (opened at the end of phase 1; link in the PR section below).

**Step 4.1 and 4.2: detector and export**

- YOLOX (Apache-2.0), pinned to commit `6ddff482`. YOLOX-s is the student, YOLOX-l the teacher (not yet exported; `export_yolox.py l` does it).
- `models/yolox_s.onnx`: static 1x3x640x640, opset 11, decoding in the graph, NMS outside. Provenance (weights and ONNX SHA-256, versions, PyTorch agreement check) in `case_studies/01_perception_detector/exports.json`; the choices and why in that folder's README.
- The same file is on the 5090, the Thor and the Orin (`models/`), same hash on all three.
- `case_studies/01_perception_detector/sweeps/phase1_baseline.yaml` is the baseline sweep.

**Step 4.3 and 4.4: baselines, profile, findings** (all under `case_studies/01_perception_detector/`)

- `runs/bench_5090/` and `runs/bench_thor/`: six reports plus manifest each, TensorRT fp32 alternated with CUDA fp32, 10/30/100 Hz, three repeats.
- `runs/profile_5090_*.json`: the pipeline profile on rover camera frames (720p, 480p) and a busy scene (640), plus the 720p run with `--no-spin`. `runs/nsys/`: the `nsys stats` summaries; the `.nsys-rep` is at `~/bench2field_profiles/phase1_5090_rover720p.nsys-rep`, outside git.
- `PHASE1_FINDINGS.md`: the write-up. Headlines: request rate changes latency 2.5x on the Thor (GPU power states); preprocessing rivals inference and exceeds it at 720p; onnxruntime's spin-waiting thread pool puts a 40 ms p95 on preprocess that `--no-spin` removes; copies are a third of inference on the 5090; "fp32" in TensorRT on the 5090 is TF32.
- Frame sets live under `data/` (ignored) on the 5090 host and the Orin, with a hash manifest each; `tools/capture_frames.py` and `tools/frames_from_video.py` regenerate them.

**Start of step 4: repeats and `b2f sweep`** (the deferred methodology work, built before any baseline run)

- `field_retention` and `b2f retention` take the repeats of each run: speedups from medians, each group's range printed, warnings for fewer than three repeats or a gain no larger than the spread. They refuse runs with different deadlines or drop-late policies, repeats that differ in variant, environment or power mode, and a baseline and optimized run in different power modes. Bench and field in different power modes is a warning, because they may be different boards.
- `b2f sweep <config.yaml> --out-dir <dir>` runs the repeats of several variants alternately, one process and one report per run, with a manifest of the order. Checked on the 5090: `bringup/rtx5090/sweep_tiny/`.
- Attribution and replay validity still take one run each; they are phase 5 work.

## What broke on real hardware, and the fix

| Where | Problem | Fix |
|---|---|---|
| 5090 | CUDA provider loaded, then failed on the first Conv: `dlopen failed for libcudnn.so`. The pip-installed cuDNN is under `site-packages/nvidia/`, off the loader path. | `e46d27f`: the backend calls `onnxruntime.preload_dlls()` for CUDA and TensorRT. |
| 5090 | NVML kept reporting the flat-out warmup for about 3 s into the tier, so a 70 W tier was summarised with a 354 W peak. Real capture: `tests/fixtures/nvml_5090_after_flatout_warmup.json`. | `e468e64`: warmup runs at the tier's own rate. |
| 5090 | With no TensorRT libraries, `--provider tensorrt` logged an error, fell back to CPU and exited 0. | `8440572`: the backend raises if the active provider is not the one requested. |
| 5090 | pip-installed TensorRT is in `site-packages/tensorrt_libs`, off the loader path. | `b8873aa`: the backend loads `libnvinfer`, its plugin library and the ONNX parser by full path. |
| 5090 | The host exports a ROS 2 `PYTHONPATH`, which leaks ROS's Python packages into any venv. | Not a code change: see the box below. |
| Thor | NVML on the Thor raises `NotSupported` for memory info and clock info, so the NVML sampler failed outright. | `044bf15`: unsupported channels are left out, other NVML errors still propagate. |
| Thor | The PyPI `onnxruntime-gpu==1.30.0` aarch64 wheel has no TensorRT provider, and its CUDA provider fails on the first Relu with `cudaErrorNoKernelImageForDevice`. | Use the Jetson AI Lab wheel (1.24.0). No code change. |
| Thor | tegrastats prints no `GR3D_FREQ` and no `EMC_FREQ`, idle or under load. | Documented and pinned by a test; see open questions. |
| Thor | A sweep meant to be `--no-spin` ran spin-on because the Thor's checkout was two commits behind (my mistake). | `0ac4fbf`: every report and manifest records the git commit and dirty flag; `b2f sweep --expect-commit` refuses a wrong or dirty checkout. Always sync the board and pass `--expect-commit` before a remote sweep. |
| Thor | `docker stop physical-ai-vllm` for the baseline sweep **destroyed the container**: it runs with `--rm`. The Safety Observability API (`uvicorn api.main:app`, port 8081, under the user's systemd) re-created it three minutes later, so the sweep ran with it up and idle. | Not fixable from here. The reports' `background` snapshots record the truth; the findings say so. Do not `docker stop` that container again; ask the owner how the API should be told to stand down. |
| 5090 | Preprocess p95 of 40 ms in the pipeline profile. | Not a bug in the profiler: onnxruntime's spin-waiting intra-op threads compete with OpenCV. `--no-spin` on the profiler and on `b2f run` removes it; phase 2 should run with spinning off everywhere. |

**Every command on the 5090 host must be run as `env -u PYTHONPATH .venv/bin/<command>`.** The host's shell sources ROS 2 Jazzy, whose `PYTHONPATH` takes precedence over the venv; without unsetting it, pytest picks up ROS's plugins and fails on import. The Thor's shell does not set `PYTHONPATH`, so plain `.venv/bin/<command>` works there.

## Git history note

Commit `236aefc` (`b2f sweep --stopped`) accidentally includes two in-progress run files from the 5090 sweep (`runs/bench_5090/trt_fp32_r1.json`, identical to its final content, and a partial sweep manifest, superseded in `aad717a`). Harmless, and left as it was when the history was rewritten for publication.

**Baseline run directories** (all under `case_studies/01_perception_detector/runs/`): the baselines are `bench_5090_nospin/`, `bench_thor_nospin/` and `field_orin/` (idle Orin, labelled `bench-idle`). References kept as evidence: `bench_5090_rerun/` (spin-on, four tiers), `bench_thor_4tier_spin/` (clean, spin-on: ran from a stale checkout), `bench_5090/` and `bench_thor/` (first three-tier sweeps, spin-on; the Thor one with the container up), `bench_thor_rerun_stopped/` (one run, stopped by the owner).

**Thor stand-down procedure that worked** (for any future clean Thor run): `systemctl --user stop physical-ai-safety` (the Safety Observability API is the user unit `physical-ai-safety.service`, enabled, `Restart=on-failure`), then `docker stop physical-ai-vllm` (it runs with `--rm`, so it is removed; expected), wait 3 minutes, confirm `docker ps` empty and `nvidia-smi --query-compute-apps` empty, run with `b2f sweep ... --expect-commit <hash>`, then `systemctl --user start physical-ai-safety`; the API re-creates the container within a few minutes.

## Blocked, waiting on the owner

- Owner's review of `PHASE1_FINDINGS.md` against the reports on GitHub before PR #1 merges.
- Phase 2 needs the decisions under open questions (TF32, the kernel's output scale) before its first variant is measured.

## Deferred by decision

- **Thermal hold** (fix before phase 5). `--only thermal` soaks and then stops heating, so the device cools during the tiers. On a discrete GPU the soak heats the CPU but reads GPU temperature, so it would run to its 600 s timeout.

- **Orin replay drift (explain before trusting replay validity, phase 5).** On the Orin, the CPU stressor was calibrated to a 40% target with the model idle: duty frozen at 0.118, 41.7% measured. During the 60 s tier that followed, CPU utilisation had a median of 35.5% with the duty unchanged. The rover's own services were still settling after boot, so the background load the calibration absorbed was not steady. Until this is explained (and the recording rule "steady-state background load" is enforced or checked), a replay's validity number should not be trusted on its own. No fix now.

- **Field-load profile must be recorded without rviz2 (phase 5).** `bringup/orin/tegrastats_rover_stack.txt` (60 s with micro-ROS agent, `car_base.launch.py` with the camera, and `slam_stack.sh`) includes `rviz2`, which `slam_stack.sh` starts and which took a full core; rviz does not run on the rover in the field. Record the replay profile with the field stack only: micro-ROS agent, `car_base` with the camera, SLAM, no rviz. Keep the existing capture as a reference, not as the profile.

## Open questions

- **TF32 in the fp32 baseline.** TensorRT runs the fp32 graph with TF32 tensor-core kernels on the 5090 by default (76% of GPU kernel time in the nsys capture). Phase 2's fp16 and INT8 gains will be measured against TF32, not true fp32. Keep that (it is what a deployment gets), or also build a TF32-off engine (`trt_builder_optimization_level`/`TF32` flags via `extra_provider_options`) for the write-up? The Thor's kernels were not profiled.
- **Spinning.** Decided: baselines and every phase 2 variant run with `--no-spin` (`no_spin: true` in the sweep config; reports record `platform.allow_spinning = false`). The library default is unchanged.
- **Request-rate dependence.** Latency at 10 to 30 Hz is far worse than at 100 Hz on all three machines because the GPU drops power states between frames; on the Orin the detector misses the 33.3 ms deadline at 10 Hz and fits it at 26 Hz. Retention at 26 Hz will therefore compare numbers partly set by clock state. Options for the owner: accept (it is what the robot sees), add `jetson_clocks`/locked clocks as a recorded setting (changes the board, so not without asking), or report both tiers. Nothing changed.
- **Unexplained fast repeat on the Orin (corrected in v1.0.1).** In the idle Orin sweep the 26 Hz TensorRT p95 was 23.1 ms in the first repeat (junction 59.7 °C) and 26.9 ms in the next two (63.5 °C). v1.0 called this a thermal swing. A later sweep on the same board (phase 2, `runs/phase2_orin_fp16/` on branch `phase2`) measured the same FP32 variant at 26.84–26.90 ms in all three repeats with the junction at 58.0–60.3 °C, including a repeat cooler than the fast one here. The cause of the 23.1 ms repeat is not established; candidates not yet tested include GPU clock state at the start of a sweep and CPU frequency. The thermal-hold work before phase 5 is still needed for replay, but not on the strength of this observation.
- **Kernel spec versus the model.** `kernels/README.md` says the fused preprocessing kernel scales pixels to [0, 1]; YOLOX takes raw 0..255. The kernel's scale (and BGR/RGB order) should match `export_yolox.py`'s input convention before phase 3.
- **Postprocess cost.** NumPy NMS over all 8400 candidates costs 1.0 ms per frame even with no detections. Thresholding before decoding would make it near zero. Left as is so phase 2 measures against the same baseline; worth fixing in the pipeline before phase 5.
- **Copies and pinned memory.** Host-device copies run at pageable speed (21 GB/s) on the 5090. Pinned host memory and, on Jetson, unified memory belong to the phase 3 kernel work and are not in `b2f run`.
- **Frames looking at a wall.** The rover camera frames used for the profile show a blank wall, so decode and NMS are at their cheapest. Re-capture with the rover in its working environment before accuracy work.

- **ONNX Runtime versions differ between machines.** The 5090 runs 1.30.0 (TensorRT 10.16.1.11) and the Thor runs 1.24.0 (system TensorRT 10.13.3.9), because no 1.30.0 wheel works on the Thor. Latency on the two is not a same-runtime comparison. Options: accept and record it, pin the 5090 to 1.24.0, or build 1.30.0 from source on the Thor (well over 15 minutes).
- **Memory-controller load is not measurable on either Jetson** (no `EMC_FREQ` on the Thor or the Orin NX), so the memory-bandwidth stressor is always open-loop there. GPU load: the Orin reports `GR3D_FREQ`; the Thor does not, though NVML on the Thor does report GPU utilisation, so the Jetson sampler could take that one channel from NVML. Not built; needs a decision.
- **Bench machines were not idle.** The Thor had the `urban-edge-vllm` container loaded during bring-up, and an Ollama evaluation job started on the 5090 part-way through (one contaminated run was discarded). Reports now record the containers and busiest processes at run start, but nothing refuses to run; the operator still has to make the machine quiet.
- **What the power budget means.** `rover_perception.yaml` describes `max_power_w: 15.0` as perception's share of the Orin NX envelope, but the gate compares it with total board power. Either the limit or the comment needs to change before a verdict on power means anything.
- **TensorRT minor version.** ONNX Runtime's docs do not say which 10.x minor 1.30.0 was built against (the table stops at 1.22). 10.16.1.11 loads and runs fp16 correctly on the 5090.

## v1.0 release work (branch `v1-release`)

- `b2f report <case_study_dir> --out <file.html>`: one self-contained page from the committed runs, driven by `report.yaml` in the case study folder (which run sets are baselines, references, profiles). Generated output committed under `case_studies/01_perception_detector/report/` (`index.html`, `headline.svg`). The page footer records the generator's commit; because the generated file is itself tracked, the tree reads "dirty" at generation time whenever the report changed, so that flag on the footer is expected.
- README rewritten as the entry point; `bringup/clean_clone/TRANSCRIPT.md` records the clean-clone test (two passes, second clean) and the two README fixes it forced.
- Identifiers scrubbed in every tracked file and in the full history (rewritten before publication). Commit IDs recorded in run files, manifests and the findings are the original ones; `docs/COMMIT_MAP.md` maps them to this history.
- Tagging v1.0 and the GitHub Release (with the report attached) are the owner's call after review.

## Future direction (after case study 01)

Logged by the owner's instruction; not to be built until case study 01 is done.

**Model fit matrix (`b2f evaluate`)**

- Goal: one command that takes a newly released model and answers whether it can run on our robots and at what cost.
- Input: a model plus a target list of boards, precisions and a budget.
- Steps: export to ONNX where supported, build fp32/fp16/int8 variants, run each on every available board at the robot's real request rate, check accuracy against the fp32 reference.
- Output: a board x precision matrix with response p95/p99 at the robot's rate, accuracy delta, memory footprint, power, and the GO / NO-GO verdict per cell. Rendered by `b2f report`.
- Prerequisites, in order: phase 2 accuracy harness and quantization ladder validated on YOLOX; per-process memory measurement on Jetson (currently only NVML on the 5090); per-frame samples in the RunReport schema (needed by the report anyway).
- Out of scope for this path: models that don't go through ONNX (vLLM, TensorRT-LLM served VLA/LLM models). Note it as a separate track if needed later.

## Ignored on purpose

- An external, billed, multi-agent cloud code review was offered by the development tooling during this work. The owner chose not to run it. Nothing in this repo has been through an external review.

## Pull request

`hardware-bringup` -> `master`: https://github.com/obiedeh/bench2field/pull/1, opened so CI runs. Merging is the owner's call. `master` is still the v0.1 commit until then.

## Commands to start phase 2

On the 5090 host, from `~/github/bench2field` (prefix everything with `env -u PYTHONPATH`); on the Jetsons, from the same path without the prefix.

```bash
# 1. Export the teacher (weights download once):
.venv/bin/python case_studies/01_perception_detector/export_yolox.py l
rsync -a models/yolox_l.onnx bench-thor:github/bench2field/models/ ; rsync -a models/yolox_l.onnx field-orin:github/bench2field/models/

# 2. First rung of the ladder, fp16 TensorRT, same sweep shape as the baseline. Copy
#    sweeps/phase1_baseline.yaml to sweeps/phase2_fp16.yaml with a trt_fp16 variant
#    (precision: fp16) alongside trt_fp32, and decide on --no-spin (see open questions).
.venv/bin/b2f sweep case_studies/01_perception_detector/sweeps/phase2_fp16.yaml --out-dir case_studies/01_perception_detector/runs/p2_fp16_5090

# 3. Accuracy for every variant from here on (methodology rule 9): needs the COCO val
#    subset and labelled rover frames; nothing for this exists in the repo yet.

# 4. Compare at the rover's tier, with the repeats:
.venv/bin/b2f retention "runs/.../trt_fp32_r*.json" "runs/.../trt_fp16_r*.json" "<field fp32>" "<field fp16>" --hz 30
```

Do not stop the Thor's `physical-ai-vllm` container; see the hardware table. Keep `PHASE1_FINDINGS.md` as the reference for what the frame costs around the model.
