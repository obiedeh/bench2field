# Handoff

State of the hardware bring-up and case study 01 phase 1. Work is on the `hardware-bringup` branch; `master` is still the v0.1 commit and has not been merged into or pushed.

The GitHub repo exists: `github.com/obiedeh/bench2field`, **private**, with only `hardware-bringup` pushed (the owner's choice). CI has not run yet, because the workflow triggers on `master`/`main` and on pull requests. Before making the repo public, note that this file and `bringup/` name the owner's hosts and a LAN address.

**Before the repo goes public:** scrub hostnames (`bench-5090`, `bench-thor`, `field-orin`) and the LAN address (`192.0.2.10`) from this file and from `bringup/` (the README, `versions.txt` files and the `host` field in every run JSON). Not done; logged here so it is not forgotten.

Last updated 2026-10-02. Steps 1 and 2 (5090, Thor and Orin) are done. `b2f sweep` is built. Step 4: YOLOX chosen, YOLOX-s exported and on all three machines; the baseline sweeps have not been run (they take about 20 minutes per machine and need the owner's OK).

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

**Step 3**: `LICENSE` (Apache-2.0), `CONTRIBUTING.md`, `.github/workflows/ci.yml` (pytest on CPU, Python 3.10 and 3.12), and the private GitHub repo.

**Step 4.1 and 4.2: detector and export**

- YOLOX (Apache-2.0), pinned to commit `6ddff482`. YOLOX-s is the student, YOLOX-l the teacher (not yet exported; `export_yolox.py l` does it).
- `models/yolox_s.onnx`: static 1x3x640x640, opset 11, decoding in the graph, NMS outside. Provenance (weights and ONNX SHA-256, versions, PyTorch agreement check) in `case_studies/01_perception_detector/exports.json`; the choices and why in that folder's README.
- The same file is on the 5090, the Thor and the Orin (`models/`), same hash on all three.
- `case_studies/01_perception_detector/sweeps/phase1_baseline.yaml` is the baseline sweep, ready to run.

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

**Every command on the 5090 host must be run as `env -u PYTHONPATH .venv/bin/<command>`.** The host's shell sources ROS 2 Jazzy, whose `PYTHONPATH` takes precedence over the venv; without unsetting it, pytest picks up ROS's plugins and fails on import. The Thor's shell does not set `PYTHONPATH`, so plain `.venv/bin/<command>` works there.

## Blocked, waiting on the owner

1. **Phase 1 baseline sweeps (step 4.3)**: about 20 minutes per machine, so they need the owner's OK before starting. On the Thor the `urban-edge-vllm` container may be stopped for them (owner's permission) and must be restarted afterwards; when last checked the container up was `physical-ai-vllm`, which that permission does not cover.
2. **The end-to-end profile and `nsys` capture (step 4.4)** follow the baselines.

## Deferred by decision

- **Thermal hold** (fix before phase 5). `--only thermal` soaks and then stops heating, so the device cools during the tiers. On a discrete GPU the soak heats the CPU but reads GPU temperature, so it would run to its 600 s timeout.

- **Orin replay drift (explain before trusting replay validity, phase 5).** On the Orin, the CPU stressor was calibrated to a 40% target with the model idle: duty frozen at 0.118, 41.7% measured. During the 60 s tier that followed, CPU utilisation had a median of 35.5% with the duty unchanged. The rover's own services were still settling after boot, so the background load the calibration absorbed was not steady. Until this is explained (and the recording rule "steady-state background load" is enforced or checked), a replay's validity number should not be trusted on its own. No fix now.

## Open questions

- **ONNX Runtime versions differ between machines.** The 5090 runs 1.30.0 (TensorRT 10.16.1.11) and the Thor runs 1.24.0 (system TensorRT 10.13.3.9), because no 1.30.0 wheel works on the Thor. Latency on the two is not a same-runtime comparison. Options: accept and record it, pin the 5090 to 1.24.0, or build 1.30.0 from source on the Thor (well over 15 minutes).
- **Memory-controller load is not measurable on either Jetson** (no `EMC_FREQ` on the Thor or the Orin NX), so the memory-bandwidth stressor is always open-loop there. GPU load: the Orin reports `GR3D_FREQ`; the Thor does not, though NVML on the Thor does report GPU utilisation, so the Jetson sampler could take that one channel from NVML. Not built; needs a decision.
- **Bench machines were not idle.** The Thor had the `urban-edge-vllm` container loaded during bring-up, and an Ollama evaluation job started on the 5090 part-way through (one contaminated run was discarded). Reports now record the containers and busiest processes at run start, but nothing refuses to run; the operator still has to make the machine quiet.
- **What the power budget means.** `rover_perception.yaml` describes `max_power_w: 15.0` as perception's share of the Orin NX envelope, but the gate compares it with total board power. Either the limit or the comment needs to change before a verdict on power means anything.
- **TensorRT minor version.** ONNX Runtime's docs do not say which 10.x minor 1.30.0 was built against (the table stops at 1.22). 10.16.1.11 loads and runs fp16 correctly on the 5090.

## Ignored on purpose

- An external automated review was offered during this session; the owner chose not to run it. Nothing in this repo has been through one.

## Commands to start phase 2

Pending: phase 1 has not been run.
