# Handoff

State of the hardware bring-up and case study 01 phase 1. Work is on the `hardware-bringup` branch; `master` is still the v0.1 commit. Nothing has been pushed and no GitHub repo exists yet.

Last updated 2026-10-02, part-way through: step 1 is done except TensorRT, steps 2 and 4 have not started.

## Done

**Fixes requested before bring-up** (one commit each, with tests)

| Commit | Change |
|---|---|
| `02c3ded` | Python 3.10: `timezone.utc` instead of `datetime.UTC`. Suite verified under CPython 3.10.20. |
| `2f685d9` | `.gitignore`: `runs/` anchored to the repo root, so `case_studies/*/runs/` can be committed. |
| `9b60aaf` | Schema 1.1: response-time stats per tier (`TierResult.response`), `--drop-late` with `dropped` counted apart from `deadline_misses`. Response p95 is the default stat for retention, attribution and validity. |
| `c7c5d1c` | Replay calibrates stressor duty with the model idle, then freezes it. Result recorded as `platform.replay_calibration`. |
| `3dedb45` | Loud `TelemetryWarning` when a sampler is missing, reason recorded as `platform.telemetry_missing`; samplers closed after a run. |
| `16c462b` | GPU stressor in its own spawned process with its own CUDA context; all stressors use spawn. |

**Step 1, RTX 5090** (evidence in `bringup/rtx5090/`)

- Environment: `.venv`, Python 3.12.3, `pip install -e ".[dev,nvml]"`, `onnxruntime-gpu[cuda,cudnn]==1.30.0`, `onnx`, and `torch==2.11.0+cu130` for the GPU stressor.
- ONNX Runtime wheel: 1.30.0 is the current PyPI release and is built for CUDA 13.0 and cuDNN 9, which is what driver 580.178.04 supports.
- 66 tests pass in the venv, including three that only run on real hardware (CUDA Conv, NVML channels, GPU stressor process).
- NVML sampler validated: all six channels populated and plausible in `run_cuda_fp32_30hz.json`.
- GPU stressor at a 50% target, open loop: NVML measured a median of 50% (range 49 to 51) over 30 s, in `gpu_stressor_50pct.json`.

**Step 3, parts 1 and 2**: `LICENSE` (Apache-2.0), `CONTRIBUTING.md`, and `.github/workflows/ci.yml` (pytest on CPU, Python 3.10 and 3.12).

## What broke on real hardware, and the fix

| Problem | Fix |
|---|---|
| CUDA provider loaded, then failed on the first Conv: `dlopen failed for libcudnn.so`. The pip-installed cuDNN is under `site-packages/nvidia/`, off the loader path. | `e46d27f`: the backend calls `onnxruntime.preload_dlls()` for CUDA and TensorRT. |
| NVML kept reporting the flat-out warmup for about 3 s into the tier, so a 70 W tier was summarised with a 354 W peak. Real capture: `tests/fixtures/nvml_5090_after_flatout_warmup.json`. | `e468e64`: warmup runs at the tier's own rate. |
| With no TensorRT libraries, `--provider tensorrt` logged an error, fell back to CPU and exited 0. | `8440572`: the backend raises if the active provider is not the one requested. |
| This host exports a ROS 2 `PYTHONPATH`, which leaks ROS's Python packages into any venv. | Not a code change: run with `env -u PYTHONPATH`. |

## Blocked, waiting on the owner

1. **TensorRT (step 1.4).** The TensorRT provider in onnxruntime-gpu 1.30.0 needs `libnvinfer.so.10` and `libnvonnxparser.so.10`, and neither is installed. Proposed: `tensorrt-cu13==10.16.1.11` from PyPI into `.venv`. Needs the owner's OK.
2. **Thor (step 2).** SSH access, or the owner runs the commands.
3. **Detector choice (step 4.1)** and **creating/pushing the GitHub repo (step 3.3)**.

## Deferred by decision

- **Thermal hold** (fix before phase 5). `--only thermal` soaks and then stops heating, so the device cools during the tiers. On a discrete GPU the soak heats the CPU but reads GPU temperature, so it would run to its 600 s timeout.
- **`b2f sweep`** (build at the start of step 4, before the baseline runs): repeats, A/B alternation, median and spread, and checks that power mode and deadline match across runs compared by retention.
- **`power_board_w`**: map the Jetson total-power rail once real Thor tegrastats output is captured (step 2).

## Open questions

- **GPU stressor steering on NVML.** NVML does report `gpu_util_pct`, so on the 5090 replay calibration steers the GPU stressor closed-loop; it is not open-loop there. In the measurement the open-loop duty already landed on target, so calibration changed nothing. Decide whether discrete GPUs should be forced open-loop.
- **Verdict percentile.** The rover budget sets `p99_ms`, so the verdict gates response p99. `p95_ms` is available as a budget key but the budget file was not changed.
- **Miss gate and drops.** The verdict's miss gate counts late plus dropped frames against `max_miss_rate`, so `--drop-late` cannot turn failures into passes.
- **Exact TensorRT minor version** onnxruntime 1.30.0 was built against is not published in its docs (the table stops at 1.22). The provider links the `.so.10` major, so any 10.x should load; to be confirmed by the fp16 run.

## Commands to start phase 2

Pending: phase 1 has not been run.
