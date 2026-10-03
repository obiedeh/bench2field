# Hardware bring-up evidence

Runs made while getting Bench2Field working on each machine. They use `tools/make_tiny_model.py`, not a real detector, so they validate the tooling and say nothing about any model. Every bring-up figure quoted in a commit message or in `HANDOFF.md` comes from a file here or from `tests/fixtures/`.

## rtx5090/ (host bench-5090, 2026-10-02)

RTX 5090, driver 580.178.04 (CUDA 13.0), Python 3.12.3, onnxruntime-gpu 1.30.0, TensorRT 10.16.1.11 (`tensorrt-cu13`), cuDNN 9.19.0.56, PyTorch 2.11.0+cu130, nvidia-ml-py 13.615.71. The full list is in `versions.txt`.

| File | What it is | Command |
|---|---|---|
| `run_cuda_fp32_30hz.json` | 30 s at 30 Hz, CUDA provider, idle bench. NVML validation. | `b2f run models/tiny_conv.onnx --provider cuda --tiers 30 --duration 30` |
| `gpu_stressor_50pct.json` | `GpuStressor` at a 50% target measured by NVML: 30 s open loop, then `calibrate()`, then 30 s more. | `python bringup/gpu_stressor_check.py <out.json>` |
| `run_trt_fp16_30hz.json` | 30 s at 30 Hz, TensorRT provider in fp16, idle bench. | `b2f run models/tiny_conv.onnx --provider tensorrt --precision fp16 --tiers 30 --duration 30` |
| `run_cuda_fp32_30hz_replay_gpu50.json` | The same 30 Hz run with `synth_gpu50.json` replayed (GPU stressor only). | `b2f run models/tiny_conv.onnx --provider cuda --replay bringup/synth_gpu50.json --only gpu --tiers 30 --duration 30` |

| `sweep_tiny/` | `b2f sweep` check: three alternating repeats of CUDA fp32 and TensorRT fp16, 10 s at 30 Hz each, with the manifest. | `b2f sweep configs/sweeps/bringup_tiny.yaml --out-dir bringup/rtx5090/sweep_tiny` |

`run_cuda_fp32_30hz.json` was recorded with cuDNN 9.27.0.42; installing PyTorch afterwards moved the environment to cuDNN 9.19.0.56, which the other two files used.

`synth_gpu50.json` is a hand-written profile with one target (`gpu_util_pct: 50`). It is not a field recording.

## thor/ (host `bench-thor`, 2026-10-02)

Jetson AGX Thor, L4T R38.4.0, driver 580.00 (CUDA 13.0), system TensorRT 10.13.3.9 and cuDNN 9.12.0.46, Python 3.12.3. Power mode `120W` (nvpmodel mode 1, the board's default), left as found. Package versions are in `pip_freeze.txt`.

The board was not idle: the `urban-edge-vllm` container was loaded (about 42 GB of RAM in use) and one CPU core sat near 28%. That is fine for checking the tooling and would not be for a baseline.

**ONNX Runtime wheel:** `onnxruntime-gpu==1.24.0` from the Jetson AI Lab index (`https://pypi.jetson-ai-lab.io/sbsa/cu130`), which has the CUDA and TensorRT providers. The PyPI `onnxruntime-gpu==1.30.0` aarch64 wheel was tried first and does not work on the Thor: it has no TensorRT provider, and its CUDA provider fails on the first Relu with `cudaErrorNoKernelImageForDevice`. So the Thor runs ONNX Runtime 1.24.0 and the 5090 runs 1.30.0.

| File | What it is |
|---|---|
| `tegrastats_idle.txt` | 20 lines at 500 ms, nothing of ours running. |
| `tegrastats_cuda_load.txt`, `tegrastats_trt_load.txt` | 20 lines each during the 30 Hz CUDA and TensorRT runs. |
| `nvpmodel.txt` | `nvpmodel -q` output. |
| `nvml_supported_calls.txt` | Which NVML queries the Thor answers. Memory info and clock info are not supported. |
| `run_cuda_fp32_30hz.json` | `b2f run models/tiny_conv.onnx --provider cuda --tiers 30 --duration 30` |
| `run_trt_fp16_30hz.json` | `b2f run models/tiny_conv.onnx --provider tensorrt --precision fp16 --tiers 30 --duration 30` |
| `run_trt_fp16_30hz_replay_cpu40_emc30.json` | The TensorRT run for 60 s with `synth_cpu40_emc30.json` replayed. |

What the captures show:

- **Total board power is the `VIN` rail** (reported as `power_board_w`). It reads 20 to 24 W here and is larger than `VDD_GPU`, `VDD_CPU_SOC_MSS` and `VIN_SYS_5V0` combined.
- **tegrastats on the Thor prints no `GR3D_FREQ` and no `EMC_FREQ`**, idle or under GPU load, with or without `--readall`. So there is no GPU-load or memory-controller channel to steer a stressor with.
- **Replay steering, CPU target 40%:** calibration froze the duty at 0.342 and measured 42.4% with the model idle; the tier then ran at a median of 41.9%. The memory-bandwidth stressor ran open-loop at a duty of 0.30 with nothing to measure it against.

`synth_cpu40_emc30.json` is a hand-written profile, not a field recording.

## orin/ (host `field-orin`, the ROSMASTER rover, 2026-10-02)

Jetson Orin NX, "Engineering Reference Developer Kit Super", L4T R36.4.7 (JetPack 6), driver 540.4.0 (CUDA 12.6), system TensorRT 10.7.0, **Python 3.10.12** (the case the `timezone.utc` fix was for). Power mode `MAXN_SUPER` (mode 0), left as found; `nvpmodel -q` also prints errors about CPU cores 6 and 7, which are offline in this mode, so six cores are online. Details in `versions.txt`.

The board had just been switched on and is the rover: its own services were running (`openclaw-gateway` was using more than one core while settling, plus the OLED script and desktop). It is the field machine, so that is expected. Disk was 93% full before the install (8.1 GB free after).

**ONNX Runtime wheel:** `onnxruntime-gpu==1.24.0` from `https://pypi.jetson-ai-lab.io/jp6/cu126`, the same version as the Thor. The system has no `python3-venv`, so the venv was created with `--without-pip` and pip bootstrapped into it.

**cuDNN that actually runs is 9.3.0, not the installed 9.11.0.98.** `ldconfig` resolves `libcudnn.so.9` to `/usr/local/cuda/targets/aarch64-linux/lib/`, which belongs to the `libcudnn9-cross-aarch64-cuda-12 9.3.0.75` package, and `cudnnGetVersion()` on the library onnxruntime loads says 9.3.0. Every run report records it. Not changed.

| File | What it is |
|---|---|
| `tegrastats_idle.txt`, `tegrastats_cuda_load.txt`, `tegrastats_trt_load.txt` | 20 lines each at 500 ms: nothing of ours running, then during the CUDA and TensorRT runs. |
| `nvpmodel.txt`, `device_tree_model.txt` | `nvpmodel -q` output (with its error lines) and the board model. |
| `nvml_supported_calls.txt` | NVML on the Orin answers only the device name and driver version. |
| `run_cuda_fp32_30hz.json` | `b2f run models/tiny_conv.onnx --provider cuda --tiers 30 --duration 30` |
| `run_trt_fp16_30hz.json` | `b2f run models/tiny_conv.onnx --provider tensorrt --precision fp16 --tiers 30 --duration 30` |
| `run_trt_fp16_30hz_replay_cpu40_emc30.json` | The TensorRT run for 60 s with `synth_cpu40_emc30.json` replayed. |

What the captures show:

- **Total board power is the `VDD_IN` rail** (reported as `power_board_w`): 5.4 W idle, 6 to 13 W in the runs, always above `VDD_CPU_GPU_CV` plus `VDD_SOC`. The rover budget's `power_channel: power_board_w` therefore resolves on the rover.
- **tegrastats on the Orin prints `GR3D_FREQ`** (GPU load, visible during the runs) **but no `EMC_FREQ`**, with or without `--readall`. The memory-bandwidth stressor is open-loop on the Orin too.
- **Replay steering, CPU target 40%:** calibration converged at a duty of only 0.118, measuring 41.7% with the model idle, because the rover's own services were already using CPU; the tier then ran at a median of 35.5% as those services settled. This is the calibrate-then-freeze design doing what it should on a machine whose background load is not ours, and a reminder that a field profile must be recorded with that load in its steady state.
- **NVML is useless on the Orin** (every telemetry query is NotSupported), so the sampler now reports itself unavailable there rather than returning empty samples.
