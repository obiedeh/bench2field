# Hardware bring-up evidence

Runs made while getting Bench2Field working on each machine. They use `tools/make_tiny_model.py`, not a real detector, so they validate the tooling and say nothing about any model. Every bring-up figure quoted in a commit message or in `HANDOFF.md` comes from a file here or from `tests/fixtures/`.

## rtx5090/ (host `bench-5090`, 2026-10-02)

RTX 5090, driver 580.178.04 (CUDA 13.0), Python 3.12.3, onnxruntime-gpu 1.30.0, PyTorch 2.11.0+cu130, nvidia-ml-py 13.615.71.

| File | What it is | Command |
|---|---|---|
| `run_cuda_fp32_30hz.json` | 30 s at 30 Hz, CUDA provider, idle bench. NVML validation. | `b2f run models/tiny_conv.onnx --provider cuda --tiers 30 --duration 30` |
| `gpu_stressor_50pct.json` | `GpuStressor` at a 50% target measured by NVML: 30 s open loop, then `calibrate()`, then 30 s more. | `python bringup/gpu_stressor_check.py <out.json>` |
| `run_cuda_fp32_30hz_replay_gpu50.json` | The same 30 Hz run with `synth_gpu50.json` replayed (GPU stressor only). | `b2f run models/tiny_conv.onnx --provider cuda --replay bringup/synth_gpu50.json --only gpu --tiers 30 --duration 30` |

`run_cuda_fp32_30hz.json` was recorded with cuDNN 9.27.0.42; installing PyTorch afterwards moved the environment to cuDNN 9.19.0.56, which the other two files used.

`synth_gpu50.json` is a hand-written profile with one target (`gpu_util_pct: 50`). It is not a field recording.

## thor/

Pending.
