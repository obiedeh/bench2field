# Phase 2, step 1: FP16 against FP32 on the rover's Orin NX

**Speed only, accuracy not yet measured.** Nothing here is a finding until the accuracy harness exists; do not quote these numbers in the findings or the README.

- Sweep: `sweeps/phase2_orin_fp16.yaml`, TensorRT fp32 and fp16 alternating, 10/26/30/100 Hz, three repeats each, `--no-spin`, rover stack off, `MAXN_SUPER`, checkout `41fb1483` (`--expect-commit`).
- Same ONNX file for both variants (`models/yolox_s.onnx`, SHA-256 `00b70c44…`); FP16 is TensorRT's `trt_fp16_enable`. Each report records the engine build options as `platform.provider_options` and what the run asked for as `platform.provider_options_requested`.
- Pipeline profiles taken right after the sweep, FP16 first and FP32 immediately after, on the same 480p rover frame set (`b39dce66…`): `../phase2_profile_orin_rover_frames_480p_fp16.json` and `..._fp32.json`.
