# Phase 2: alternated pipeline profiles on the Orin NX, FP32 and FP16

**Speed only, accuracy not yet measured.** Not for the findings or the README.

Six profiles of the same 480p rover frame set (`b39dce66…`), run back to back in the order fp32, fp16, fp32, fp16, fp32, fp16 (`<precision>_r<repeat>.json`), TensorRT, `--no-spin`, rover stack off, `MAXN_SUPER`, checkout `2ff1e663` (`--expect-commit`). Purpose: establish how much the host-side stages vary between runs and whether that variation follows CPU frequency.

Each file records, besides the stage timings: `cpu_freq` (governor and limits per core, per-core p50/min/max frequency during the timed frames, the cross-core mean as a time series) and `per_frame` (start time, host-stage and inference time per frame, on the same clock as that series). The governor (`schedutil`) and the clocks were recorded, not changed.
