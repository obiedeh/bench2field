# Case study 01: perception detector on the ROSMASTER rover

**Question:** which optimizations of the rover's object detector keep their speedup on the robot itself, and what costs the rest?

## Setup

| Item | Choice | Notes |
|---|---|---|
| Model | YOLO-family detector, one large (teacher) and one small (student) size | Check the licence of the chosen implementation before publishing weights |
| Accuracy set | Standard validation subset + frames captured from the rover camera | Label a few hundred rover frames; report both |
| Bench | RTX 5090 (development, Nsight Compute, Triton), Jetson AGX Thor (edge target) | |
| Field | ROSMASTER M3 Pro, Orin NX, with SLAM (lidar + camera + IMU) running | Same power mode for every run |
| ROCm | AMD Developer Cloud MI300X | Kernel port and MIGraphX runs |
| Budget | `configs/budgets/rover_perception.yaml` | Confirm the rail name and baseline mAP first |

## Phases

Ordered for the AMD application: 1, 3 and 4 first, then the rest.

### Phase 1: baseline and end-to-end profile
- [ ] Export the detector to ONNX; FP32 baseline with `b2f run` on Thor and the 5090.
- [ ] Profile the full pipeline (frame in, preprocess, inference, postprocess, ROS 2 publish) with Nsight Systems and the PyTorch profiler.
- [ ] Write down where the time goes. Expect preprocessing and host-device copies to matter as much as inference.

### Phase 2: optimization ladder (accuracy reported at every step)
- [ ] FP16 TensorRT
- [ ] INT8 PTQ with calibration on rover frames
- [ ] 2:4 structured sparsity, fine-tune, TensorRT sparse engine
- [ ] Distillation: large to small, trained on the 5090

### Phase 3: custom CUDA preprocessing kernel
- [ ] Fused resize + normalise + HWC→CHW + dtype cast, one pass (spec in `kernels/README.md`).
- [ ] Nsight Compute: achieved bandwidth against the device peak; explain the gap.
- [ ] Memory paths compared: zero-copy / unified memory on Jetson vs pinned host memory and CUDA streams on the 5090.

### Phase 4: ROCm port
- [ ] hipify the kernel; run and profile on MI300X with rocprof.
- [ ] Same ONNX model through ONNX Runtime MIGraphX; report in the same schema.
- [ ] Note every change the port needed.

### Phase 5: field
- [ ] `b2f record-load rover-slam` on the rover.
- [ ] Field runs of every variant; field retention per optimization.
- [ ] Replay on Thor; replay validity; single-stressor runs; gap attribution.
- [ ] Verdict for each variant against the budget.

### Phase 6: serving
- [ ] Triton on the 5090: dynamic batching, concurrency sweep with perf_analyzer.

## Write-up

Results table (variant × platform: accuracy, p50/p95/p99, miss rate, power), retention per optimization, attribution chart, verdicts, and a section on what the bench predicted versus what the robot did.
