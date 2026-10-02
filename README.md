# Bench2Field

**How much of a model optimization survives contact with the robot, and why.**

An EmbodiedEdge Labs project.

Quantize a perception model to INT8 and the benchmark says it is three times faster. Put it on the robot, next to SLAM, sensor drivers and a warm enclosure, and some of that speedup disappears. Most published numbers come from an idle device, so nobody says how much.

Bench2Field measures that loss directly. Every optimization is run twice, once on the idle bench and once in the field, and the result is reported as **field retention**: the share of the bench gain that is still there on the robot. Then it works out where the rest went.

## What it measures

| Metric | Question it answers |
|---|---|
| **Field retention** | Of the speedup this optimization showed on the bench, how much is left on the robot? `(S_field − 1) / (S_bench − 1)` |
| **Gap attribution** | Of the slowdown from bench to field, how much is thermal, CPU contention, memory bandwidth, GPU sharing, and how much is interaction between them? |
| **Replay validity** | Does a recorded field load, replayed on the bench, reproduce field latency closely enough to stand in for the robot? |
| **Readiness verdict** | GO / NO-GO / INCOMPLETE against a deployment budget written before testing: loop deadline, miss rate, power, temperature, accuracy floor. |

Latency is measured open-loop at fixed request rates, the way camera frames arrive. Each tier reports response time (arrival to completion, queueing included) and service time (inference alone); comparisons use response p95. A frame misses its deadline when it finishes late relative to when it arrived, and `--drop-late` skips frames that are already stale, counting them separately.

## How it works

1. **Bench, idle.** Run the baseline and each optimized variant with no background load. Telemetry (power rails, temperatures, GPU and memory-controller load) is sampled throughout.
2. **Record the field load.** On the robot, with everything running except the model under test, record a load profile: CPU, GPU and memory-controller utilisation and the operating temperature.
3. **Field.** Run the same variants on the robot.
4. **Replay.** Replay the load profile on the bench, first all stressors together (checked against the field run with replay validity), then one stressor at a time for attribution.
5. **Report.** Field retention per optimization, gap attribution per variant, and a readiness verdict against the budget.

Once a profile passes replay validity, new variants can be screened on the bench without driving the robot for every test.

## Quick start

```bash
pip install -e ".[ort,dev]"          # use onnxruntime-gpu, the Jetson wheel, or a ROCm build as needed
pytest

# idle bench run at 10, 30 and 100 Hz
b2f run models/detector_fp32.onnx --provider tensorrt --precision fp32 --tiers 10,30,100 --out runs/bench_fp32.json
b2f run models/detector_fp32.onnx --provider tensorrt --precision int8 --technique ptq --out runs/bench_int8.json

# on the robot, without the model running: record the background load
b2f record-load rover-slam --duration 300

# on the robot: the same two variants
b2f run models/detector_fp32.onnx --provider tensorrt --precision fp32 --environment field --out runs/field_fp32.json
b2f run models/detector_fp32.onnx --provider tensorrt --precision int8 --technique ptq --environment field --out runs/field_int8.json

b2f retention runs/bench_fp32.json runs/bench_int8.json runs/field_fp32.json runs/field_int8.json --hz 30

# back on the bench: replay the field load, check it, then attribute
b2f run models/detector_fp32.onnx --provider tensorrt --replay profiles/rover-slam.json --out runs/replay_all.json
b2f validity runs/replay_all.json runs/field_fp32.json --hz 30
b2f run models/detector_fp32.onnx --provider tensorrt --replay profiles/rover-slam.json --only thermal --out runs/r_thermal.json
b2f run models/detector_fp32.onnx --provider tensorrt --replay profiles/rover-slam.json --only membw   --out runs/r_membw.json
b2f attribute runs/bench_fp32.json runs/field_fp32.json --hz 30 --stressor thermal=runs/r_thermal.json --stressor membw=runs/r_membw.json

b2f verdict runs/field_int8.json configs/budgets/rover_perception.yaml
```

## Development setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,tools,nvml]"     # add gpu-stress for the GPU stressor (PyTorch)
.venv/bin/pytest
```

Install the ONNX Runtime build that matches the machine instead of the `ort` extra: `onnxruntime-gpu[cuda,cudnn]` on a discrete NVIDIA GPU (plus `tensorrt-cu13` for the TensorRT provider), the Jetson wheel on a Jetson. The exact versions validated on each machine are listed under `bringup/`.

**If the shell has ROS 2 sourced, unset `PYTHONPATH` first.** ROS exports a `PYTHONPATH` that puts its own Python packages ahead of the venv's, and a venv does not override it. Prefix every command with `env -u PYTHONPATH`, for example `env -u PYTHONPATH .venv/bin/pytest`, or run from a shell that has not sourced ROS.

Tests that need onnxruntime, a GPU, NVML or PyTorch skip on machines without them.

## Platforms

One report schema across vendors, so results compare directly.

| Platform | Telemetry | Backend paths |
|---|---|---|
| NVIDIA Jetson AGX Thor, Orin NX | tegrastats (power rails, temps, GPU, EMC) | ONNX Runtime: CUDA, TensorRT |
| NVIDIA RTX (discrete) | NVML | ONNX Runtime: CUDA, TensorRT; Triton (planned) |
| AMD Instinct (MI300X via AMD Developer Cloud) | rocm-smi | ONNX Runtime: MIGraphX, ROCm |

The rocm-smi parser is written against documented output but not yet validated on hardware.

## Layout

```
src/bench2field/
  schema.py        report contract (one JSON per run; add fields, never break them)
  runner.py        open-loop tiers, deadline accounting, cooldown gating
  metrics.py       field retention, gap attribution, replay validity
  verdict.py       readiness gates against a deployment budget
  telemetry/       tegrastats, NVML, rocm-smi samplers
  backends/        ONNX Runtime (CPU, CUDA, TensorRT, MIGraphX, ROCm)
  loadreplay/      record a field-load profile; replay it as CPU, memory-bandwidth, GPU and thermal stressors
configs/budgets/   deployment budgets
case_studies/      one folder per model studied
kernels/           custom kernels (CUDA, HIP ports)
docs/METHODOLOGY.md
```

## Case studies

| # | Model | Focus | Status |
|---|---|---|---|
| 01 | Perception detector on the ROSMASTER rover | FP16 / INT8 / 2:4 pruning / distillation; custom CUDA preprocessing kernel with a HIP port; Triton serving | Planned |
| 02 | Vision-language model (Cosmos-Reason2-2B, Gemma) | INT4 / FP8; KV-cache memory under load | Planned |
| 03 | Manipulation policy (ACT, GR00T) | Inference latency against the control-loop deadline | Planned |
| 04 | Edge intrusion-detection models (from jetson-edge-ai-security) | ONNX Runtime thread tuning on Thor | Measured there; to be imported |

## Related work

- [ros2_benchmark](https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark) measures throughput and latency of ROS 2 graphs as they are, from rosbag or live data.
- [RobotPerf](https://arxiv.org/abs/2309.09212) is a vendor-neutral suite for robotics computing performance across ROS 2 pipelines.
- [Beyond Benchmarks](https://arxiv.org/abs/2606.17241) reports a 20–30% drop from static benchmark to streaming deployment for a roadside perception model on a Jetson Orin Nano.

Bench2Field asks a narrower question none of these answer: whether a specific optimization's gain survives deployment, how much of it, and what took the rest. It uses ONNX Runtime and the vendors' own profilers underneath rather than replacing them.

## Status

v0.1: core runner, report contract, metrics, telemetry, load record/replay and verdicts, with tests. No case-study results yet; numbers will be published only from committed runs on named hardware.
