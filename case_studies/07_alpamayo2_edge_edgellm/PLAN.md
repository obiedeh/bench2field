# Case study 07: Alpamayo 2 Edge with TensorRT Edge-LLM on the Jetson AGX Thor

Written 2026-11-12. The replication runs (FP8, as NVIDIA's recipe is written) were
made the same day before this file; the budget and the protocol below were fixed
before the first optimization run (NVFP4). Changes after that are dated amendments at
the end; nothing above them is rewritten.

## Question

Case study 06 left Alpamayo 2 Edge at 0.33 s median per decision on the
jetson-agx-thor, in PyTorch with NVIDIA's acceleration, vision reuse, five Euler
steps, FP8 on the language model and MAXN. NVIDIA's deployment recipe for the model
(`NVlabs/alpamayo-recipes`, `recipes/alpamayo2edge_edgellm`, published 2026-11-10,
TensorRT Edge-LLM v0.11.1) reports about 220 ms on a DRIVE AGX Thor with FP8 engines
and a speculative drafter. Embedl's model card for the same model on the same Jetson
board (`embedl/Alpamayo2-Edge`, 2026-11-11) reports 91.9 ms median, 140.3 ms p90 and
12.45 J per decision with NVFP4 on the language model, MXFP8 on the vision encoder and
the action expert, caches and speculation; their FP16 Edge-LLM engines sat at 489 ms.

What does NVIDIA's recipe deliver on the Jetson board as written, what does each
further step buy, and how close does a reproducible open build get to Embedl's card?

## Budget (fixed before the NVFP4 run)

| Item | Limit | Why |
|---|---|---|
| Decision deadline | 0.1 s (10 Hz) | the planner's rate, as in case study 06 |
| Owner's target for this study | p95 under 0.25 s on the eight clips | halfway from 0.49 s (FP8 recipe) to the deadline |
| Chain of causation | identical text to the FP8 engines on every clip that writes one | a faster build that says something else is a different model |
| Trajectory | mean ADE between an optimized build and the FP8 engines below the spread among the FP8 candidates (NVIDIA's own acceptance for FP8 against PyTorch: 0.3 to 0.45 m) | |
| Power | report mean and peak board power and energy per decision for every run; no limit | MAXN is the owner's chosen mode |

## Setup

| Item | Value |
|---|---|
| Model | `nvidia/Alpamayo2-Edge`, Hugging Face revision aaaa32c6 (the recipe's pin; sha256 of both weight files equal to the 2026-11-10 copy) |
| Recipe | `NVlabs/alpamayo-recipes` 622f9d2, `recipes/alpamayo2edge_edgellm`; TensorRT Edge-LLM v0.11.1 (358de11c) |
| Board | `jetson-agx-thor` (label `bench-thor`): JetPack 7.1 (L4T r38.4), TensorRT 10.13.3.9, CUDA 13.0, Python 3.12.3, MAXN, CPU governor schedutil, GPU 1575 MHz |
| Host | `rtx-5090-workstation`: quantization, ONNX export, inputs |
| Clips | the eight of case studies 02 and 06 at t0 = 5.1 s with the derived navigation text of case study 06, plus NVIDIA's example clip ea7bbd31 at 4.0 s, "Turn right in 30m" |
| Calibration | FP8 and NVFP4 scales calibrated on the example clip's 22 frames, as the recipe does |

## Protocol

- **Replication first:** the recipe as written (FP8 language model, vision encoder and
  action expert, FP8 K/V cache, reduced 8192-row output vocabulary, greedy decoding
  because the DFlash2 drafter is not released), with the two Jetson adaptations the
  board needs (JetPack TensorRT instead of a DRIVE OS SDK, nvcc pinned to CUDA 13.0).
- **Timing:** `tools/thor_stream.py` loads the engines once and runs decisions back to
  back: 180 on the example clip (Embedl's method: one clip, 180 decisions, inputs on
  the GPU) and 20 on each of the nine clips. Rolling mode drops the six newest frames
  from the vision cache before every decision, the recipe's `--rolling` on a loop;
  cold mode encodes all 22. Stage times by stream synchronization at each boundary,
  as the recipe's runner does. One untimed warm-up per clip.
- **Power:** tegrastats at 100 ms, board input rail (VIN); energy per decision is the
  integral over the timed block divided by the decisions; idle sampled 5 s before.
- **Candidates:** six trajectory candidates (the recipe's default, Embedl's minADE6)
  and one (case study 06 measured one sample), each with its own action engine.
- **Accuracy:** the chain-of-causation text per clip is compared across builds; the
  trajectories are decoded with the recipe's `host/compare.py` and compared across
  builds and against the PyTorch BF16 waypoints recorded in case study 06 for the
  same clips.
- **Each change is one run set**, kept, so the table shows what each step bought.

## Steps planned, in order

1. FP8 recipe as written, 6 and 1 candidates, rolling and cold.
2. NVFP4 language model (W4A16; W4A4 if TensorRT 10.13 builds it), vision and expert
   unchanged.
3. MXFP8 or fewer steps on the action expert.
4. Edge-LLM media and K/V reuse across decisions.
5. Speculative decoding only if a DFlash2 drafter for the Edge model becomes available
   or is trained with NVIDIA's SpecForge recipe; out of scope for this study otherwise.

## Not measured here

Closed-loop driving quality, the AlpaGym or minADE6 benchmark of the model card, the
DRIVE AGX Thor itself, and Embedl's containers (not run; their card is quoted).
