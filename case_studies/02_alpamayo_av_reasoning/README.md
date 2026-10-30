# Case study 02: Alpamayo 1, an AV reasoning model, on the bench and the edge

**Question:** what does one driving decision from NVIDIA's open Alpamayo 1 model
cost on the hardware in this lab, in time, memory and power, and how much of the
bench's performance survives on a Jetson?

Alpamayo 1 (`nvidia/Alpamayo-R1-10B`, 10B parameters, weights under OpenMDW-1.1,
inference code Apache-2.0) reads four cameras over 0.4 s plus the ego vehicle's
recent motion, writes a chain-of-causation reasoning trace, and samples a 6.4 s
trajectory at 10 Hz. Every number here comes from a committed JSON record on a
named board, as in case study 01.

**Status: phases 2 to 4 done, the 5090 is measured and profiled, the Thor is measured.** Results below come from the
committed JSON only; a cell says "pending" until the run that produces it is committed.

## Boards

| Board | Role | Target for | State |
|---|---|---|---|
| RTX 5090 (`bench-5090`) | bench | full-precision run, profile, plots | measured, sdpa and flash-attn (phase 2); profiled (phase 3) |
| Jetson AGX Thor (`bench-thor`) | edge bench | same clips, with power | measured, sdpa, board power (phase 4); JetPack, not DriveOS |
| Jetson Orin NX 8 GB (`field-orin`) | field | nothing | does not fit; measured facts in [ORIN_NX.md](ORIN_NX.md) |

## Results

Measured values only, from the committed JSON under `runs/`. The table is the
output of `plot_results.py --table` (phase 3) and is regenerated, not edited,
when a run is added. Protocol: eight clips, one warm-up decision excluded, three
decisions per clip, upstream sampling defaults, seed 42 before each decision.
Power on the 5090 is the GPU rail from NVML; power on the Thor is the whole
board (`VIN` from tegrastats). The two power rows are therefore not like for
like; the Thor figure includes CPU, memory and carrier board.

| Metric | RTX 5090, flash-attn | RTX 5090, sdpa | Jetson AGX Thor, sdpa |
|---|---|---|---|
| Model load time | 8.3 s | 15.7 s | 25.9 s |
| Latency per decision, median | 0.93 s | 0.89 s | 4.38 s |
| Latency per decision, p95 | 1.24 s | 1.16 s | 5.88 s |
| Time to first token, median | 0.37 s | 0.38 s | 1.84 s |
| Decode rate, median | 50.9 tok/s | 60.0 tok/s | 9.7 tok/s |
| Reasoning tokens, median | 18 | 18 | 18 |
| Decisions per minute | 64.3 | 67.8 | 13.7 |
| GPU memory after load | 20.6 GiB | 20.6 GiB | 20.6 GiB |
| Peak GPU memory during a decision | 21.5 GiB | 21.5 GiB | 21.5 GiB |
| Power during a decision, median | 359 W | 377 W | 64 W |
| Energy per decision, median | 323 J | 316 J | 290 J |
| Decisions | 24 (8 clips x 3) | 24 (8 clips x 3) | 24 (8 clips x 3) |


The flash-attn column is flash-attn 2.8.3 on the VLM with sdpa kept on the
diffusion expert (PLAN.md, Amendments). On this GPU, at batch size one and
these sequence lengths, it is slower than sdpa in decode (50.9 against 60.0
tokens per second) and equal in time to first token, so sdpa stays the
baseline and is the variant the Thor is compared against.

## Where the time goes (phase 3, 5090, sdpa)

One decision on clip `030c760c` under `torch.profiler`, after one warm-up,
from `runs/profile_bench-5090_sdpa.json`. The profiler adds overhead: this
decision took 1.01 s against 0.94 s for the same clip in the
unprofiled run, so the split below is read as shares, not as the latency.

| Stage | Device time | Share of the decision |
|---|---|---|
| Vision tower (inside the rollout) | 117 ms | 12 % |
| Language rollout: prefill and 22 decode steps | 784 ms | 78 % |
| Trajectory sampling (diffusion expert) | 223 ms | 22 % |
| Whole decision | 1009 ms | 100 % |

- The first token arrives at 0.39 s; after that the decode runs at one token every
  19 ms (53 tokens per second under the profiler). With a median
  of 18 reasoning tokens, decode is the largest single block of a decision,
  and a longer trace costs 19 ms per extra token (`report/tokens_vs_latency.svg`).
- The kernel table says what that decode is: the two largest kernels are a
  cutlass bf16 tensor-op GEMM (23 % of kernel
  time, prefill and the diffusion head) and a bf16 GEMV (17 %
  plus 8 % in a second instance), which is the
  batch-one decode streaming the 8B weights once per token. Attention is third
  (memory-efficient attention kernel, 9 %). sdpa on
  this stack dispatches to torch's own flash kernel for the prefill and to the
  memory-efficient kernel elsewhere.
- The kernel names carry `sm80` and `cutlass_80`: torch 2.8.0+cu128 runs its
  Ampere-generation kernels on this Blackwell GPU. There are no sm_120 specific
  GEMM or attention kernels in this build, which is one reason the flash-attn
  2.8.3 wheel (also built for older architectures) does not win either.
- The operator table overlaps by construction (`aten::linear` contains
  `aten::matmul` contains `aten::mm`), so its shares are of the table, not of
  the decision. The kernel table does not overlap.
- Plots under `report/`: `stage_split` (medians over all decisions, both
  variants), `latency_per_clip`, `tokens_vs_latency`, `memory`,
  `power_per_decision`, and `profile_timeline` (NVML power and utilisation
  around the profiled decision). SVG with a PNG next to each.

## 5090 against Thor (phase 4)

Same model snapshot, same inference code, same eight clips, same protocol,
same seed, sdpa on both. The software differs where the Thor leaves no
choice: the 5090 runs the upstream pin, torch 2.8.0+cu128 on CUDA 12.8; the
Thor runs torch 2.11.0+cu130 on CUDA 13.0 (JetPack 7.1), because 2.8.0 does
not exist for that stack. transformers 4.57.1 and physical_ai_av 0.2.3 are
the same on both. The Thor is a Jetson AGX Thor developer kit on JetPack,
not a DRIVE AGX Thor on DriveOS. From `runs/decisions_bench-5090_sdpa.json`
and `runs/decisions_bench-thor_sdpa.json`:

| Stage or metric | RTX 5090, sdpa | Jetson AGX Thor, sdpa | Thor as a share of the 5090 |
|---|---|---|---|
| Latency per decision, median | 0.89 s | 4.38 s | 20 % |
| Time to first token, median | 0.38 s | 1.84 s | 21 % |
| Rollout (prefill and decode), median | 0.67 s | 3.53 s | 19 % |
| Trajectory sampling, median | 0.21 s | 0.84 s | 25 % |
| Decode rate, median | 60.0 tok/s | 9.7 tok/s | 16 % |
| Decisions per minute | 67.8 | 13.7 | 20 % |

- **Retention is about one fifth.** Every stage of the decision runs at 16 to
  25 percent of the 5090's speed on the Thor, so the shape of the decision is
  the same on both boards: the rollout dominates, the first token is the
  largest single wait, trajectory sampling is the smallest block
  (`report/stage_split.svg`). Nothing falls off a cliff; the Thor is slower
  evenly.
- **Memory is identical.** 20.6 GiB after load, 21.5 GiB peak on both. The
  Thor's 122 GB of unified memory makes this comfortable; the Orin NX's 8 GB
  does not (see ORIN_NX.md).
- **Power is not like for like, and the energy still lands close.** The 5090
  figure is the GPU rail alone from NVML: 377 W median during a decision.
  The Thor figure is the whole board from tegrastats `VIN`: 64 W median
  during a decision, 119 W peak sample, against 21 W mean before
  the first decision (model load and clip streaming, not idle). Energy per
  decision, mean power times latency, is 316 J on the 5090's GPU rail and
  290 J for the Thor's whole board. The Thor spends five times longer
  at one sixth of the power.
- **The traces agree in meaning, not in wording.** With the same seed the two
  boards write different token counts for the same clip (32 against 35 on
  `06b483cf`); the three committed traces per board say the same thing in
  different words. Different kernels sample differently; upstream's note on
  nondeterminism says as much.
- **What a 10 Hz planner would see.** A median decision of 4.4 s on the Thor
  is one decision every 44 camera frames at 10 Hz, before any batching,
  quantization, CUDA graphs or a smaller trace budget, none of which this
  study tried. The 5090's 0.89 s is one decision per nine frames. Neither
  board runs this model at frame rate as shipped; that is the measured
  starting point, not a verdict on the model.

## Phases

1. Review and environment: the venv, the pins, the friction log, this scaffold. Done.
2. 5090 run on a handful of PhysicalAI-AV clips: load time, latency per decision, peak memory, throughput, sample reasoning traces. Done.
3. Profile: where the time goes between the vision encoder, the language rollout and the diffusion trajectory head, with plots. Done.
4. Thor, same clips, power from tegrastats, compared with the 5090. Done.
5. Orin NX: why a quantized run is not feasible on this module. Done: [ORIN_NX.md](ORIN_NX.md).

## Layout

```text
README.md            this file
PLAN.md              question, protocol fixed before the first run, phases
SETUP.md             exact setup per board
FRICTION_LOG.md      what broke, how it was found, what fixed it, with times
ORIN_NX.md           why the field computer is not a target, with measured facts
setup_env.sh         builds .venv on the 5090 host
capture_env.py       writes runs/env_<board>.json
requirements.txt     the pins, for reading; setup_env.sh installs them in the right order
runs/                committed records only (environment, decisions, profile)
report/              plots written by plot_results.py (SVG and PNG)
profile_decision.py  torch.profiler on one decision, stage labels, kernel and op tables
plot_results.py      the plots and the README results table, from runs/ only
traces/              three reasoning traces per board and variant, verbatim
clips.yaml           the eight clip ids, chosen by rule before the first run
run_decisions.py     the decision runner (one JSON per board and variant)
.venv/, weights/, clips/, data/   ignored
```

## Sources

- [NVlabs/alpamayo](https://github.com/NVlabs/alpamayo), pinned at commit `11a0e01`
- [Model card](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [PhysicalAI-Autonomous-Vehicles dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) (gated; terms accepted by the owner)
- Alpamayo 1.5 and the recipes hub are newer; this study uses Alpamayo 1 deliberately and keeps the model id as a parameter so a 1.5 row can be added with the same runner.
