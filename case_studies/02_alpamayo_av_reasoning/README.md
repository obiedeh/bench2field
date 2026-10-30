# Case study 02: Alpamayo 1, an AV reasoning model, on the bench and the edge

**Question:** what does one driving decision from NVIDIA's open Alpamayo 1 model
cost on the hardware in this lab, in time, memory and power, and how much of the
bench's performance survives on a Jetson?

Alpamayo 1 (`nvidia/Alpamayo-R1-10B`, 10B parameters, weights under OpenMDW-1.1,
inference code Apache-2.0) reads four cameras over 0.4 s plus the ego vehicle's
recent motion, writes a chain-of-causation reasoning trace, and samples a 6.4 s
trajectory at 10 Hz. Every number here comes from a committed JSON record on a
named board, as in case study 01.

**Status: phases 2 and 3 done, the 5090 is measured and profiled.** Results below come from the
committed JSON only; a cell says "pending" until the run that produces it is committed.

## Boards

| Board | Role | Target for | State |
|---|---|---|---|
| RTX 5090 (`bench-5090`) | bench | full-precision run, profile, plots | measured, sdpa and flash-attn (phase 2); profiled (phase 3) |
| Jetson AGX Thor (`bench-thor`) | edge bench | same clips, with power | planned (phase 4); JetPack, not DriveOS |
| Jetson Orin NX 8 GB (`field-orin`) | field | nothing | does not fit; documented in the friction log |

## Results

Measured values only, from the committed JSON under `runs/`. The table is the
output of `plot_results.py --table` (phase 3) and is regenerated, not edited,
when a run is added. Protocol: eight clips, one warm-up decision excluded, three
decisions per clip, upstream sampling defaults, seed 42 before each decision.
Power on the 5090 is the GPU rail from NVML; power on the Thor is the whole
board (`VIN` from tegrastats). The two power rows are therefore not like for
like; the Thor figure includes CPU, memory and carrier board.

| Metric | RTX 5090, flash-attn | RTX 5090, sdpa |
|---|---|---|
| Model load time | 8.3 s | 15.7 s |
| Latency per decision, median | 0.93 s | 0.89 s |
| Latency per decision, p95 | 1.24 s | 1.16 s |
| Time to first token, median | 0.37 s | 0.38 s |
| Decode rate, median | 50.9 tok/s | 60.0 tok/s |
| Reasoning tokens, median | 18 | 18 |
| Decisions per minute | 64.3 | 67.8 |
| GPU memory after load | 20.6 GiB | 20.6 GiB |
| Peak GPU memory during a decision | 21.5 GiB | 21.5 GiB |
| Power during a decision, median | 359 W | 377 W |
| Energy per decision, median | 323 J | 316 J |
| Decisions | 24 (8 clips x 3) | 24 (8 clips x 3) |

The flash-attn column is flash-attn 2.8.3 on the VLM with sdpa kept on the
diffusion expert (PLAN.md, Amendments). On this GPU, at batch size one and
these sequence lengths, it is slower than sdpa in decode (50.9 against 60.0
tokens per second) and equal in time to first token, so sdpa stays the
baseline and is the variant the Thor is compared against. The Thor column is
added in phase 4.

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

## Phases

1. Review and environment: the venv, the pins, the friction log, this scaffold. Done.
2. 5090 run on a handful of PhysicalAI-AV clips: load time, latency per decision, peak memory, throughput, sample reasoning traces.
3. Profile: where the time goes between the vision encoder, the language rollout and the diffusion trajectory head, with plots.
4. Thor, same clips, power from tegrastats, compared with the 5090.
5. Orin NX: why a quantized run is not feasible on this module.

## Layout

```text
README.md            this file
PLAN.md              question, protocol fixed before the first run, phases
SETUP.md             exact setup per board
FRICTION_LOG.md      what broke, how it was found, what fixed it, with times
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
