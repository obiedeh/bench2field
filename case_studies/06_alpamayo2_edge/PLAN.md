# Case study 06: Alpamayo 2 Edge, the student, on the bench

Fixed on 2026-11-10, before the first timed run. Changes after that are dated
amendments at the end of this file; nothing above them is rewritten.

## Question

Case study 02 measured NVIDIA's Alpamayo 1 (10B), which NVIDIA positions as a
teacher model, on the rtx-5090-workstation and the jetson-agx-thor. Neither ran it
at the 10 Hz a driving planner needs (0.89 s and 4.38 s median per decision).
NVIDIA released Alpamayo 2 Edge on 2026-11-10: a 2B reasoning VLA, offline-distilled
from Alpamayo 2 Super (34B), built for DRIVE AGX Thor. NVIDIA reports 160 to 190 ms
pixel to trajectory after FP8, TensorRT Edge-LLM and speculative decoding on DRIVE
AGX Thor-X, and states that the BF16 PyTorch model is not real-time.

What does one decision of the student cost on the same two boards, as shipped and
with the speedups that can be reproduced here, and how far is it from the teacher
in latency, energy, memory and trajectory error on the same clips?

## Setup

| Item | Value | Note |
|---|---|---|
| Model | `nvidia/Alpamayo2-Edge` at revision `f8951d095de3589b3face1dec3f4394c8ecedb38`, BF16 | about 5.3 GB, not gated |
| Code | NVlabs/alpamayo2-edge at commit `02f55a7` | PyTorch 2.8.0, torchvision 0.23.0, Transformers 5.17.0 (its lock) |
| Teacher, for comparison | case study 02 records (Alpamayo 1, `nvidia/Alpamayo-R1-10B`) | not re-run |
| Boards | `rtx-5090-workstation`, `jetson-agx-thor` | weights and environments under `/mnt/backup` |
| Data | PhysicalAI-Autonomous-Vehicles, streamed by `physical_ai_av` | gated dataset; clips never committed |

## Protocol

- **Clips:** the eight clips of case study 02 (`../02_alpamayo_av_reasoning/clips.yaml`)
  at the same `t0_us = 5_100_000`, so teacher and student see the same moments.
  NVIDIA's own example (clip `ea7bbd31-b7a5-4972-8dbd-7089e6b53de4`, 4.0 s,
  "Turn right in 30m") is run first as a smoke test and kept out of the statistics.
- **Inputs:** the Edge input profile, unchanged: six cameras, 22 frames, 20 ego poses.
  The teacher saw four cameras, 16 frames and 16 ego poses. The comparison is model
  as shipped against model as shipped, and the write-up says the inputs differ.
- **Navigation text:** required by Edge, absent from the public dataset. Derived per
  clip by the fixed rule in `derive_nav.py` from the logged future path (heading
  change of at least 30 degrees in 6.4 s is a turn, else "Continue straight"),
  written to `nav_clips.json` before any timed run. A route planner knows the route;
  this stands in for it and is stated as using logged data.
- **One decision:** one `sample_trajectories_from_data` call with
  `num_traj_sets=1, num_traj_samples=1`, the Edge CLI's default sampling settings,
  a fixed seed before each call, BF16. Timed from inputs on the device to outputs
  on the device, CUDA synchronised. Input loading and preprocessing are timed
  separately and reported apart, as in case study 02.
- **Repeats:** one warm-up decision (excluded), then three decisions per clip,
  repeats as the outer loop: 24 decisions per board and variant.
- **Measured:** latency median and p95, decisions per minute, generated tokens
  (and whether the adaptive `<NOMINAL>` path was taken), peak GPU memory, NVML
  power on the 5090 (GPU rail) and tegrastats `VIN` on the Thor (whole board),
  energy per decision, model load time, and minADE against the logged future as a
  sanity value, the same value case study 02 recorded for the teacher.

## Variants, in order

1. `bench-5090`, BF16 eager, SDPA attention (the shipped default).
2. `bench-5090`, `--accel tuned` (torch.compile and CUDA graphs; needs flash-attn 2.8.3). Same weights and precision.
3. `bench-thor`, BF16 eager.
4. `bench-thor`, accelerated, if flash-attn builds for the Thor.
5. `bench-thor`, the TensorRT Edge-LLM recipe from NVlabs/alpamayo-recipes (FP8). It targets DRIVE OS; whether it runs on JetPack is part of the result.

## Budgets (fixed before the first timed run, AGENTS.md rule 3)

- **Decision deadline:** 0.1 s per decision, the 10 Hz a driving planner consumes
  (the same reference case study 02 used). A board and variant meets it only if its
  p95 decision latency is at or under 0.1 s.
- **NVIDIA's reference:** 160 to 190 ms pixel to trajectory on DRIVE AGX Thor-X with
  FP8 and TensorRT Edge-LLM. Recorded for comparison, not as a budget: it is a
  different board, software stack and precision.
- **Memory:** must fit beside a perception stack on the Thor; reported, no limit set.
- **Accuracy:** no pass mark is set for minADE; it is a sanity value against one
  logged future, as in case study 02.

## Rules

- Every error, workaround and dead end goes into `FRICTION_LOG.md` as it happens.
- Results in the README come from committed JSON records only.
- No board host names, home paths or LAN addresses in any committed file.

## Amendments

- **2026-11-10, after the first full 5090 run (record not kept as a result):** a
  student decision takes about 0.35 s, shorter than the 0.5 s telemetry interval,
  so most decisions held zero or one power sample and the energy per decision was
  not trustworthy. The teacher's decisions (0.9 to 4.4 s) did not have this
  problem. Changes: telemetry every 0.1 s; on the 5090, the NVML cumulative energy
  counter read before and after each decision; and a sustained block after the
  timed decisions (about 30 s of decisions back to back over the clips), whose
  energy per decision is mean power times the window divided by the decisions in
  it. The sustained figure is the one reported for both boards; the per-decision
  counter is reported beside it on the 5090. Latency, tokens, memory and minADE
  are unchanged by this. The first record is kept outside git with the run log.
