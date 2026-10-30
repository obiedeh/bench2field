# Case study 02 plan: Alpamayo 1 on the bench and the edge

**Question:** what does one driving decision from Alpamayo 1 cost on this lab's
hardware, in time, memory and power, and how much of the bench's performance
survives on a Jetson?

Written before any decision was measured, so the protocol cannot be fitted to
the numbers (AGENTS.md rule 3). Changes to this plan are appended, dated, under
"Amendments"; nothing above that line is edited after the first run.

## Setup

| Item | Choice | Why |
|---|---|---|
| Model | `nvidia/Alpamayo-R1-10B` (Alpamayo 1), bf16, inference code NVlabs/alpamayo at commit `11a0e01` | The open reasoning VLA for driving; the newer 1.5 has the same pins, so the model id stays a parameter |
| Precision | bf16 everywhere, as upstream | No quantization in this case study |
| Attention | `sdpa` on every board as the baseline; flash-attn 2.8.3 as a second variant on the 5090 only | sdpa is the only option on the Thor; the comparison on the 5090 shows what the Jetson gives up |
| Bench | RTX 5090 (`bench-5090`) | development machine, Nsight and torch.profiler |
| Edge bench | Jetson AGX Thor developer kit on JetPack 7.1 (`bench-thor`) | the edge target; Jetson, not DRIVE AGX Thor on DriveOS |
| Field | Jetson Orin NX 8 GB (`field-orin`) | does not fit; documented, not run |
| Data | PhysicalAI-Autonomous-Vehicles clips, streamed by `physical_ai_av` | the dataset Alpamayo 1 was released with |

## Protocol (fixed before the first run)

- **Clips:** the upstream example clip `030c760c-ae38-49aa-9ad8-f5650a545d26`
  plus the first seven clip ids of `notebooks/clip_ids.parquet` in NVlabs/alpamayo
  in file order. Eight clips, listed in `clips.yaml`. Selected by rule, not by result.
- **Reference time:** `t0_us = 5_100_000` for every clip, as in the upstream
  script. Inputs: four cameras, four frames each (0.4 s at 10 Hz), 16-point ego
  history. Nothing else.
- **One decision** = one call of `sample_trajectories_from_data_with_vlm_rollout`
  with `num_traj_samples=1`, `top_p=0.98`, `temperature=0.6`,
  `max_generation_length=256`, seed 42 before each call, under bf16 autocast.
  Upstream defaults, unchanged.
- **Repeats:** one warm-up decision (reported separately, excluded from
  statistics), then three decisions per clip per variant, clips in file order,
  repeats as outer loop.
- **Measured per decision:** wall time from inputs on device to outputs on
  device (CUDA synchronised), generated token count, trace text, the 64
  waypoints, peak allocated and reserved GPU memory (torch) and peak used
  memory (NVML), NVML power and utilisation sampled at 0.5 s; on the Thor,
  tegrastats `VIN` as `power_board_w`.
- **Measured once per run:** model load time (from `from_pretrained` to the
  model on the device), memory after load, the environment record the run
  used.
- **Reported:** per board and variant, median and p95 latency over the 24
  decisions, decisions per minute (60 / median), tokens per second of the
  rollout (generated tokens / rollout time), peak memory, and on the Thor the
  mean board power during decisions and the energy per decision (mean power x
  median latency). The 5090 to Thor ratio is the field retention of this case
  study. Three sample traces committed verbatim.
- **Profile (phase 3):** `torch.profiler` on one clip, one decision, CPU and
  CUDA activities, split into vision encoding, language rollout and trajectory
  sampling; plotted next to the NVML timeline of the same decision.

## Phases

### Phase 1: environment and scaffold
- [x] venv with the upstream pins on the 5090; flash-attn wheel checked on the GPU
- [x] friction log, setup notes, environment record
- [x] this plan

### Phase 2: 5090 run
- [x] `clips.yaml` and the runner
- [x] `runs/decisions_bench-5090_sdpa.json`
- [x] `runs/decisions_bench-5090_flash.json` (flash on the VLM, sdpa on the expert; see Amendments)
- [x] three traces in `traces/`; README results table filled from the JSON only

### Phase 3: profile
- [x] profiler and NVML timeline on one decision; plots under `report/`

### Phase 4: Thor
- [x] same clips, same protocol, sdpa, Jetson torch wheel named; `runs/decisions_bench-thor_sdpa.json`
- [x] comparison table and retention

### Phase 5: Orin NX
- [ ] why-not note with measured facts

## Out of scope

Accuracy (minADE against the dataset's future trajectory is printed by the
runner for sanity, not reported as a result: eight clips are not an evaluation),
quantization, CUDA graphs, batch sizes above one, Alpamayo 1.5 and 2.

## Amendments

Dated, appended after the first run; the protocol above is not edited.

- **2026-10-29, what "repeats" measure.** Seed 42 before every decision makes
  the three repeats of a clip produce the same tokens, trace and waypoints on
  a given board. The repeats therefore measure timing spread, not output
  variance. Across boards the same seed does not reproduce the text (the Thor
  and the 5090 wrote different token counts on the same clip). The protocol
  stands as written; this is how to read it.
- **2026-10-29, the flash variant.** The protocol said "flash-attn 2.8.3 as a
  second variant on the 5090". Two facts changed what that means. First,
  upstream's config declares `flash_attention_2` as its default, but the
  config round-trip in transformers 4.57.1 drops that field to `None`, so
  upstream's own demo runs `sdpa` on the VLM and on the diffusion expert
  (checked by loading the model the way the demo does and reading the
  attention implementation in effect). Second, forcing `flash_attention_2`
  on both sub-models makes trajectory sampling fail inside flash-attn's
  varlen forward. The flash variant is therefore defined as flash-attn on
  the VLM (vision tower, prefill, decode) with `sdpa` kept on the diffusion
  expert, and the JSON records both choices. The sdpa baseline is unchanged.
