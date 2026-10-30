# Case study 05: Cosmos Predict 2.5, the cost of one video forecast

**Question:** what does one forecast from NVIDIA's open Cosmos-Predict2.5-2B
world model cost on the RTX 5090, in wall time, peak GPU memory and frames
produced, and does it fit the Jetson AGX Thor at all?

**Status: filled from measured records, 2026-10-30.** The numbers below were
not produced by this case study's own protocol ([PLAN.md](PLAN.md)); they are
read from generation records committed in a private repository that ran the
same model on the same two boards, pointed to by hash in
[`sources.json`](sources.json). Where the plan asked for something those
records do not hold (three repeats, power, a 100 ms NVML trace, the time
inside the denoising loop), this page says "not recorded". No forecast
quality is claimed anywhere; the generated frames are not shown.

## Boards

| Board | Role here |
|---|---|
| RTX 5090 workstation (32,607 MiB, driver 580.178.04) | one native generation, 2026-10-20 |
| Jetson AGX Thor (122.8 GB unified memory, 120 W mode, L4T R38.4) | admission and one native attempt, 2026-10-30 |

## Results, RTX 5090

One generation through the upstream inference code at commit `a2c298b0`,
checkpoint revision `15a82a2ec231bc318692aa0456a36537c806e7d4`, BF16,
torch 2.7.0+cu128, Python 3.10, seed 20261020, guidance 7, 35 steps,
guardrails on (prompt and generated video both passed, face blur kept).
Input: a generated simulation clip, 32 frames supplied, of which the model
conditions on the last 5 frames (0.25 s).

| Measure | Value | Source |
|---|---|---|
| Native generation wall time (model construction, guardrails and postprocessing included; a CPU check ran concurrently) | **773.9 s** | `rtx5090-native-integration-2026-10-20`, `timing.native_seconds` |
| Bridge and orchestration wall time | 776.9 s and 783.2 s | same, `timing` |
| Time inside the denoising loop | not recorded | |
| Peak process GPU memory | **33,185,333,248 bytes (30.9 GiB)** of 32,607 MiB | same, `memory.peak_process_vram_bytes` |
| Allocator peak allocated / reserved | 31,124,009,984 / 32,554,090,496 bytes | same, `memory.torch_peak_*` |
| Free before the run | 31,769 MiB; headroom over the prior measured peak 126,877,696 bytes | same, `device` |
| Native output | 93 frames, 1280 x 704, 16 fps; 88 future frames, 5.5 s | same, `native_output` |
| Exported future clip | 48 frames, 1280 x 720, 3.0 s | same, `exported_future` |
| A second record, reused without a new GPU call | wall 744.5 s, peak 33,185,333,248 bytes | `rtx5090-confirmed-native-pilot-2026-10-21` |
| GPU power, host RAM, repeats | not recorded | |

What it says: one forecast of 5.5 s of video takes about 13 minutes on the
5090 and uses the whole card. The peak sits 127 MB under the card's free
memory, which is why the plan's first question was whether it fits at all:
it fits, with nothing to spare. The two records give wall times of 773.9 and
744.5 s; whether they are one generation or two is not established here, so
this is not a repeatability figure.

## Results, Jetson AGX Thor

The Thor attempt runs under a shared-host admission policy that was fixed
before any Thor work: at least 33,185,333,248 bytes of CUDA free memory
(the 5090 peak above) and 68,719,476,736 bytes of host memory available,
observed within 15 s of launch, with a named resident model server as the
only other GPU process. No threshold was changed for this case study.

**Finding, admission on a unified-memory board.** On the Thor,
`cudaMemGetInfo` reports free memory that follows the host's `MemFree`,
not `MemAvailable`. After a day of weight copies and engine builds the page
cache held 88.0 GB; with no GPU process running the board reported 5.5 GB
MemFree, 94.6 GB MemAvailable and 5.16 GB CUDA free of 131.9 GB. The first
pilot attempt that day was refused at admission with 2.09 GB CUDA free
against the 33.19 GB floor while 83.0 GB of host memory was available. One
authorized page-cache drop (`echo 3 > /proc/sys/vm/drop_caches`) changed
the same idle board to 124.8 GB MemFree, 2.3 GB cached and 126.0 GB CUDA
free. Nothing on the GPU changed between the readings. An admission rule
keyed on CUDA free memory therefore needs the cache state recorded beside
it, or it refuses a board that has the room.

**Result: admitted, no video.** After that drop, attention controls passed and
admission allowed the run with 84,957,700,096 bytes CUDA free and
93,739,421,696 bytes host memory available. The container exited 1 after 67 s
in the text guardrail model load, before generation, with no out-of-memory
kill. The cause is not established. So this case study has no Thor generation
cost: not wall time, not peak memory.

## Phases, as they stand

1. Environment: done privately, not in this repository.
2. One forecast on the 5090: done, the numbers above.
3. Three repeats, the table, the plots: not done; the plan stands.
4. Thor run: attempted once; admitted and failed before generation, as above.

## What this is not

Not a benchmark of the model's usefulness, not a comparison between world
models, and not a product feature. The input scene, the generated frames
and anything about the private project's purpose stay out of this page.
