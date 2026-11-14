# Clock and thread-spinning retest, 2026-11-11

Two phase 1 statements were tested again on 2026-11-11 with the same
profiler (`profile_pipeline.py`), the same saved frame sets and TensorRT
FP32. Both are corrected in `PHASE1_FINDINGS.md` by dated notes that point
here.

1. **The rover's 47.8 ms full frame was a default-clock figure.** With the
   board's clocks pinned at maximum the same frame takes about 21 ms.
2. **The thread-spinning frame spikes did not reproduce.** On three
   machines, idle and under load, the frame's tail is the same with
   onnxruntime's thread spinning on or off.

Every figure below is printed by `python3 scripts/summarize.py` from the
records in this directory.

## 1. Clocks as shipped against clocks pinned

Method: `scripts/board_run_pinned.sh`. The whole-frame profile (300 frames
after 30 of warmup, one frame at a time, thread spinning off) on the 640x480
rover frame set, in the order default, pinned, pinned, default, twice: four
runs per condition per board. "Pinned" is `jetson_clocks` switched on through
the board's monitoring service, which raises the CPU, GPU and memory-bus
clocks to their maximum together; "default" is the board as shipped, with
frequency scaling. Clock state was written before each run (`*.state`) and
sampled every half second during it (`*.freq`). Both boards were restored to
default at the end (`final.state`).

| Board | Clocks | Whole frame p50 | Whole frame p95 | Inference p50 | GPU clock during the run |
|---|---|---|---|---|---|
| Jetson Orin NX (rover), MAXN_SUPER | default | 45.1 to 47.4 ms | 48.6 to 49.3 ms | 24.8 to 24.9 ms | mostly 408 of 1,173 MHz |
| Jetson Orin NX (rover), MAXN_SUPER | pinned | 20.7 to 20.9 ms | 21.1 to 21.5 ms | 10.8 to 10.9 ms | 1,173 MHz |
| Jetson AGX Thor, MAXN | default | 14.2 to 15.1 ms | 15.3 to 16.7 ms | 4.7 to 5.3 ms | 315 to 1,575 MHz, moving |
| Jetson AGX Thor, MAXN | pinned | 6.9 to 7.0 ms | 7.1 to 7.2 ms | 2.4 to 2.5 ms | 1,575 MHz |

The default-clock runs reproduce the 2026-10-02 record
(`../profile_orin_rover_frames_480p.json`: 43.9 ms p50, 47.8 ms p95). With
the clocks pinned the Orin NX frame is inside the 33.3 ms deadline.

### What pinning costs in power

Method: `scripts/board_run_power_orin.sh` and `scripts/board_run_power.sh`
with `scripts/power_driver.py`: 45 s idle with the session loaded, 60 s
paced, 60 s back to back, in the order default, pinned, pinned, default (two
runs per condition). Board input rail from `tegrastats` once a second.

| Board | Clocks | Idle | Paced at 30 frames per second | Frames done in 60 s at 30 per second | Back to back |
|---|---|---|---|---|---|
| Jetson Orin NX | default | 5.5 W | 8.9 W | 1,322 and 1,331 of 1,800 | 8.8 to 8.9 W |
| Jetson Orin NX | pinned | 8.3 to 8.4 W | 14.5 W | 1,800 of 1,800 | 18.2 W |
| Jetson AGX Thor | default | 20.8 to 21.2 W | 24.9 to 25.0 W | 1,800 of 1,800 | 29.4 to 31.2 W |
| Jetson AGX Thor | pinned | 24.6 to 24.8 W | 28.5 to 29.7 W | 1,800 of 1,800 | 49.2 to 49.4 W |

On the Orin NX the pinned clocks cost about 60% more board power at 30
frames per second (8.9 W to 14.5 W). At default clocks the Orin NX could not
keep 30 frames per second at all: it finished about 1,330 of the 1,800
frames.

## 2. Thread spinning on against off

Method: alternating pairs of the same profile with onnxruntime's
`allow_spinning` on and off (`scripts/run.sh`, `run_saturated.sh`,
`board_run.sh`, `board_run_busy.sh`). "Busy" means that many background
processes each holding a CPU core.

| Machine and condition | Frame set | Runs each | Frame p95, spinning on | Frame p95, spinning off | Longest frame, on / off |
|---|---|---|---|---|---|
| RTX 5090 workstation, quiet | 720p | 5 | 5.6 to 6.5 ms | 5.3 to 6.1 ms | 6.9 / 9.2 ms |
| RTX 5090 workstation, 4 busy | 720p | 3 | 5.4 to 5.8 ms | 5.3 to 5.7 ms | 6.2 / 5.8 ms |
| RTX 5090 workstation, 12 busy | 720p | 3 | 7.1 to 7.3 ms | 6.7 to 7.2 ms | 8.1 / 8.9 ms |
| RTX 5090 workstation, 24 busy | 720p | 3 | 6.1 to 12.8 ms | 5.4 to 12.6 ms | 16.5 / 16.3 ms |
| Jetson AGX Thor, quiet | 480p | 5 | 15.6 to 20.0 ms | 15.0 to 15.7 ms | 20.6 / 16.5 ms |
| Jetson AGX Thor, 10 busy | 480p | 3 | 7.5 to 7.8 ms | 7.5 to 8.2 ms | 9.4 / 8.9 ms |
| Jetson Orin NX, quiet | 480p | 5 | 47.8 to 48.7 ms | 48.5 to 49.0 ms | 49.9 / 50.5 ms |
| Jetson Orin NX, quiet | 720p | 5 | 61.9 to 64.8 ms | 59.1 to 64.2 ms | 65.7 / 66.3 ms |
| Jetson Orin NX, 4 busy | 480p | 3 | 37.2 to 37.6 ms | 37.5 to 39.3 ms | 45.7 / 50.7 ms |

The 2026-10-03 record that started the claim
(`../profile_5090_rover_frames_720p.json`) has a 44.7 ms frame p95 and
preprocess frames up to 83 ms with spinning on, against 6.6 ms with it off.
No run here comes near that with spinning on: the longest preprocess step in
any workstation run is 5.7 ms, and that one had spinning off. The October
spike is a real record, but it was a single run per setting and its cause is
not established. It is not evidence that thread spinning steals the frame.

Both Jetson boards ran faster with busy background processes than when
quiet (Orin NX 37 ms against 48 ms, Thor 7.5 ms against 15 ms). That fits the
clock result above: background load keeps the governors from clocking down.

## Limits

- Saved image files, not the live camera with the rover's stack running.
  The frames show a blank wall, so decode and postprocess are at their
  cheapest.
- The CPU, GPU and memory-bus clocks were pinned together, not one at a
  time, so the share each one contributes is not known.
- One test session per board, on one day. Four runs per clock condition,
  two per power condition.
- The Thor ran in MAXN here; the October Thor profile ran in the 120 W mode.
- Pinning the clocks is a setting, not a deployment: whether the rover's
  power budget and thermals allow it with the full stack running has not
  been measured.

## Files

- `orin/`, `thor/`, `workstation/`: profile records (`*.json`), profiler
  logs, load averages, clock state and samples, `tegrastats` logs.
- `scripts/`: the scripts as run, and `summarize.py`.
- `SHA256SUMS.as_measured`: hashes of every file as it came off the
  machines. Host names and home-directory paths were then replaced with
  board labels and `~` in the JSON records and one log, so those files no
  longer match their as-measured hash; the originals are kept offline. The
  TensorRT engine caches and one trial directory are not committed.
