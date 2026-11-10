# Case study 06: Alpamayo 2 Edge, the student, on the bench and the edge

**Question:** case study 02 measured NVIDIA's 10B Alpamayo 1, which NVIDIA
positions as a teacher model. Neither board ran it at the 10 Hz a driving
planner needs. NVIDIA released Alpamayo 2 Edge on 2026-11-10, a 2B-class model
distilled from Alpamayo 2 Super for the in-vehicle DRIVE AGX Thor. What does one
decision of the student cost on the same boards and clips, as shipped and with
the speedups that can be reproduced here?

Every number below comes from a committed JSON record under `runs/`. The plan,
budgets and amendments are in [PLAN.md](PLAN.md); every error and workaround,
in the order met, is in [FRICTION_LOG.md](FRICTION_LOG.md).

**Status (2026-11-10):** BF16 eager measured on both boards; the 5090
accelerated variant and the Thor accelerated variant are in progress; NVIDIA's
TensorRT deployment is blocked on a recipe that is not public yet.

## The model, and how one decision works

| | Alpamayo 1 (teacher, case study 02) | Alpamayo 2 Edge (student, this study) |
|---|---|---|
| Checkpoint | `nvidia/Alpamayo-R1-10B` | `nvidia/Alpamayo2-Edge` at `f8951d09` |
| Parameters (counted) | 10B | 2,651,163,378 (the "2B" is the language backbone; vision tower and action expert come on top) |
| Inputs | 4 cameras x 4 frames, 16 ego poses | 6 cameras, 22 frames (front cameras 5, others 3), 20 ego poses, a navigation instruction |
| Reasoning | always writes a chain-of-causation trace | adaptive: writes `<NOMINAL>` when the scene is routine, a trace when it is not |
| Output | 64 future poses at 0.1 s | 64 future poses at 0.1 s |
| Code | NVlabs/alpamayo `11a0e01` | NVlabs/alpamayo2-edge `02f55a7`, unchanged |

One decision is one upstream `sample_trajectories_from_data` call: the vision
tower encodes the 22 images, the language model prefills the prompt and writes
either `<NOMINAL>` or a short reasoning trace, and the diffusion action expert
samples the trajectory in 10 steps.

## Results

Eight clips (the same as case study 02, same moment), one warm-up excluded,
three decisions per clip, the upstream CLI's sampling settings, seed 43.
Energy per decision is from a 30 s block of decisions back to back (PLAN.md,
amendment of 2026-11-10); on the 5090 the NVML energy counter gives the same
figure within 1 percent. Power on the 5090 is the GPU rail; on the Thor it is
the whole board. The 0.1 s deadline was set before the first run.

| Metric | RTX 5090, BF16 eager | RTX 5090, accelerated | Jetson AGX Thor, BF16 eager | Jetson AGX Thor, accelerated |
|---|---|---|---|---|
| Latency per decision, median | 0.344 s | pending | 2.150 s | pending |
| Latency per decision, p95 | 0.475 s | pending | 2.703 s | pending |
| Meets the 0.1 s deadline (p95) | no | pending | no | pending |
| Time to first token, median | 0.170 s | pending | 1.196 s | pending |
| Diffusion action expert, median | 0.129 s | pending | 0.562 s | pending |
| Generated tokens, median | 6 | pending | 11 | pending |
| Decisions on the nominal path | 62.5 % | pending | 50 % | pending |
| Peak GPU memory | 6.0 GiB | pending | 6.0 GiB | pending |
| Power in the sustained block, mean | 350 W (GPU rail) | pending | 58 W (whole board) | pending |
| Energy per decision (sustained) | 135 J | pending | 130 J | pending |
| Input preparation on the CPU, median per clip | 0.80 s | pending | 1.88 s | pending |
| minADE against the logged future, median (sanity only) | 0.81 m | pending | 0.80 m | pending |

## Teacher against student, same clips

| | RTX 5090 | Jetson AGX Thor |
|---|---|---|
| Latency per decision, median | 0.89 s to 0.34 s | 4.38 s to 2.15 s |
| Energy per decision | 316 J to 135 J | 290 J to 130 J |
| Peak GPU memory | 21.5 GiB to 6.0 GiB | 21.5 GiB to 6.0 GiB |
| minADE, median (sanity only) | 1.98 m to 0.81 m | 2.16 m to 0.80 m |

Read the comparison with its limits. The student sees six cameras and a route
instruction; the teacher saw four cameras and no route. The route instruction
here is derived from each clip's logged path, which gives the student
information the teacher did not have. minADE is one sample per decision on
eight clips, a sanity value, not an accuracy result. The teacher's energy was
measured per decision with 0.5 s sampling (its decisions were long enough);
the student's in a sustained block. Both models ran as shipped.

## Issues found

The full list with causes is in [FRICTION_LOG.md](FRICTION_LOG.md). The ones a
developer should know before starting:

1. **The public dataset has no route instruction, and the model will not run
   without one.** Derived here by a fixed rule from the logged path.
2. **Input preparation runs on the CPU and costs more than the decision on the
   5090** (0.80 s per clip against 0.34 s). The same pattern case study 02 found
   for the teacher, where a GPU kernel cut it to milliseconds.
3. **Short decisions break coarse power sampling.** At 0.5 s sampling most
   0.35 s decisions held no power sample; energy needs a counter or a sustained
   window.
4. **As shipped, the student misses the 10 Hz deadline on both boards**, as
   NVIDIA states for BF16 PyTorch.
5. **NVIDIA's 160 to 190 ms on DRIVE AGX Thor cannot be reproduced yet:** the
   Edge TensorRT recipe named in NVIDIA's blog is not in the public recipes
   repository.

## What was optimized

Pending: the upstream acceleration package (torch.compile and CUDA graphs with
flash-attn) on both boards, same weights and precision.

## Reproduce

```bash
# 5090 host: environment, weights and the logged futures under /mnt/backup
cd /mnt/backup/alpamayo2-edge-20261110/code && UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --extra dataset --dev
python derive_nav.py --clips ../02_alpamayo_av_reasoning/clips.yaml --out nav_clips.json --futures /mnt/backup/alpamayo2-edge-20261110/nav_futures.json
PYTHONPATH=<bench2field>/src .venv/bin/python run_decisions.py --board bench-5090 --variant eager \
    --model /mnt/backup/alpamayo2-edge-20261110/model --nav /mnt/backup/alpamayo2-edge-20261110/nav_futures.json
# Thor: setup_env_thor.sh, then the same runner with --board bench-thor
```
