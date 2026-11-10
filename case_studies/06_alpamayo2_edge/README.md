# Case study 06: Alpamayo 2 Edge, the student, on the bench and the edge

**Question:** case study 02 measured NVIDIA's 10B Alpamayo 1, which NVIDIA
positions as a teacher model. Neither board ran it at the 10 Hz a driving
planner needs. NVIDIA released Alpamayo 2 Edge on 2026-11-10, a 2B-class model
distilled from Alpamayo 2 Super for the in-vehicle DRIVE AGX Thor. What does one
decision of the student cost on the same boards and clips, as shipped and with
the optimizations that can be reproduced here?

Every number below comes from a committed JSON record under `runs/`. The plan,
budgets and amendments are in [PLAN.md](PLAN.md); every error and workaround,
in the order met, is in [FRICTION_LOG.md](FRICTION_LOG.md).

**Status (2026-11-10):** measured on both boards as shipped and with four
optimizations, singly and combined. On the Jetson AGX Thor the p95 decision
latency went from 2.70 s as shipped to 0.61 s on a 10 Hz stream. Neither board
meets the 0.1 s deadline at p95; the 5090 reaches it at the median (0.100 s).
NVIDIA's TensorRT deployment is blocked on a recipe that is not public yet. All
Thor runs are at the 90 W power mode; MAXN is pending an owner decision.

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
samples the trajectory.

## Results: one decision per clip

Eight clips (the same as case study 02, same moment), one warm-up excluded,
three decisions per clip (24), the upstream CLI's sampling settings, seed 43.
Energy per decision is from a 30 s block of decisions back to back (PLAN.md,
amendment of 2026-11-10); on the 5090 the NVML energy counter agrees within 1
percent. Power on the 5090 is the GPU rail; on the Thor it is the whole board.

| Board and variant | Median | p95 | Meets 0.1 s | Time to first token | Energy per decision | minADE (sanity) |
|---|---|---|---|---|---|---|
| RTX 5090, BF16 eager (as shipped) | 0.344 s | 0.475 s | no | 0.170 s | 135 J | 0.81 m |
| RTX 5090, 5 diffusion steps | 0.281 s | 0.410 s | no | 0.172 s | 120 J | 0.80 m |
| RTX 5090, FP8 eager | 0.494 s | 1.014 s | no | 0.190 s | 161 J | 0.83 m |
| RTX 5090, upstream acceleration | 0.186 s | 0.249 s | no | 0.145 s | 93 J | 0.82 m |
| RTX 5090, acceleration + 5 steps | 0.195 s | 0.260 s | no | 0.161 s | 92 J | 0.81 m |
| RTX 5090, acceleration + FP8 + 5 steps | 0.172 s | 0.219 s | no | 0.124 s | 80 J | 0.78 m |
| Jetson AGX Thor, BF16 eager (as shipped) | 2.150 s | 2.703 s | no | 1.196 s | 130 J | 0.80 m |
| Jetson AGX Thor, 5 diffusion steps | 1.713 s | 2.288 s | no | 1.046 s | 104 J | 0.72 m |
| Jetson AGX Thor, upstream acceleration | 0.934 s | 1.209 s | no | 0.671 s | 74 J | 0.76 m |
| Jetson AGX Thor, acceleration + 5 steps | 0.906 s | 1.175 s | no | 0.668 s | 74 J | 0.73 m |
| Jetson AGX Thor, acceleration + FP8 + 5 steps | 1.024 s | 1.344 s | no | 0.679 s | 66 J | 0.73 m |

## Results: a 10 Hz stream, with and without vision reuse

`run_stream.py`: three clips (a right turn, a straight run that writes
reasoning, a left turn), one decision every 0.1 s of clip time for 3 s; the
first 20 fill the cache as two seconds of driving would, the last 10 per clip
are timed (30 decisions). With reuse, the vision tower runs only on frames it
has not seen; every cache hit is checked against a fingerprint of the image
(no mismatch in any run), and one decision per run is recomputed without the
cache and compared.

| Board and variant | Without reuse: median / p95 | With vision reuse: median / p95 | Largest waypoint difference with reuse |
|---|---|---|---|
| RTX 5090, BF16 eager | 0.347 s / 0.479 s | 0.292 s / 0.422 s | 1.7 cm |
| RTX 5090, upstream acceleration | 0.183 s / 0.243 s | 0.128 s / 0.188 s | 3.2 cm |
| RTX 5090, acceleration + FP8 + 5 steps | 0.156 s / 0.215 s | **0.100 s / 0.161 s** | 1.3 cm |
| Jetson AGX Thor, BF16 eager | 1.642 s / 2.093 s | 1.264 s / 1.708 s | 0.0 cm |
| Jetson AGX Thor, upstream acceleration | 0.834 s / 1.121 s | 0.458 s / 0.743 s | 0.0 cm |
| Jetson AGX Thor, acceleration + 5 steps | 0.797 s / 1.090 s | 0.429 s / 0.722 s | 0.0 cm |
| Jetson AGX Thor, acceleration + FP8 + 5 steps | 0.780 s / 0.983 s | **0.404 s / 0.611 s** | 0.0 cm |

## Teacher against student, same clips, as shipped

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

## What was optimized, and what each change bought

1. **Upstream acceleration** (`alpamayo2_edge.accel`, torch.compile and CUDA
   graphs, same weights and precision). The largest single gain on both boards:
   5090 p95 0.475 to 0.249 s, Thor p95 2.70 to 1.21 s, Thor energy 130 to 74 J.
   On the Thor it needed flash-attn built from source and Triton pointed at the
   JetPack `ptxas` (FRICTION_LOG).
2. **Vision reuse across a 10 Hz stream.** Encode only the frames not seen
   before. Saves about 55 ms per decision on the 5090 and about 0.38 s on the
   Thor; with acceleration it takes the Thor stream from 1.12 s to 0.74 s at p95.
3. **Five diffusion steps instead of ten**, as in NVIDIA's DRIVE Thor recipe.
   Saves 60 to 440 ms in eager mode, little once the expert is compiled.
   minADE did not get worse on these clips.
4. **FP8 (torchao, per tensor, language model only).** Worse in eager mode on the
   5090 (p95 above 1 s); best on the 5090 under compilation (0.219 s p95, 80 J);
   on the Thor small and inconsistent (better in the stream, slightly worse in the
   per-clip runs).
5. **Input preparation outside the decision:** building the image processor once
   instead of on every call cut preparation from 0.80 to 0.04 s per clip on the
   5090 and from 1.88 to 0.12 s on the Thor, with byte-identical inputs.

Not done: the TensorRT Edge-LLM deployment (recipe not public), speculative
decoding (no Edge drafter released), MAXN power mode (owner decision pending).

## Issues found

The full list with causes is in [FRICTION_LOG.md](FRICTION_LOG.md). The ones a
developer should know before starting:

1. **The public dataset has no route instruction, and the model will not run
   without one.** Derived here by a fixed rule from the logged path.
2. **Upstream input preparation reloads the processor from disk on every call**,
   which cost more than the decision itself on the 5090.
3. **Short decisions break coarse power sampling.** At 0.5 s sampling most
   0.35 s decisions held no power sample; energy needs a counter or a sustained
   window.
4. **As shipped, the student misses the 10 Hz deadline on both boards**, as
   NVIDIA states for BF16 PyTorch.
5. **On the Thor, torch.compile fails until Triton is pointed at the system
   `ptxas`**: Triton 3.6's bundled `ptxas-blackwell` does not know sm_110a, and
   only `TRITON_PTXAS_BLACKWELL_PATH` fixes it. flash-attn needs a source build,
   and without `ninja` it compiles one file at a time.
6. **FP8 is not free.** In eager PyTorch it slowed the model down; it needs
   compilation to pay off, and the row-wise variant fails under the upstream
   autocast.
7. **NVIDIA's 160 to 190 ms on DRIVE AGX Thor cannot be reproduced yet:** the
   Edge TensorRT recipe named in NVIDIA's blog is not in the public recipes
   repository.

## Reproduce

```bash
# 5090 host: environment, weights and the logged futures under /mnt/backup
cd /mnt/backup/alpamayo2-edge-20261110/code && UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --extra dataset --dev
python derive_nav.py --clips ../02_alpamayo_av_reasoning/clips.yaml --out nav_clips.json --futures /mnt/backup/alpamayo2-edge-20261110/nav_futures.json
PYTHONPATH=<bench2field>/src .venv/bin/python run_decisions.py --board bench-5090 --variant eager \
    --model /mnt/backup/alpamayo2-edge-20261110/model --nav /mnt/backup/alpamayo2-edge-20261110/nav_futures.json
# variants: --variant tuned, --diffusion-steps 5, --fp8, --prep cached; stream: run_stream.py
# Thor: setup_env_thor.sh, then thor_matrix.sh and thor_matrix2.sh (TRITON_PTXAS_BLACKWELL_PATH set there)
```
