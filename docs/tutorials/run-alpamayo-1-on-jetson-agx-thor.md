# Run Alpamayo 1 on a Jetson AGX Thor

A numbered walk from a fresh JetPack 7.1 board to a measured driving decision
from NVIDIA's open Alpamayo 1 model (`nvidia/Alpamayo-R1-10B`), with the
numbers you should expect at each step. It is the path case study 02 took on
2026-10-29; every figure quoted comes from a committed record under
`case_studies/02_alpamayo_av_reasoning/runs/`. Where the board forces a
departure from the upstream recipe, the step says so and why.

What you need: a Jetson AGX Thor developer kit on JetPack 7.1 (L4T R38.4,
CUDA 13.0) in its 120 W power mode, about 30 GB free on the board, a
workstation that can hold the 21 GB of weights once, a Hugging Face account
that has accepted the terms of the `nvidia/PhysicalAI-Autonomous-Vehicles`
dataset, and ssh between the two. Nothing in this guide changes the board's
system packages, driver, JetPack or power mode. Everything lands under the
home directory.

## 1. Understand what you are about to run

Alpamayo 1 reads four camera views over 0.4 s plus the vehicle's recent
motion, writes a short chain-of-causation reasoning trace, and samples a
6.4 s trajectory at 10 Hz. The model is a 10B-parameter vision-language
backbone (Qwen3-VL-2B as the vision and language tower inside a larger
reasoning model) with a diffusion trajectory head. In bf16 it needs about
21.5 GiB of GPU memory at peak, which is why the 128 GB Thor runs it and an
8 GB Orin NX cannot (`case_studies/02_alpamayo_av_reasoning/ORIN_NX.md`).

One decision on the Thor costs 4.38 s median at 64 W board power, against
0.89 s at 377 W on the GPU rail of an RTX 5090. The rest of this guide gets
you to that number.

## 2. Build the environment on the Thor

Upstream pins Python 3.12, `torch==2.8.0`, `transformers==4.57.1` and
`flash-attn>=2.8.3`. Two of those do not exist for the Thor and the guide
departs from them deliberately:

- `torch 2.8.0` has no CUDA 13 aarch64 wheel. Use `torch 2.11.0+cu130`
  from the PyTorch cu130 index (the `manylinux_2_28_aarch64` build).
- `flash-attn` has no prebuilt wheel for the Thor's sm_110 and a source
  build takes hours. Run with `attn_implementation="sdpa"`, which upstream
  supports. On the 5090 flash-attn was measured slower than sdpa at batch
  one anyway (0.93 s against 0.89 s per decision), and upstream's own demo
  never actually enables it under these pins (see step 7).

Run the case study's setup script, which installs exactly that into
`~/.venv/alpamayo` and installs `alpamayo_r1` with `--no-deps` so pip never
tries to build flash-attn:

```bash
# on the workstation: copy the repository without .git, venvs, data or models
cd ~/github/bench2field && git rev-parse HEAD > SOURCE_COMMIT
rsync -a --exclude .git --exclude '.venv' --exclude '*/.venv' --exclude data --exclude models \
      --exclude .trt_cache --exclude __pycache__ ./ <thor>:~/alpamayo-work/bench2field/ && rm SOURCE_COMMIT
# on the Thor
bash ~/alpamayo-work/bench2field/case_studies/02_alpamayo_av_reasoning/setup_env_thor.sh
```

Expect about 330 s, most of it the torch download. The script ends by
importing `AlpamayoR1` and checking that CUDA is visible.

## 3. Record the environment before measuring anything

```bash
~/.venv/alpamayo/bin/python ~/alpamayo-work/bench2field/case_studies/02_alpamayo_av_reasoning/capture_env.py --board bench-thor
```

This writes `runs/env_bench-thor.json`: JetPack, CUDA, torch, transformers,
the `alpamayo_r1` commit, the power mode, free memory. Every later record
names this file. If you change the venv, run it again.

## 4. Get the weights and the dataset onto the board

The weights (`nvidia/Alpamayo-R1-10B`, 21 GB, not gated) download with
`hf download`. On this board name resolution stalled during the download, so
the guide's fallback is to download once on the workstation and copy the
Hugging Face cache directory for the model with rsync into the same layout
on the Thor; the loader does not care which path filled the cache. Verify the
five shards by SHA-256 against their blob names before trusting the copy.

The dataset (`nvidia/PhysicalAI-Autonomous-Vehicles`) is gated. Accept the
terms on its Hugging Face page once, then `hf auth login` on the board. The
case study streams only the eight clips it needs (about 43 MB) and
`HfApi().auth_check(..., repo_type="dataset")` tells you which of the two
403 states you are in: no token, or terms not accepted.

## 5. Smoke-test with synthetic input

```bash
cd ~/alpamayo-work/bench2field/case_studies/02_alpamayo_av_reasoning
~/.venv/alpamayo/bin/python run_decisions.py --board bench-thor --variant sdpa --synthetic
```

Random frames, output under `runs/scratch/`, labelled "not a result". Expect
the model to load in about 25 s and a 20-token decision in about 4.5 s. If
the first decision is much slower, that is the warm-up; the measured protocol
excludes it. The record carries `power_board_w` from tegrastats, the board
rail (`VIN`); the Thor's NVML answers power queries too, but for the GPU
rail only, so the sampler is chosen by board.

## 6. The measured run

```bash
~/.venv/alpamayo/bin/python run_decisions.py --board bench-thor --variant sdpa
```

Protocol, fixed before the first run (`PLAN.md`): eight clips, one warm-up
decision excluded, three decisions per clip, upstream sampling defaults,
seed 42 before each decision, the same seed on every board. Expect, from
`runs/decisions_bench-thor_sdpa.json`:

| Metric | Jetson AGX Thor, sdpa |
|---|---|
| Model load time | 25.9 s |
| Latency per decision, median | 4.38 s |
| Latency per decision, p95 | 5.88 s |
| Time to first token, median | 1.84 s |
| Decode rate, median | 9.7 tok/s |
| Decisions per minute | 13.7 |
| Peak GPU memory during a decision | 21.5 GiB |
| Power during a decision, median | 64 W (whole board) |
| Energy per decision, median | 290 J |

Clip streaming is outside the timings and slow on the board (230 s for eight
clips cold); the record keeps it as `data_load_s`. The seed makes the three
repeats of a clip produce the same trajectory, so repeats measure timing
spread only.

## 7. Read the result against the bench

Same clips, same protocol, same seed on an RTX 5090: 0.89 s per decision,
60 tokens per second, 67.8 decisions per minute. The Thor retains 16 to 25
percent of the 5090's speed on every stage (`README.md`, "5090 against
Thor"), so the shape of the decision is the same on both: the language
rollout dominates, the first token is the largest single wait, trajectory
sampling is the smallest block. Energy per decision is close (316 J on the
5090's GPU rail, 290 J for the whole Thor board).

Two findings you will meet on the way, both in `FRICTION_LOG.md`:

- Upstream's config declares `attn_implementation: flash_attention_2`, but
  `AlpamayoR1Config.from_pretrained` on transformers 4.57.1 returns `None`
  for it, so upstream's demo runs sdpa everywhere and the flash-attn
  requirement is installed but unused. Forcing flash on both sub-models
  crashes trajectory sampling inside flash-attn's variable-length forward.
- torch 2.8.0+cu128 on the 5090 runs sm80 and cutlass_80 kernels; the Thor's
  torch 2.11.0+cu130 runs native sm110 `nvjet` GEMMs
  (`runs/profile_bench-thor_sdpa.json`). The boards do not run the same
  kernels, only the same model.

## 8. Profile a decision on the board

```bash
~/.venv/alpamayo/bin/python profile_decision.py --board bench-thor --variant sdpa
```

`torch.profiler` with stage labels, one decision after a warm-up. Expect the
profiled decision to take about 4.9 s of device time: rollout 4.08 s (vision
0.76 s inside it), trajectory sampling 0.85 s, decode about 100 ms per
token. The kernel table is dominated by bf16 `nvjet_sm110` GEMMs; the top
three take 39 percent of device time. The profiler adds overhead; read the
split as shares, not as the latency.

## 9. Move the camera preprocessing onto the GPU (optional)

Alpamayo calls the Hugging Face image processor on the CPU: 37 ms per
decision on the Thor's 14 cores, 366 ms on the 5090 host. The case study's
CUDA kernels (`kernels/`, `KERNEL_WALKTHROUGH.md`) do the same resize,
normalize and patch layout on the GPU. Build and verify them with the
JetPack toolkit, which matches the torch build:

```bash
uv pip install --python ~/.venv/alpamayo/bin/python ninja
PATH=$HOME/.venv/alpamayo/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST=11.0 \
  ~/.venv/alpamayo/bin/python bench_preprocess.py --board bench-thor
```

The script refuses to write a record unless both kernel variants agree with
the PyTorch GPU reference within one uint8 step. Expect the separable pair
at about 10.7 ms per 16 frames on the Thor against 15.6 ms for PyTorch's own
ops and 37 ms for the fused one-pass kernel; the numbers and the reasons are
in the case study README.

## 10. Keep the record honest

Commit the JSON records, the environment record and the friction log
together. The README table is generated from the records
(`plot_results.py --table`) and a test fails if the README and the records
disagree. Do not quote a number that is not in a committed file, and label
anything synthetic as such. What this guide does not cover, because the case
study did not do it: quantization, batching, CUDA graphs, TensorRT, Alpamayo
1.5, and any accuracy claim beyond the single-sample trajectory error the
runner prints as a sanity figure.
