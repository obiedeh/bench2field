# Setup notes: Alpamayo 1 for case study 02

Exact, reproducible steps for each board. The friction behind them is in
[FRICTION_LOG.md](FRICTION_LOG.md); the versions that resulted are in
`runs/env_<board>.json`.

## What gets installed, and where

Everything lives in `case_studies/02_alpamayo_av_reasoning/.venv`, which git
ignores. bench2field's own `.venv` at the repository root is not touched: it
carries torch 2.11 for the case study 01 export, and Alpamayo 1 pins torch 2.8.0
exactly. Weights go to the Hugging Face cache (`HF_HOME`, 22.2 GB for
`nvidia/Alpamayo-R1-10B`); dataset clips are streamed per clip id by
`physical_ai_av` into the same cache. Nothing heavy is written inside the
repository.

Upstream pins (NVlabs/alpamayo at commit `11a0e01`): Python 3.12.x, torch 2.8.0,
transformers 4.57.1, flash-attn 2.8.3 or newer, physical_ai_av 0.2.0 or newer.

## 5090 host (`bench-5090`)

Prerequisites: `uv` on PATH, an NVIDIA driver at or above what CUDA 12.8 needs
(the host has 580.196.07), about 5 GB of disk for the venv and 23 GB for the
weights under `HF_HOME`.

```bash
cd ~/github/bench2field
bash case_studies/02_alpamayo_av_reasoning/setup_env.sh
env -u PYTHONPATH case_studies/02_alpamayo_av_reasoning/.venv/bin/python \
    case_studies/02_alpamayo_av_reasoning/capture_env.py --board bench-5090
```

The script builds the venv, installs torch 2.8.0 (cu128), the prebuilt
flash-attn 2.8.3 wheel for torch 2.8 / CPython 3.12 / `cxx11abiTRUE`,
transformers 4.57.1, `alpamayo_r1` from the pinned commit and bench2field
itself (editable, for provenance and telemetry), then verifies the imports
and that CUDA is visible. On 2026-10-29 it took 73 s end to end.

`env -u PYTHONPATH` is not optional on this host: the login shell sources
ROS 2 Jazzy, whose `PYTHONPATH` takes precedence over any venv.

Before phase 2, once only, as the owner:

```bash
hf auth login          # the model is not gated; the dataset is (terms accepted on the Hugging Face page)
```

## Thor (`bench-thor`, Jetson AGX Thor developer kit, JetPack 7.1)

Done on 2026-10-29; environment record in `runs/env_bench-thor.json`. This is
Jetson AGX Thor on JetPack, not DRIVE AGX Thor on DriveOS. Nothing outside the
home directory changed: no JetPack, apt, driver or `nvpmodel` change (120W mode,
as found).

```bash
# on the 5090 host: copy the repository (no .git, venvs, data or models) to the Thor,
# with the source commit in a file so the Thor's records can name it
cd ~/github/bench2field && git rev-parse HEAD > SOURCE_COMMIT
rsync -a --exclude .git --exclude '.venv' --exclude '*/.venv' --exclude data --exclude models \
      --exclude .trt_cache --exclude __pycache__ ./ thor:~/alpamayo-work/bench2field/ && rm SOURCE_COMMIT
# on the Thor
bash ~/alpamayo-work/bench2field/case_studies/02_alpamayo_av_reasoning/setup_env_thor.sh
~/.venv/alpamayo/bin/python ~/alpamayo-work/bench2field/case_studies/02_alpamayo_av_reasoning/capture_env.py --board bench-thor
```

What differs from the 5090, and why:

- `torch 2.11.0+cu130` instead of the upstream `torch==2.8.0`: 2.8.0 has no
  wheel for CUDA 13 on aarch64. The 2.11.0 wheel is the `manylinux_2_28_aarch64`
  build from the PyTorch cu130 index, the same one the board's other venvs use.
  The environment record names it; every Thor result carries this difference.
- No flash-attn; `attn_implementation="sdpa"`, which the upstream README supports.
  `alpamayo_r1` is installed with `--no-deps` so pip never tries to build it.
- Weights: the Hugging Face download stalled on this board (name resolution),
  so the 21 GB were copied from the 5090's cache with rsync into the same cache
  layout. The loader reads it unchanged.
- Power comes from tegrastats (`VIN`, reported as `power_board_w`), the board
  rail PLAN.md fixed before the first run. The Thor's NVML does answer power,
  utilisation and temperature queries, but for the GPU rail only, and it has no
  memory or clock channels; GPU memory comes from torch's own counters.

Setup took 329 s, 263 s of it the torch download.

## Building the preprocessing kernel (phase 3b)

The kernel in `kernels/preprocess.cu` is compiled on first use by
`preprocess_ext.load_kernel()` with `torch.utils.cpp_extension.load`, into
`runs/scratch/kernel_build/` (ignored by git). No system package changes on
either board; `ninja` is installed into each venv.

Thor: the JetPack toolkit matches the torch build (both CUDA 13), so

```bash
uv pip install --python ~/.venv/alpamayo/bin/python ninja
cd ~/alpamayo-work/bench2field/case_studies/02_alpamayo_av_reasoning
PATH=$HOME/.venv/alpamayo/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST=11.0 \
  ~/.venv/alpamayo/bin/python bench_preprocess.py --board bench-thor
```

`ninja` must be on `PATH` as an executable; importing the Python package is
not enough for torch's check. Build time about 40 s.

5090 host: torch 2.8.0+cu128 wants a CUDA 12.x nvcc and the host only has
13.2. The pip package named `nvidia-cuda-nvcc-cu12` (12.8.93) was tried first
and contains only `ptxas` and `nvPTXCompiler.h`, no compiler front end, on
PyPI and on NVIDIA's own index alike. The recorded path is therefore the
13.2 compiler with torch's major-version check relaxed, as an explicit
opt-in:

```bash
uv pip install --python .venv/bin/python ninja
ALPAMAYO_RELAX_CUDA_VERSION_CHECK=1 CUDA_HOME=/usr/local/cuda-13.2 TORCH_CUDA_ARCH_LIST=12.0 \
  env -u PYTHONPATH .venv/bin/python bench_preprocess.py --board bench-5090
```

What that means: the extension is compiled for sm_120 by nvcc 13.2 and links
`libcudart.so.13`, which then lives in the same process as torch's
`libcudart.so.12`. Both runtimes use the same driver context and stream
handle; the benchmark checks the kernel's output against the processor
element by element, so a broken interaction would show as a wrong answer, not
a silent one. The clean alternative, a user-local CUDA 12.8 toolkit from the
runfile installer (about 4 GB, no root needed with a custom path), was not
taken because the opt-in is smaller and the check covers it. Build time
about 25 s.

## Building the standalone C++ driver (phase 3b, no PyTorch)

`kernels/standalone/` builds the same kernels with CMake and the CUDA
toolkit alone: `main.cpp` reads raw frames, builds the resampling tables in
C++ (`tables.cpp`, the same arithmetic as `preprocess_ext.bicubic_aa_table`),
runs a variant, times it with CUDA events and writes the output.
`compare_standalone.py` then checks that output and those tables against the
extension. The compiler is named explicitly because the distribution `nvcc`
on PATH is an older toolkit than the one that built the extension.

```bash
# frames for the driver: bench_preprocess.py --dump-frames writes them (16 x 3 x 1080 x 1920 bytes)
# 5090 host
cmake -S kernels/standalone -B runs/scratch/standalone_build -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
# Thor
cmake -S kernels/standalone -B runs/scratch/standalone_build -DCMAKE_CUDA_ARCHITECTURES=110 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
# both
cmake --build runs/scratch/standalone_build
runs/scratch/standalone_build/preprocess_standalone runs/scratch/frames_clip0.u8 16 separable \
      runs/scratch/standalone_separable.f32 50 5 runs/scratch/standalone_tables
<venv python> kernels/standalone/compare_standalone.py runs/scratch/frames_clip0.u8 16 separable \
      runs/scratch/standalone_separable.f32 runs/scratch/standalone_tables
```

The driver prints one JSON line with the timings; the comparison exits
non-zero unless the output matches the extension exactly and the C++ tables
match the Python tables exactly. Build time a few seconds on either board.

## Orin NX (`field-orin`)

Not run. The board is an 8 GB Orin NX on JetPack 6 with Python 3.10; see the
friction log for the memory and disk arithmetic.

## Checking an environment later

`capture_env.py` is idempotent. Re-run it after any change to the venv, and
commit the new record alongside the results that used it.
