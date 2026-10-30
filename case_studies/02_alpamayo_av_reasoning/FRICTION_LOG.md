# Developer friction log: Alpamayo 1 on the 5090 host, the Thor and the Orin NX

What it took to get NVIDIA Alpamayo 1 running outside its reference setup, in the
order it happened. Times are wall clock on the day. Each entry says what broke,
how it was found and what fixed it, so the next person skips the search.
Entries are added as the case study moves through its phases; nothing is
rewritten afterwards (rule 6 in AGENTS.md).

## 2026-10-29, phase 1: review and environment, 5090 host

| Time | Entry |
|---|---|
| about 25 min | Reading before the first command: NVlabs/alpamayo README and pyproject, the alpamayo1.5 and alpamayo-recipes READMEs, the model card, bench2field's own layout and rules, and a read-only probe of the Thor and the Orin NX. Not a failure, but it is part of the real setup time. |
| 58 s | `uv venv -p 3.12` plus `torch==2.8.0` and `torchvision==0.23.0` from the cu128 index. The pins come from NVlabs/alpamayo `pyproject.toml`; the host driver is 580.196.07 with a CUDA 13.2 toolkit, and the cu128 wheel runs on it because the driver is newer than the runtime it asks for. |
| 7 s | flash-attn. The PyPI package is a source distribution that compiles for a long time and needs `nvcc` from a CUDA 12 toolkit; this host has 13.2. Fix: the prebuilt wheel from the Dao-AILab release page for torch 2.8, CUDA 12, CPython 3.12, `cxx11abiTRUE` (PyTorch Linux wheels since 2.6 use the C++11 ABI, so the TRUE variant is the matching one). A forward call on a small bf16 tensor ran on compute capability 12.0 (the 5090), so the wheel does carry kernels this GPU can run. Whether it is faster than SDPA here is a phase 2 measurement, not an assumption. |
| 3 s | `transformers==4.57.1`, `alpamayo_r1` from git at commit `11a0e01`, `nvidia-ml-py`, `matplotlib`. `physical-ai-av` resolved to 0.2.3 (Alpamayo 1 asks for `>=0.2.0`; Alpamayo 1.5 pins `==0.2.0`). Recorded in `runs/env_bench-5090.json`. |
| 73 s | Total from `uv venv` to a verified import of `AlpamayoR1` and a flash-attn kernel on the GPU. |
| 0 s | `alpamayo_r1` installs as a namespace package: there is no `__init__.py`, so `alpamayo_r1.__file__` is `None`. Harmless for imports; anything that locates the package by `__file__` must use `__path__` instead. |
| 0 s | `uv venv` creates no `pip`; `pip freeze` in the venv prints nothing. Use `uv pip freeze --python .venv/bin/python`. |
| 1 min | `import alpamayo_r1.test_inference` is not an import, it is the demo: module-level code downloads the example clip and the 22 GB of weights. On this host it failed with a 401 on the dataset, because the dataset is gated and `hf auth login` had not been run. The model repository itself is not gated (the Hugging Face API reports `gated: false`), so weights need no approval; only the dataset does. Fix: the case study's runner never imports that module. The owner accepts the dataset terms and logs in before phase 2. |
| standing | The 5090 host's shell sources ROS 2 Jazzy, whose `PYTHONPATH` shadows any venv. Every command here is `env -u PYTHONPATH .venv/bin/...`; `setup_env.sh` does it internally. Same rule as case study 01. |
| standing | `b2f run` cannot drive this model: it times ONNX sessions at fixed request rates. Alpamayo is a PyTorch autoregressive VLM plus a diffusion trajectory head, so the case study has its own runner that writes JSON, and reuses `bench2field.provenance`, `bench2field.background` and the NVML sampler so the records look like every other b2f record. |

## Thor (Jetson AGX Thor developer kit, JetPack 7.1), planned for phase 4

Read-only findings; nothing was installed on the Thor in phase 1.

- Python 3.12.3 is the system interpreter, so the `==3.12.*` pin is met without a rebuild.
- `torch==2.8.0` cannot be met: the Thor runs CUDA 13 on aarch64 and the usable
  torch wheels are the Jetson builds (the existing GR00T venv on that board has
  `torch 2.11.0+cu130`). Phase 4 uses a Jetson torch wheel and records the
  deviation in its environment record. This is Jetson AGX Thor on JetPack, not
  DRIVE AGX Thor on DriveOS.
- No flash-attn: `attn_implementation="sdpa"`, which the upstream README supports.
- 70 GB free on the root disk against a 22 GB download. Enough, but the owner
  frees space first.
- JetPack 7.2.1 is current; the Thor stays on 7.1 for this work. Updating is a
  Jetson Linux 38 to 39 move and would invalidate every JetPack-built wheel and
  r38 container on the board.

## Orin NX (JetPack 6, 8 GB), phase 5 is documentation only

The brief assumed a 16 GB module. The board is an Orin NX 8 GB: 7.4 GiB visible,
12 GB zram swap, 6.6 GB free on a 117 GB disk, Python 3.10 only, CUDA 12.6.
Estimates, not measurements: the bf16 checkpoint is 22.2 GB; an 8-bit version
would be about 11.5 GB and a 4-bit version about 6.5 GB before the CUDA context,
the vision encoder's activations and the KV cache. With 2.5 GB already in use
by the desktop and the rover's services, neither fits, and the disk cannot hold
the download to quantize from. bitsandbytes development wheels do exist on the
Jetson AI Lab jp6/cu126 index, but memory and disk block before tooling matters.

## 2026-10-29, phase 2: the runner, 5090 host, and the Thor environment

| Time | Entry |
|---|---|
| 17 s | First load of the 22.2 GB checkpoint in bf16 on the 5090: 17.4 s from `from_pretrained` to the model on the GPU with the shards in the page cache, 20.6 GiB allocated after load. The upstream README's 24 GB requirement is right: the 5090's 32 GB leaves room for one decision, not for `num_traj_samples` above a few. |
| 10 min | The upstream call pops `input_ids` out of the tokenized dict it is given, so anything read from that dict after the call is gone. First smoke run crashed on the prompt token count after a successful decision. Fix: read the prompt length inside the `generate` hook, where the ids still exist. |
| 5 min | `TelemetrySampler.samples` is a property in bench2field, not a method; the runner called it. Found on the second smoke run, after the decisions had completed. Fix: attribute access. Both found by running on synthetic input before the dataset was available, which is what the `--synthetic` flag is for. |
| 0 s | Synthetic frames give a 15-token trace ("keep lane, nothing needs attention"), so synthetic latency says nothing about real clips, where traces are longer. Smoke numbers stay in `runs/scratch/` and are labelled as not a result. |
| 0 s | NVML on the 5090 reports the GPU rail only (`power_gpu_w`); the Thor's NVML answers no power, memory or clock query at all, as case study 01 found. The runner uses bench2field's NVML sampler where it answers and the tegrastats sampler (`VIN` as `power_board_w`) where it does not, and records which channel each power number came from. |
| 329 s | Thor venv (`~/.venv/alpamayo`, system Python 3.12.3): `torch==2.8.0` is not available for this stack, so the decision in PLAN.md stands: `torch 2.11.0+cu130`, the `manylinux_2_28_aarch64` wheel from the PyTorch cu130 index, the same one the board's other venvs use. 263 s of the 329 s was that download. CUDA 13.0 visible, compute capability 11.0, `transformers 4.57.1`, `alpamayo_r1` at `11a0e01` installed with `--no-deps` so pip does not try to build flash-attn. |
| 25 min | The weight download on the Thor stalled at 129 MB: pypi and Hugging Face name resolution failed intermittently from that board (`Temporary failure in name resolution` in the pip log, zero bytes in 10 s on the model shards). Fix: stop it and copy the 21 GB from the 5090's Hugging Face cache over the LAN with rsync into the same cache layout (`blobs/`, `snapshots/`, `refs/`), which the Hugging Face loader reads unchanged. |

## 2026-10-29, phase 4 preparation: Thor smoke run

| Time | Entry |
|---|---|
| 27 s | First load on the Thor: 26.6 s to the model on the device, 20.6 GiB allocated after load, same as the 5090 (same bf16 weights). A synthetic decision ran: 4.5 s for a 20-token trace, 9.8 tokens/s decode, 0.83 s trajectory sampling, 21.5 GiB peak. Smoke numbers, not results. |
| 15 min | Correction to the phase 2 entry above: the Thor's NVML is not silent. It answers power, utilisation and temperature (it reported 24.6 W mean on the GPU rail during the smoke decision), and only memory and clock queries are missing, as bench2field's NVML sampler already documents. The runner had picked NVML first on every board, so the Thor smoke recorded GPU-rail power instead of the VIN board rail PLAN.md fixed. Fix: the sampler is chosen by board (NVML on the 5090, tegrastats on the Thor) and every power number carries its channel name. The smoke record was discarded and the smoke rerun. |
| 0 s | The Thor runs from a copy of the tree without `.git`, so `bench2field.provenance.git_state()` finds no commit there. The copy step now writes the source commit to a `SOURCE_COMMIT` file at the tree root and the runner reads it when git is absent. |

## 2026-10-29, phase 2: the measured runs

| Time | Entry |
|---|---|
| 90 min | The dataset gate. The weights are ungated, the clips are not. Nothing in the pipeline could be measured until the owner accepted the dataset terms in a browser and placed a token on each board. Facts worth knowing before the next person hits it: one account token works on both boards (it is tied to the account, not the machine); accepting the terms is a separate step from logging in, and `HfApi().auth_check(repo_id, repo_type="dataset")` tells the two states apart in one call (403 "restricted" means no token or no terms, 403 "not in the authorized list" means token present, terms not accepted). Both error texts were seen tonight. |
| 79 s | Streaming the eight clips cold took 79 s on the 5090 and 141 s on the Thor (16 s per clip on the 5090 from the probe). The runner loads every clip before the first decision and keeps them on the host, so this cost is outside the decision timings and is recorded as `data_load_s`. |
| 0 s | The warm-up decision matters more than expected: the warm-up decision on clip `030c760c` took 1.77 s with a 0.96 s time to first token; the same clip measured right after it takes 0.94 s with 0.38 s to first token. The warm-up is recorded in the JSON under `warmup` and excluded from the statistics, as the protocol said. |
| 0 s | Seed 42 before each decision makes the three repeats of a clip produce the same tokens, the same trace and the same waypoints on a given board, so the repeats measure timing spread only, not output variance. Across boards the same seed does not give the same text: the Thor wrote 32 tokens where the 5090 wrote 35 on clip `06b483cf`, with the same meaning. Different kernels, different rounding, a different sample. The protocol stands; the reading of "repeats" is recorded in PLAN.md Amendments. |
| 45 min | The flash variant crashed in trajectory sampling, not in the rollout: `RuntimeError: cu_seqlens_q must have shape (batch_size + 1)` from `flash_attn_varlen_forward`, reached through the diffusion expert's step function (`alpamayo_r1/models/alpamayo_r1.py` step_fn, a Qwen3-VL decoder layer). Upstream's own demo, run unchanged on this venv with dataset access, completed with the same trace and minADE as the sdpa run, which looked like a contradiction until the attention in effect was read off the loaded model: `vlm: sdpa, expert: sdpa`. The hub config declares `attn_implementation: flash_attention_2`, but `AlpamayoR1Config.from_pretrained` on transformers 4.57.1 returns it as `None`, and `None` resolves to sdpa in both sub-models. So the documented default never runs flash-attn here, and the `flash-attn>=2.8.3` requirement is satisfied but unused by the demo. Forcing the field to `flash_attention_2` applies it to the expert too, which is where the varlen call fails. Fix in the runner: the expert keeps sdpa in every variant (`model.expert.set_attn_implementation("sdpa")` after load), the VLM takes the variant's attention, and the JSON records both. A one-clip probe passed; the full flash run followed. PLAN.md Amendments redefines the variant accordingly. |
| 0 s | The flash variant, once it ran, was not faster: median decision 0.93 s against 0.89 s for sdpa, decode 50.9 against 60.0 tokens per second, time to first token equal within 0.01 s, memory identical. The wheel is the cu12, torch 2.8 build of flash-attn 2.8.3; it runs on this sm_120 GPU but at batch size one with a short decode it loses to the sdpa kernels that ship with torch. Load was quicker in the flash run (8.3 s against 15.7 s) because it ran second with the shards in the page cache; load time is not a variant result. sdpa stays the baseline. |

## 2026-10-29, phase 3: the profile

| Time | Entry |
|---|---|
| 20 min | The first profile record's `top_kernels` and `top_ops` were the same list: `key_averages()` mixes host-side operators, device-side kernels and the `record_function` stage labels, and the first version only sorted it. The stage labels appeared twice (host and device side) at the top of both tables, and the share column divided by a total that counted the same microseconds three times. Fix: kernels are the entries whose device type is CUDA, operators are the `aten::` entries, stage labels are excluded from both (they live in `stages`), and each share is of its own table. The profile was rerun; the stage totals did not change. |
| 0 s | Profiler overhead on this decision is about 7 percent (1.01 s under the profiler against 0.94 s for the same clip in the measured run), with CUDA activity recording on. The stage shares are the result of phase 3; the latency result stays the unprofiled run. |
| 0 s | Every large kernel in the trace is an `sm80` or `cutlass_80` build: torch 2.8.0+cu128 has no Blackwell-specific GEMM or attention kernels, and the decode is a bf16 GEMV at batch one. This is the shape of the workload on the 5090: weight streaming per token, not compute. It also frames the flash-attn result: a wheel built for older architectures cannot be expected to beat torch's own kernels here. Recorded as an observation from the kernel table, not tested further. |

## 2026-10-29, phase 4: the Thor measured run

| Time | Entry |
|---|---|
| 230 s | Streaming the eight clips on the Thor took 141 s in the first run and 230 s in the run that is committed (the first run's record was replaced when the provenance block changed, see phase 2). Both are outside the decision timings (`data_load_s`). The Thor's path to the dataset is slower and noisier than the 5090's; the clips are not cached on disk between runs by this loader. |
| 0 s | `other_gpu_processes` on the Thor lists two holders of GPU memory as `unknown`: NVML names their pids but `/proc/<pid>/exe` is not readable for another user's process without privileges, and the runner does not escalate. Memory sizes are recorded (84 and 19 MiB, the display stack). |
| 0 s | The Thor smoke run in the phase 4 preparation recorded NVML GPU-rail power by mistake; the measured run records `power_board_w` from tegrastats `VIN`, and `channel` in every decision's power block says which it is. The README's power rows name the channel on each board and say they are not comparable. |

## 2026-10-29, phase 3b: the preprocessing kernel

| Time | Entry |
|---|---|
| 20 min | No CUDA 12.x compiler for the 5090's torch 2.8.0+cu128 without touching the system. The host has nvcc 13.2 only. The pip wheel `nvidia-cuda-nvcc-cu12==12.8.93` was installed into the venv together with the runtime and CCCL header wheels and a CUDA_HOME tree was assembled from them; the wheel turned out to contain `ptxas` and `nvPTXCompiler.h` and nothing else, on PyPI and on NVIDIA's index. Removed again. A trivial extension built with nvcc 13.2 and torch's major-version check relaxed ran correctly and linked its own `libcudart.so.13`; that became the documented opt-in (`ALPAMAYO_RELAX_CUDA_VERSION_CHECK=1`), with the full output comparison as the safety net. |
| 5 min | The Thor had the Python `ninja` package importable but torch's `verify_ninja_availability` wants the executable on `PATH`; an ssh command's PATH does not include the venv's `bin`. Fix: prefix `PATH` in the command. The JetPack toolkit (CUDA 13.0 and 13.2 both installed) matches torch 2.11.0+cu130, so no version relaxation is needed there. |
| 2 min | `--use_fast_math=false` is not an nvcc flag (fast math is a bare switch, on or off). The first Thor build failed on it; removed. Fast math stays off because the kernel reproduces torchvision's rounding and fast math would move values across the .5 boundary. |
| 10 min | The weight-table test first compared against torchvision's uint8 CPU path and found one grey level of difference on 11 percent of pixels. That path uses fixed-point weights; the GPU path, which the kernel is compared against, casts to float first. The test now resamples a float image, which isolates the tables from that implementation detail. Lesson for anyone verifying a resize kernel: say which torchvision path you are matching, the uint8 CPU one and the float one differ by design. |
| 0 s | Result: PyTorch wins on the GPU on both boards (1.24 against 3.29 ms on the 5090, 15.6 against 37.3 ms on the Thor). The fused 2D resample's 196 byte gathers per output value are the cost; a separable version is the next step and is not claimed. Recorded as measured, as the brief required. |
| 0 s | The first processor timing was wrong by 5x: `reference_processor` rebuilt the tokenizer and the processor on every call, so the 5090 host showed 1883 ms per batch. With the processor built once it is 366 ms. The first record was discarded and both boards rerun before anything was written down. Lesson: time the call you mean to time; construction belongs outside the loop. |
| 0 s | The processor's CPU and GPU paths disagree: up to five grey levels (0.039 after normalization) on 9.6 percent of values, because torchvision's uint8 CPU resize uses fixed-point weights and the GPU path casts to float. The kernel matches the GPU path to one rounding step on 0.004 percent of values. Alpamayo runs the CPU path, so the model's inputs depend on where preprocessing runs; whether that moves a trace is not measured here. |
| 0 s | `build_s` in the committed records is the cached build (0.28 s and 0.06 s); cold builds were about 25 s on the 5090 and 40 s on the Thor from the probe logs. The Thor's CPU processor time (37 ms, 14 cores, torchvision 0.26) is far below the 5090 host's (366 ms, 24 cores, torchvision 0.23); the torchvision versions differ and this is recorded, not explained. |
