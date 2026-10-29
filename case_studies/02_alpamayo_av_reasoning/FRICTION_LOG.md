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
