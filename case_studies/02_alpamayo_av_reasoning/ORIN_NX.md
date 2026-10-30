# Orin NX: why Alpamayo 1 is not run on the field computer

The rover's computer is a Jetson Orin NX 8 GB on JetPack 6. Alpamayo 1 does
not fit it, in memory, on disk or in the Python it needs. This note records
the measured facts behind that decision so it can be revisited when the board
or the model changes. Captured read-only on 2026-10-29; raw output in
`runs/facts_field-orin.txt`.

## Measured facts

| Fact | Value | Source |
|---|---|---|
| Module | NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super | `/proc/device-tree/model` |
| Memory | 7619 MiB total, 4630 MiB available at capture, 2599 MiB in use by the desktop and the rover's services | `free -m` |
| Swap | 12001 MiB, zram (compressed RAM, so it borrows from the same 7.6 GiB) | `free -m` |
| Disk | 117 GB root, 6.6 GB free (95% used) | `df -h /` |
| Jetson Linux | R36.4.7 (JetPack 6 line) | `/etc/nv_tegra_release`, `nvidia-l4t-core` |
| CUDA | 12.6 | `/usr/local/cuda-12.6` |
| Python | 3.10.12 is the only interpreter installed | `python3 --version`, `/usr/bin/python3.1*` |
| torch | 2.5.0a0 (NVIDIA build nv24.08) on the system Python | `torch.__version__` |
| Jetson torch wheels for this JetPack | 2.8.0, 2.9.1, 2.10.0, 2.11.0, all `cp310` only | Jetson AI Lab jp6/cu126 index, checked 2026-10-29 |
| bitsandbytes wheels for this JetPack | 0.47.0.dev0, 0.48.0.dev0, `cp310` only | same index |

## What Alpamayo 1 needs

- The bf16 checkpoint is 22.2 GB in five safetensors shards (Hugging Face file
  sizes for `nvidia/Alpamayo-R1-10B`).
- NVlabs/alpamayo requires Python 3.12 exactly (`requires-python = "==3.12.*"`)
  and `torch==2.8.0`.
- The upstream README asks for a GPU with at least 24 GB.

## Why it does not fit

**Disk.** 6.6 GB free cannot hold the 22.2 GB download, nor a quantized copy
made elsewhere once the venv is added.

**Memory, calculated, not measured.** Dividing the 22.2 GB bf16 checkpoint by
the bits per weight gives the floor each format would need before the CUDA
context, the vision encoder's activations and the key-value cache: 11.1 GB at
8 bits, 5.55 GB at 4 bits. These are lower bounds: real schemes keep
embeddings, norms and parts of the diffusion head in higher precision. Against
4630 MiB available on a 7619 MiB module, the 8-bit floor is above the whole
board and the 4-bit floor is above what is free. Swap is zram, so it does not
add memory.

**Python.** Alpamayo 1 needs 3.12; the board has 3.10 and every Jetson torch
wheel for its JetPack is built for 3.10. Running the model would mean either a
3.12 interpreter with no CUDA torch wheel, or a 3.10 interpreter with a
dependency pin the upstream code does not support.

No speed figure is given: nothing ran, so there is nothing to measure.

## What would change the answer

A 16 GB Orin NX module with free disk, JetPack 7.2 (Python 3.12, CUDA 13) and
a checkpoint quantized off-board would make the 4-bit question worth asking
again. That is a different board from the rover's.
