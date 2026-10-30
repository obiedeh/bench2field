"""Build, run and check the fused CUDA preprocessing kernel for Alpamayo 1.

This module is the Python side of ``kernels/preprocess.cu``. It does four
things, each small enough to read in one sitting:

``bicubic_aa_table``
    Builds the per-axis resampling tables the kernel consumes. They are the
    same numbers torch computes inside ``_upsample_bicubic2d_aa`` for an
    antialiased downscale: for every output index the first input index of
    its window and up to ``MAX_TAPS`` normalized weights of the cubic filter
    (a = -0.5) stretched by the scale factor. Writing them here, in twenty
    lines of plain Python, keeps the kernel free of filter arithmetic and
    makes the resampling testable on a CPU.

``load_kernel``
    Compiles the extension with ``torch.utils.cpp_extension.load`` into a
    build directory inside the case study (ignored by git) using the venv's
    own toolchain. Nothing is installed system wide. The 5090 host needs a
    CUDA 12.x nvcc next to torch 2.8.0+cu128 and the Thor uses the JetPack
    toolkit; SETUP.md has the exact recipe per board.

``preprocess``
    Runs the kernel on a ``[N, 3, 1080, 1920]`` uint8 CUDA tensor and returns
    the processor's ``pixel_values`` layout, ``[N * 720, 1536]`` float32.

``reference_torch`` and ``reference_processor``
    Two independent references for the same tensor. ``reference_torch`` is
    the processor's recipe written out in PyTorch ops on the GPU
    (torchvision antialiased bicubic resize, rescale, normalize, the
    view/permute/reshape into patches). ``reference_processor`` is the real
    Hugging Face processor Alpamayo calls, run on the CPU exactly as the
    model's helper runs it. ``bench_preprocess.py`` reports the maximum
    absolute error of the kernel against both and of the two references
    against each other, so a disagreement can be placed.

Geometry is fixed to Alpamayo's camera setting (four cameras, four frames,
1080x1920 in, 320x576 out). The constants below are asserted against the
processor's own ``image_grid_thw`` in the tests.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
KERNEL_SRC = HERE / "kernels" / "preprocess.cu"
BUILD_DIR = HERE / "runs" / "scratch" / "kernel_build"   # ignored by git

IN_H, IN_W = 1080, 1920
OUT_H, OUT_W = 320, 576          # smart_resize(1080, 1920, factor=32, 163840, 196608)
PATCH, MERGE, TEMPORAL, CHANNELS = 16, 2, 2, 3
GRID_H, GRID_W = OUT_H // PATCH, OUT_W // PATCH            # 20, 36
PATCHES_PER_IMAGE = GRID_H * GRID_W                        # 720
ROW_LEN = CHANNELS * TEMPORAL * PATCH * PATCH              # 1536
MAX_TAPS = 16
MEAN = STD = 0.5
RESCALE = 1.0 / 255.0


def cubic(x: float, a: float = -0.5) -> float:
    """The bicubic convolution kernel torch and PIL use (Keys, a = -0.5)."""
    x = abs(x)
    if x < 1.0:
        return (a + 2.0) * x**3 - (a + 3.0) * x**2 + 1.0
    if x < 2.0:
        return a * x**3 - 5.0 * a * x**2 + 8.0 * a * x - 4.0 * a
    return 0.0


def bicubic_aa_table(in_size: int, out_size: int, max_taps: int = MAX_TAPS) -> tuple[torch.Tensor, torch.Tensor]:
    """Start index and normalized weights per output index for an antialiased
    bicubic downscale, as torch's ``compute_indices_weights_aa`` computes them.

    Antialiasing stretches the filter by the scale factor so that every input
    pixel under an output pixel contributes; without it, bicubic at 3.4x
    downscale would sample four of every fourteen input pixels and alias.
    """
    scale = in_size / out_size
    support = 2.0 * scale                      # bicubic has support 2 on each side
    invscale = 1.0 / scale
    starts = torch.zeros(out_size, dtype=torch.int32)
    weights = torch.zeros(out_size, max_taps, dtype=torch.float32)
    for i in range(out_size):
        center = scale * (i + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        n = xmax - xmin
        if n > max_taps:
            raise ValueError(f"{n} taps exceed MAX_TAPS={max_taps}; raise MAX_TAPS in the kernel")
        w = [cubic((j + xmin - center + 0.5) * invscale) for j in range(n)]
        total = sum(w)
        starts[i] = xmin
        for j in range(n):
            weights[i, j] = w[j] / total if total != 0.0 else 0.0
    return starts, weights


def tables(device: torch.device | str) -> dict[str, torch.Tensor]:
    ys, yw = bicubic_aa_table(IN_H, OUT_H)
    xs, xw = bicubic_aa_table(IN_W, OUT_W)
    return {"y_start": ys.to(device), "y_weight": yw.to(device), "x_start": xs.to(device), "x_weight": xw.to(device)}


_ext: Any = None


def load_kernel(verbose: bool = False) -> Any:
    """Compile (once) and return the extension module.

    Respects ``CUDA_HOME`` and ``TORCH_CUDA_ARCH_LIST`` from the environment,
    which is how each board's setup script points the build at the right
    toolkit and architecture (12.0 for the 5090, 11.0 for the Thor).
    """
    global _ext
    if _ext is not None:
        return _ext
    import torch.utils.cpp_extension as cpp

    if os.environ.get("ALPAMAYO_RELAX_CUDA_VERSION_CHECK") == "1":
        # The 5090 host has only a CUDA 13.2 nvcc next to a torch built for
        # CUDA 12.8, and no 12.x nvcc exists as a pip wheel. torch refuses a
        # major version mismatch outright. With the check relaxed, the 13.2
        # compiler builds the kernel and the extension links its own
        # libcudart.so.13 beside torch's libcudart.so.12; both talk to the same
        # driver context, and the bench verifies the output bit for bit. This
        # is an explicit opt-in recorded in SETUP.md and the friction log, not a
        # default. The Thor needs no such thing: its toolkit matches its torch.
        def _relaxed(compiler_name: str, compiler_version: str) -> None:
            print(f"[build] CUDA version check relaxed: torch {torch.version.cuda}, nvcc from {os.environ.get('CUDA_HOME')}", file=sys.stderr)
        cpp._check_cuda_version = _relaxed  # type: ignore[assignment]

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    _ext = cpp.load(
        name="alpamayo_preprocess",
        sources=[str(KERNEL_SRC)],
        build_directory=str(BUILD_DIR),
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )
    return _ext


def preprocess(frames: torch.Tensor, tabs: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    """Kernel path: uint8 [N, 3, 1080, 1920] on the GPU to float32 [N * 720, 1536]."""
    ext = load_kernel()
    tabs = tabs or tables(frames.device)
    return ext.preprocess(frames.contiguous(), tabs["y_start"], tabs["y_weight"], tabs["x_start"], tabs["x_weight"])


def reference_layout(images: torch.Tensor) -> torch.Tensor:
    """The processor's patch layout for float images [N, 3, 320, 576]: duplicate
    along the temporal axis, then view/permute/reshape exactly as
    Qwen2VLImageProcessorFast does (the permute order is copied from it)."""
    n = images.shape[0]
    p = images.unsqueeze(1).repeat(1, TEMPORAL, 1, 1, 1)                     # [N, 2, 3, H, W]
    p = p.view(n, 1, TEMPORAL, CHANNELS, GRID_H // MERGE, MERGE, PATCH, GRID_W // MERGE, MERGE, PATCH)
    p = p.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return p.reshape(n * PATCHES_PER_IMAGE, ROW_LEN)


def reference_torch(frames: torch.Tensor) -> torch.Tensor:
    """The same recipe in PyTorch ops on the frames' device."""
    from torchvision.transforms.v2 import functional as F

    resized = F.resize(frames, [OUT_H, OUT_W], interpolation=F.InterpolationMode.BICUBIC, antialias=True)  # uint8 in, uint8 out
    x = resized.to(torch.float32) * RESCALE
    x = (x - MEAN) / STD
    return reference_layout(x)


_processor: Any = None


def processor() -> Any:
    """The processor Alpamayo's helper builds, created once. Building it loads
    the tokenizer and is not part of what the benchmark times."""
    global _processor
    if _processor is None:
        from alpamayo_r1 import helper
        from transformers import AutoTokenizer

        _processor = helper.get_processor(AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct"))
    return _processor


def reference_processor(frames: torch.Tensor) -> torch.Tensor:
    """The real processor on the CPU, as Alpamayo's helper calls it (images only)."""
    out = processor().image_processor(images=list(frames.cpu()), return_tensors="pt")
    return out["pixel_values"]


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().max())
