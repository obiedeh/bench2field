# The preprocessing kernels, explained for a walk-through

This is the plain-language companion to `kernels/preprocess.cu`,
`kernels/preprocess.h`, `kernels/preprocess_torch.cpp`,
`kernels/standalone/` and `preprocess_ext.py`. It is written so the author
can explain every line to someone who has not seen the code. Numbers quoted
here come from `runs/preprocess_<board>.json` and are repeated in the README.

## What problem the kernels solve

Before Alpamayo 1 looks at a camera frame, the Hugging Face processor does
three things to it: shrinks it from 1080x1920 to 320x576, turns the 0 to 255
bytes into floats between -1 and 1, and chops the picture into 16x16 patches
laid out in the exact order the vision transformer expects. One decision
uses sixteen frames (four cameras, four moments in time), so that is sixteen
pictures of about six million bytes each, every time the car thinks.

In PyTorch this is several passes: a resize that reads the bytes and writes
floats, a rescale and normalize that read and write the floats again, and a
reshape and permute that copy them one more time into patch order. Each pass
goes to memory and back. The kernels do the same work in one or two passes
and write every value straight to its final address in the patch tensor.

## The three pieces of the work

**Resampling.** Shrinking an image by 3.4x properly means every output pixel
is a weighted average of about fourteen input pixels in each direction, with
bicubic weights stretched by the scale factor. That stretching is what
"antialias" means in torchvision; without it the resize would sample four of
every fourteen pixels and produce jagged edges. The weights depend only on
the output row or column, not on the image, so they are computed once on the
host (`bicubic_aa_table` in Python, and the same twenty lines in C++ in
`standalone/tables.cpp`) and handed to the kernel as two small tables: for
each of the 320 output rows, where its window starts in the 1080 input rows
and up to sixteen weights; the same for the 576 output columns. Taps beyond
the window carry a zero weight, and their input index is clamped to the
plane so no read ever leaves the buffer.

**Normalize.** torchvision resizes a uint8 image in float and rounds the
result back to a byte. The processor then divides by 255, subtracts 0.5 and
divides by 0.5. The kernels do the same: round, clamp to 0 to 255, then
`q / 127.5 - 1`. Reproducing the rounding is what makes the kernels agree
with the real processor to the last grey level instead of to within one.

**Layout.** The vision tower wants each image as 720 rows of 1536 numbers.
A row is one 16x16 patch, with all three colour channels and two "temporal"
copies of the same frame (the processor duplicates single images so video
and image inputs share one path). The 720 patches are ordered so that each
2x2 group of neighbouring patches is contiguous, because the model merges
them. Working out where pixel (y, x) of channel c lands is the only
arithmetic that needs care, and it is written out in the kernel header:

    block row  bh = y / 32,  merge row  mh = (y % 32) / 16,  py = y % 16
    block col  bw = x / 32,  merge col  mw = (x % 32) / 16,  px = x % 16
    patch row  = img * 720 + ((bh * 18 + bw) * 2 + mh) * 2 + mw
    column     = ((c * 2 + t) * 16 + py) * 16 + px

`tests/test_alpamayo_kernel.py` checks this mapping against the processor's
own view, permute and reshape, pixel by pixel, on the CPU. Both kernels
share one device function, `scatter`, that does it.

## Kernel one: fused, one pass

One thread per output value. A block is 32 by 8 threads: a warp (32
threads) covers 32 neighbouring output pixels of one row, eight rows per
block. The grid has one slice per image and channel, so `blockIdx.z` says
which input plane a block reads; 16 images times 3 channels is 48 slices of
18 by 40 blocks, 345,600 threads. Each thread loops over its 14 input rows
and, for each, its 14 input columns, multiplying bytes by weights: 196 loads
and multiply-adds per output value. The loads go through `__ldg`, the
read-only path. Then it rounds, normalizes and scatters.

It was written first because it is one kernel and one idea, and it was
measured honestly: 3.29 ms per 16 frames on the 5090 and 37.3 ms on the
Thor, against 1.23 ms and 15.6 ms for PyTorch's own ops. The profiler says
why it loses: the byte gathers are the cost. The windows of neighbouring
threads overlap almost entirely, so the bytes come from cache most of the
time, but 196 single-byte loads per thread are still 196 instructions each
waiting on the memory system, and the arithmetic that was supposed to be
cheap next to the memory traffic turns out to be the traffic.

## Kernel two: separable, two passes

The same resize, split the way every image library does it: first every
input row is resampled horizontally (1080 rows times 576 output columns per
plane, 14 taps each) into a float32 buffer of 1080 x 576 per plane; then
every output pixel is resampled vertically from that buffer (14 taps),
rounded, normalized and scattered. 28 taps per output value instead of 196.

`hpass_kernel`: one thread per (input row, output column), a 32 by 8 block,
48 slices of 18 by 135 blocks. A warp reads about 140 contiguous bytes of
one input row and writes 32 contiguous floats. `vpass_kernel`: one thread
per output pixel, the same grid as the fused kernel; a warp reads 32
contiguous floats from each of 14 rows of the buffer, which is the access
pattern GPUs like best, and writes the patch tensor like the fused kernel
does. The buffer costs 119 MB for 16 frames and one extra launch.

Measured: 0.65 ms on the 5090 and 10.7 ms on the Thor, faster than PyTorch
by 1.9x and 1.5x, and bit-identical to the fused kernel's output on both
boards (same tables, same rounding, same scatter, only the order of the
float sums differs, and the records show it makes no difference at the
grey level). Under the profiler the horizontal pass is nearly all of the
time (0.75 of 0.86 ms on the 5090, 9.5 of 10.6 ms of kernel time on the Thor): it still
gathers single bytes, with neighbouring threads 3.3 input pixels apart.
Reading the row once into shared memory, or loading bytes four or sixteen
at a time, is the next step, and it is not claimed.

## The host side, in C++

`kernels/preprocess.h` holds the fixed geometry and declares two launch
functions; `kernels/preprocess.cu` defines the kernels and those two
functions and nothing else, so the device code has no dependency on PyTorch.

`kernels/preprocess_torch.cpp` is the PyTorch binding, compiled by the host
C++ compiler. It is what a library entry point should be: it checks that the
frames are a contiguous uint8 CUDA tensor of exactly `[N, 3, 1080, 1920]`,
that the four tables have the right dtype, shape and device, and that the
variant name is one of the two; it allocates the output and, for the
separable pair, the intermediate buffer through PyTorch's caching allocator
so the memory is reused across calls; it launches on the stream PyTorch is
currently using so the kernel is ordered correctly with the surrounding
PyTorch work; and it turns a launch error into a Python exception with
`C10_CUDA_KERNEL_LAUNCH_CHECK`. `preprocess_ext.py` compiles the two files
together with `torch.utils.cpp_extension.load` and exposes
`preprocess(frames, tables, variant)`.

`kernels/standalone/` builds the same kernels with CMake and the CUDA
toolkit alone. `main.cpp` reads raw frames from a file, builds the tables
with the C++ version of the table code, allocates device memory with the
runtime API, runs the chosen variant on its own stream, times it with CUDA
events, writes the output and the tables to files and prints one JSON line.
Every CUDA call is checked, and a failure exits non-zero so a partial result
is never mistaken for a good one. `compare_standalone.py` then loads the
same frames into the extension and requires the two outputs to be equal
bit for bit and the two sets of tables to be equal exactly; they are, on
both boards. That is the check that the C++ host side and the Python host
side drive the same kernel the same way.

## Design choices, and the ones not taken

- **Two kernels kept side by side.** The fused one is the one to read; the
  separable one is the one to use. Keeping both, with the measurement that
  separates them, is the point of the exercise.
- **Tables, not on-the-fly weights.** Computing cubic weights inside the
  thread is possible but buries the filter mathematics in device code. The
  tables keep the inner loops to a multiply and add, let a CPU test compare
  the tables against torchvision directly, and let the C++ and Python table
  code be checked against each other.
- **Fixed geometry.** The constants are the model's. A general kernel would
  turn every index into arithmetic the reader has to undo. If the camera
  resolution changes, `smart_resize` gives a new output size and the
  constants change in one header, with the checks catching a mismatch.
- **float32 out.** The processor emits float32 and the model casts to bf16
  later. Emitting bf16 from the kernel would halve the write traffic and is
  the first thing to try if this path ever sits on the critical path.
- **Not taken:** shared-memory row tiles, vectorised byte loads, a
  single-kernel separable version with the buffer in shared memory, CUDA
  graphs. Each is a plausible next step; none is measured, so none is
  claimed.

## What the measurements say

From `runs/preprocess_*.json`, 50 timed runs after 5 warm-ups, 16 frames
per run, real frames from the first clip:

| | 5090 | Thor |
|---|---|---|
| Fused kernel | 3.29 ms | 37.29 ms |
| Separable pair | 0.65 ms | 10.74 ms |
| PyTorch ops on the GPU | 1.23 ms | 15.63 ms |
| Processor on the CPU, as Alpamayo runs it | 318.7 ms | 35.1 ms |

Both kernels are exact against the processor's GPU path up to one rounding
tie in about 25,000 values. The processor on the CPU differs from its own
GPU path by up to five grey levels on a tenth of the values, so the model
already sees slightly different pixels depending on where the preprocessing
runs; the kernels match the GPU path.

The 5090 was shared with an unrelated simulation process during these runs,
which the record states; the kernel and the PyTorch path were timed back to
back under the same conditions, so their ratio stands. The Thor was idle.

## What the measurements can and cannot say

The benchmark times each kernel with the frames already on the GPU, each
kernel including the host to device copy of the bytes, the same recipe in
PyTorch ops on the GPU, and the real processor on the CPU, which is what
Alpamayo calls today. Against the CPU processor the kernels are not a like
for like contest, they are a statement about where the work could move.
Against the PyTorch GPU recipe they are the honest comparison: same device,
same output, one or two kernels versus eleven. The fused kernel lost that
comparison and the README said so; the separable pair wins it. Neither
number is a claim about Alpamayo's decision latency, which is dominated by
the language rollout, not the preprocessing.

## How to build them without touching either machine

The 5090 host runs torch 2.8.0 built for CUDA 12.8 but only ships a CUDA
13.2 nvcc, and the pip wheel named nvcc for 12.8 contains only `ptxas`.
SETUP.md records what was done about it. The Thor's JetPack 7.1 toolkit
matches its torch and builds the extension in about 35 seconds with ninja
installed into the venv. The standalone driver needs CMake 3.24 or newer and
the toolkit's nvcc named explicitly, because a distribution nvcc on PATH is
an older toolkit. Builds go to `runs/scratch/`, which git ignores.
