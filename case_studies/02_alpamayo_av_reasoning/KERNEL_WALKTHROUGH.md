# The preprocessing kernel, explained for a walk-through

This is the plain-language companion to `kernels/preprocess.cu` and
`preprocess_ext.py`. It is written so the author can explain every line to
someone who has not seen the code. Numbers quoted here come from
`runs/preprocess_<board>.json` and are repeated in the README.

## What problem the kernel solves

Before Alpamayo 1 looks at a camera frame, the Hugging Face processor does
three things to it: shrinks it from 1080x1920 to 320x576, turns the 0 to 255
bytes into floats between -1 and 1, and chops the picture into 16x16 patches
laid out in the exact order the vision transformer expects. One decision
uses sixteen frames (four cameras, four moments in time), so that is sixteen
pictures of about six million bytes each, every time the car thinks.

In PyTorch this is several passes: a resize that reads the bytes and writes
floats, a rescale and normalize that read and write the floats again, and a
reshape and permute that copy them one more time into patch order. Each pass
goes to memory and back. The kernel does all of it in one pass: every thread
reads the bytes it needs, computes one output value, and writes it directly
to its final address in the patch tensor. Nothing is stored in between.

## The three pieces

**Resampling.** Shrinking an image by 3.4x properly means every output pixel
is a weighted average of about fourteen input pixels in each direction, with
bicubic weights stretched by the scale factor. That stretching is what
"antialias" means in torchvision; without it the resize would sample four of
every fourteen pixels and produce jagged edges. The weights depend only on
the output row or column, not on the image, so they are computed once in
Python (`bicubic_aa_table`, twenty lines that mirror torch's own formula)
and handed to the kernel as two small tables: for each of the 320 output
rows, where its window starts in the 1080 input rows and up to sixteen
weights; the same for the 576 output columns. The kernel then does a plain
double loop, rows times columns, multiply and add. That is the whole resize.

**Normalize.** torchvision resizes a uint8 image in float and rounds the
result back to a byte. The processor then divides by 255, subtracts 0.5 and
divides by 0.5. The kernel does the same: round, clamp to 0 to 255, then
`q / 127.5 - 1`. Reproducing the rounding is what makes the kernel agree with
the real processor to the last bit instead of to within one grey level.

**Layout.** The vision tower wants each image as 720 rows of 1536 numbers.
A row is one 16x16 patch, with all three colour channels and two "temporal"
copies of the same frame (the processor duplicates single images so video
and image inputs share one path). The 720 patches are ordered so that each
2x2 group of neighbouring patches is contiguous, because the model merges
them. Working out where pixel (y, x) of channel c lands is the only
arithmetic in the kernel that needs care, and it is written out in the
header comment:

    block row  bh = y / 32,  merge row  mh = (y % 32) / 16,  py = y % 16
    block col  bw = x / 32,  merge col  mw = (x % 32) / 16,  px = x % 16
    patch row  = img * 720 + ((bh * 18 + bw) * 2 + mh) * 2 + mw
    column     = ((c * 2 + t) * 16 + py) * 16 + px

`tests/test_alpamayo_kernel.py` checks this mapping against the processor's
own view, permute and reshape, pixel by pixel, on the CPU.

## How the work is split across the GPU

A block is 32 by 8 threads: a warp (32 threads) covers 32 neighbouring output
pixels of one row, eight rows per block. The grid has one slice per image and
channel, so `blockIdx.z` says which input plane a block reads; 16 images times
3 channels is 48 slices of 18 by 40 blocks, 345,600 threads, one per output
value. Each thread's input window overlaps its neighbours' almost entirely
(the windows are fourteen wide, the stride between neighbours is 3.4), so the
bytes a warp reads come from the same few cache lines; the loads use `__ldg`,
the read-only path, which is the right choice for data that is never
written. The output writes are less tidy: 32 neighbouring pixels land in two
runs of sixteen floats in two different patch rows. Two 64-byte segments per
warp per temporal copy is partial coalescing, and it is the price of writing
straight into patch order. The input is 100 MB of bytes that must be read
once; the output is 71 MB of floats. The kernel is bound by reading, so the
imperfect writes were accepted rather than adding a shared-memory transpose
that would double the code.

## Design choices, and the ones not taken

- **Fused, not separable.** A separable resize (all rows first, then all
  columns) does about three times fewer multiplies. It also needs an
  intermediate buffer of 1080x576 floats per channel and a second kernel.
  The fused form is one kernel and one idea; the arithmetic is cheap next to
  the memory traffic on both GPUs, which the timings bear out or refute
  (see the README). Chosen for readability, measured honestly.
- **Tables, not on-the-fly weights.** Computing cubic weights inside the
  thread is possible but buries the filter mathematics in device code. The
  tables keep the kernel's inner loop to a multiply and add, and let a CPU
  test compare the tables against torchvision directly.
- **Fixed geometry.** The constants are the model's. A general kernel would
  turn every index into arithmetic the reader has to undo. If the camera
  resolution changes, `smart_resize` gives a new output size and the
  constants change in one place, with the asserts catching a mismatch.
- **float32 out.** The processor emits float32 and the model casts to bf16
  later. Emitting bf16 from the kernel would halve the write traffic and is
  the first thing to try if this kernel ever sits on the critical path.

## What the measurements say

Measured on both boards (`runs/preprocess_*.json`, 50 timed runs after 5
warm-ups, 16 frames per run): the kernel takes 3.29 ms on the 5090 and
37.3 ms on the Thor; PyTorch's own GPU ops take 1.24 ms and 15.6 ms.
PyTorch wins by about 2.5x on both. The kernel is exact against the
processor's GPU path (one rounding step on 0.004 percent of values). The
processor on the CPU, which is what Alpamayo runs, takes 366 ms on the
5090 host and 37 ms on the Thor, and its output differs from the GPU
path by up to five grey levels on a tenth of the values.

Why PyTorch wins here is the most useful thing to be able to explain: the
fused kernel pays for its simplicity with 14 x 14 taps per output value,
about 1.7 billion byte gathers per batch, where a separable resize pays
about a third of that. The arithmetic was supposed to be cheap next to the
memory traffic; on these GPUs it is not, because the gathers are bytes and
the L1 cannot hide 196 of them per thread. The separable two-kernel version
is the next experiment, not a claim.

## What the measurements can and cannot say

The benchmark times the kernel with the frames already on the GPU, the kernel
including the host to device copy of the bytes, the same recipe in PyTorch
ops on the GPU, and the real processor on the CPU, which is what Alpamayo
calls today. Against the CPU processor the kernel is not a like for like
contest, it is a statement about where the work could move. Against the
PyTorch GPU recipe it is the honest comparison: same device, same output,
one kernel versus several. If PyTorch wins, the README says so.

## How to build it without touching either machine

The 5090 host runs torch 2.8.0 built for CUDA 12.8 but only ships a CUDA
13.2 nvcc, and the pip wheel named nvcc for 12.8 contains only `ptxas`.
SETUP.md records what was done about it. The Thor's JetPack 7.1 toolkit
matches its torch and builds the extension in about 35 seconds with ninja
installed into the venv. Builds go to `runs/scratch/kernel_build/`, which git
ignores.
