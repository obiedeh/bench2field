# Kernels

Custom kernels live here, one folder per kernel, with the CUDA source, the HIP port, a correctness test against a reference implementation, and the profiler numbers that justify it.

## 01: fused preprocessing (case study 01, phase 3)

**Job:** turn a camera frame (uint8, HWC, BGR, arbitrary size) into the detector's input tensor (float16 or float32, NCHW, RGB, letterboxed to the model size, scaled to [0, 1]) in a single kernel launch.

**Why:** done in Python or as separate library calls, this is several passes over memory and often a host round trip per frame. One fused pass reads each source pixel once and writes each output value once.

**Spec**
- One thread per output pixel; bilinear sampling from the source; constant fill for letterbox borders.
- BGR→RGB swap, scale, and dtype cast in registers before the store.
- Output written planar (C, H, W) so stores are coalesced per channel.
- Launch on a caller-provided stream.

**Done when**
- Max absolute error against an OpenCV + NumPy reference ≤ 1/255 on the rover test frames.
- Nsight Compute shows achieved DRAM bandwidth, with a note on why it is or isn't near the device peak.
- End-to-end frame latency with the kernel is measured with `b2f run` against the unfused baseline.
- The HIP port (`hipify-perl`) builds and passes the same test on MI300X, with rocprof numbers.

Write this kernel yourself rather than adapting a published one. It is the part of the project an interviewer will ask you to walk through line by line.
