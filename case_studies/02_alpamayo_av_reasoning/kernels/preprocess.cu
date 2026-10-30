// Alpamayo camera preprocessing as CUDA kernels: a fused version and a
// separable version of the same work.
//
// What the model's processor does to each camera frame before the vision tower
// sees it (Qwen2VLImageProcessorFast, as configured by NVlabs/alpamayo):
//
//   1. resize 1080x1920 uint8 RGB to 320x576 with antialiased bicubic filtering
//      (torchvision F.resize, interpolation BICUBIC, antialias True), then round
//      back to uint8 because the input was uint8;
//   2. rescale by 1/255 and normalize with mean 0.5, std 0.5, so a pixel p
//      becomes (p/255 - 0.5)/0.5 = p/127.5 - 1 in float32;
//   3. cut the image into 16x16 patches, group them 2x2 (merge_size 2),
//      duplicate the frame along a temporal axis of length 2, and lay every
//      patch out as one row of 3 x 2 x 16 x 16 = 1536 floats.
//
// Steps 1 to 3 are three or four passes over memory in PyTorch. Two kernels
// here do the same work, and they are kept side by side on purpose:
//
//   preprocess_fused_kernel   one pass. Each thread produces one output pixel
//                             of one channel from a 14 x 14 window of input
//                             bytes and writes it straight into the patch
//                             tensor, twice (once per temporal slot). It is the
//                             version to read first: one kernel, one idea.
//                             Measured, it loses to PyTorch's own ops on both
//                             boards, because every thread gathers 196 bytes.
//
//   hpass_kernel, vpass_kernel
//                             two passes. The first resamples every input row
//                             horizontally (1080 rows x 576 columns per plane,
//                             14 taps each) into a float32 buffer. The second
//                             resamples that buffer vertically (14 taps),
//                             rounds, normalizes and scatters into the patch
//                             tensor. 28 taps per output value instead of 196,
//                             coalesced reads in both passes, one extra buffer
//                             of 1080 x 576 floats per plane in between.
//
// Resampling in both versions is written as two small weight tables, one per
// axis, computed on the host exactly the way torch computes them for
// antialiased bicubic downscaling (preprocess_ext.py, bicubic_aa_table; the
// same code in C++ in standalone/tables.cpp). A thread reads the taps for its
// output row or column and the resample is a plain sum. Taps whose weight is
// zero are padding; their input index is clamped to the plane so no read ever
// leaves the buffer, and they add nothing.
//
// Thread and block layout
//   fused   block (32, 8); grid (ceil(576/32), ceil(320/8), images * 3): one z
//           slice per (image, channel) pair, so blockIdx.z picks the input
//           plane. A warp covers 32 horizontally adjacent output pixels of one
//           row of one channel; their 14-wide input windows overlap (stride 3.4
//           input pixels), so a warp's byte loads hit the same cache lines and
//           go through __ldg, the read-only path.
//   hpass   block (32, 8); grid (ceil(576/32), ceil(1080/8), images * 3): one
//           thread per (input row, output column). A warp reads about 140
//           contiguous bytes of one input row and writes 32 contiguous floats.
//   vpass   block (32, 8); grid (ceil(576/32), ceil(320/8), images * 3): one
//           thread per output pixel. A warp reads 32 contiguous floats from
//           each of 14 rows of the buffer (fully coalesced) and writes the
//           patch tensor like the fused kernel does.
//
// Output writes (both versions): row = patch index, column = position inside
// the patch. 32 adjacent x pixels map to two 16-float runs in two different
// patch rows, so a warp writes 2 x 64-byte contiguous segments per temporal
// slot. That is partial coalescing; the output is 71 MB of float32 for 16
// images and this is the price of writing straight into patch order.
//
// Index mapping (the only part that needs care; it mirrors the processor's
// view/permute/reshape, see preprocess_ext.py, reference_layout):
//   pixel (y, x) in the 320 x 576 image, channel c, temporal slot t in {0, 1}
//   block row  bh = y / 32,  merge row  mh = (y % 32) / 16,  py = y % 16
//   block col  bw = x / 32,  merge col  mw = (x % 32) / 16,  px = x % 16
//   patch row  = img * 720 + ((bh * 18 + bw) * 2 + mh) * 2 + mw
//   column     = ((c * 2 + t) * 16 + py) * 16 + px
//
// Everything is specialised to this case study's fixed geometry on purpose: the
// constants are the model's and the host side asserts them before launching.
// A general kernel would hide the mapping behind arithmetic the reader would
// have to undo again.
//
// This file holds device code and the two launch functions only. The PyTorch
// binding is preprocess_torch.cpp; the standalone driver is standalone/main.cpp.

#include "preprocess.h"

namespace alpamayo_pre {

namespace {

// Round the resampled value the way torchvision does for uint8 input (float
// resize, round half to even, clamp) and apply the processor's rescale and
// normalize: (q / 255 - 0.5) / 0.5 = q / 127.5 - 1.
__device__ __forceinline__ float quantize_normalize(float acc) {
    const float q = fminf(fmaxf(rintf(acc), 0.f), 255.f);
    return q * (1.f / 127.5f) - 1.f;
}

// Write one normalized value to its two places (temporal slots) in the patch
// tensor. See the index mapping in the header.
__device__ __forceinline__ void scatter(float* __restrict__ out, int img, int c, int y, int x, float v) {
    const int bh = y / (PATCH * MERGE), mh = (y % (PATCH * MERGE)) / PATCH, py = y % PATCH;
    const int bw = x / (PATCH * MERGE), mw = (x % (PATCH * MERGE)) / PATCH, px = x % PATCH;
    const std::size_t patch_row = static_cast<std::size_t>(img) * PATCHES_PER_IMAGE
                                + ((static_cast<std::size_t>(bh) * (GRID_W / MERGE) + bw) * MERGE + mh) * MERGE + mw;
    float* dst = out + patch_row * ROW_LEN;
    #pragma unroll
    for (int t = 0; t < TEMPORAL; ++t) {
        dst[((c * TEMPORAL + t) * PATCH + py) * PATCH + px] = v;
    }
}

// Fused: one thread per output value, the full 2-D window gathered as bytes.
__global__ void preprocess_fused_kernel(
    const uint8_t* __restrict__ frames,      // [N][3][1080][1920]
    const int*     __restrict__ y_start,     // [320]  first input row of each output row's window
    const float*   __restrict__ y_weight,    // [320][MAX_TAPS], zero padded
    const int*     __restrict__ x_start,     // [576]
    const float*   __restrict__ x_weight,    // [576][MAX_TAPS]
    float*         __restrict__ out)         // [N * 720][1536]
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;   // output column
    const int y = blockIdx.y * blockDim.y + threadIdx.y;   // output row
    if (x >= OUT_W || y >= OUT_H) return;
    const int img = blockIdx.z / CHANNELS;
    const int c   = blockIdx.z % CHANNELS;

    const uint8_t* plane = frames + (static_cast<std::size_t>(img) * CHANNELS + c) * IN_H * IN_W;
    const int ys = y_start[y];
    const int xs = x_start[x];
    const float* wy = y_weight + y * MAX_TAPS;
    const float* wx = x_weight + x * MAX_TAPS;

    float acc = 0.f;
    #pragma unroll 1
    for (int j = 0; j < MAX_TAPS; ++j) {
        const float wyj = wy[j];
        if (wyj == 0.f) continue;                                   // padding rows: skip the whole row
        const uint8_t* row = plane + static_cast<std::size_t>(min(ys + j, IN_H - 1)) * IN_W;
        float racc = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_TAPS; ++i) {
            racc += wx[i] * static_cast<float>(__ldg(row + min(xs + i, IN_W - 1)));   // padding taps add 0
        }
        acc += wyj * racc;
    }
    scatter(out, img, c, y, x, quantize_normalize(acc));
}

// Separable, pass 1: every input row resampled horizontally into float32.
__global__ void hpass_kernel(
    const uint8_t* __restrict__ frames,      // [N][3][1080][1920]
    const int*     __restrict__ x_start,     // [576]
    const float*   __restrict__ x_weight,    // [576][MAX_TAPS]
    float*         __restrict__ tmp)         // [N * 3][1080][576]
{
    const int x  = blockIdx.x * blockDim.x + threadIdx.x;  // output column
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;  // input row
    if (x >= OUT_W || iy >= IN_H) return;
    const std::size_t plane = blockIdx.z;                   // img * 3 + c
    const uint8_t* row = frames + plane * IN_H * IN_W + static_cast<std::size_t>(iy) * IN_W;
    const int xs = x_start[x];
    const float* wx = x_weight + x * MAX_TAPS;
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < MAX_TAPS; ++i) {
        acc += wx[i] * static_cast<float>(__ldg(row + min(xs + i, IN_W - 1)));
    }
    tmp[(plane * IN_H + iy) * OUT_W + x] = acc;
}

// Separable, pass 2: the buffer resampled vertically, then rounded,
// normalized and scattered. Reads of `tmp` are contiguous across a warp.
__global__ void vpass_kernel(
    const float* __restrict__ tmp,           // [N * 3][1080][576]
    const int*   __restrict__ y_start,       // [320]
    const float* __restrict__ y_weight,      // [320][MAX_TAPS]
    float*       __restrict__ out)           // [N * 720][1536]
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;   // output column
    const int y = blockIdx.y * blockDim.y + threadIdx.y;   // output row
    if (x >= OUT_W || y >= OUT_H) return;
    const int img = blockIdx.z / CHANNELS;
    const int c   = blockIdx.z % CHANNELS;
    const float* plane = tmp + static_cast<std::size_t>(blockIdx.z) * IN_H * OUT_W;
    const int ys = y_start[y];
    const float* wy = y_weight + y * MAX_TAPS;
    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < MAX_TAPS; ++j) {
        acc += wy[j] * __ldg(plane + static_cast<std::size_t>(min(ys + j, IN_H - 1)) * OUT_W + x);
    }
    scatter(out, img, c, y, x, quantize_normalize(acc));
}

}  // namespace

void launch_preprocess_fused(const uint8_t* frames,
                             const int* y_start, const float* y_weight,
                             const int* x_start, const float* x_weight,
                             float* out, int n, cudaStream_t stream)
{
    const dim3 block(32, 8);
    const dim3 grid((OUT_W + block.x - 1) / block.x, (OUT_H + block.y - 1) / block.y, n * CHANNELS);
    preprocess_fused_kernel<<<grid, block, 0, stream>>>(frames, y_start, y_weight, x_start, x_weight, out);
}

void launch_preprocess_separable(const uint8_t* frames,
                                 const int* y_start, const float* y_weight,
                                 const int* x_start, const float* x_weight,
                                 float* tmp, float* out, int n, cudaStream_t stream)
{
    const dim3 block(32, 8);
    const dim3 grid_h((OUT_W + block.x - 1) / block.x, (IN_H + block.y - 1) / block.y, n * CHANNELS);
    hpass_kernel<<<grid_h, block, 0, stream>>>(frames, x_start, x_weight, tmp);
    const dim3 grid_v((OUT_W + block.x - 1) / block.x, (OUT_H + block.y - 1) / block.y, n * CHANNELS);
    vpass_kernel<<<grid_v, block, 0, stream>>>(tmp, y_start, y_weight, out);
}

}  // namespace alpamayo_pre
