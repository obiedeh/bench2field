// Alpamayo camera preprocessing as one CUDA kernel.
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
// Steps 1 to 3 are three or four passes over memory in PyTorch. This kernel does
// all of them in a single pass: each thread produces one output pixel of one
// channel and writes it straight into its final place in the patch tensor,
// twice (once per temporal slot). Nothing intermediate is stored.
//
// Resampling is written as two small weight tables, one per axis, computed on
// the host exactly the way torch computes them for antialiased bicubic
// downscaling (see preprocess_ext.py, bicubic_aa_table). A thread reads the
// taps for its output row and column, and the resample is a plain double sum
// over at most MAX_TAPS x MAX_TAPS input pixels. Separable filtering would do
// fewer multiplies; the fused form was chosen because it is the easiest to
// read and verify, and the arithmetic is not the bottleneck on either GPU
// (the input is 100 MB of uint8 that has to be read once either way).
//
// Thread and block layout
//   block   = (32, 8) threads: 32 along the output x axis, 8 along y.
//   grid    = (ceil(576/32), ceil(320/8), images * 3): one z slice per
//             (image, channel) pair, so blockIdx.z picks the input plane.
//   A warp therefore covers 32 horizontally adjacent output pixels of one row
//   of one channel. Their input windows overlap (each output pixel spans about
//   3.4 input pixels per axis, the window is about 14 wide), so the uint8 loads
//   of one warp hit the same cache lines and L1 serves most of them; the loads
//   go through __ldg to use the read-only path.
//
// Memory access pattern
//   Reads:  input plane [1080][1920] uint8, row-major; a thread reads a
//           ~14 x 14 window; warps read ~14 rows of ~140 contiguous bytes.
//   Writes: output row = patch index, column = position inside the patch.
//           32 adjacent x pixels map to two 16-float runs in two different
//           patch rows, so a warp writes 2 x 64-byte contiguous segments per
//           temporal slot. That is partial coalescing; the output is small
//           (71 MB float32 for 16 images) and the kernel is read-bound.
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
// constants are the model's and the kernel asserts them at launch. A general
// kernel would hide the mapping behind arithmetic the reader would have to
// undo again.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace {

constexpr int IN_H = 1080, IN_W = 1920;     // camera frame
constexpr int OUT_H = 320, OUT_W = 576;     // smart_resize of 1080x1920 at MIN/MAX_PIXELS
constexpr int PATCH = 16, MERGE = 2, TEMPORAL = 2, CHANNELS = 3;
constexpr int GRID_H = OUT_H / PATCH;       // 20
constexpr int GRID_W = OUT_W / PATCH;       // 36
constexpr int PATCHES_PER_IMAGE = GRID_H * GRID_W;                       // 720
constexpr int ROW_LEN = CHANNELS * TEMPORAL * PATCH * PATCH;             // 1536
constexpr int MAX_TAPS = 16;                // bicubic support 2 * scale 3.375 = 6.75 each side: 14 taps

__global__ void preprocess_kernel(
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

    const uint8_t* plane = frames + ((size_t)img * CHANNELS + c) * IN_H * IN_W;
    const int ys = y_start[y];
    const int xs = x_start[x];
    const float* wy = y_weight + y * MAX_TAPS;
    const float* wx = x_weight + x * MAX_TAPS;

    // Antialiased bicubic resample: sum over the window with separable weights.
    // Rows whose weight is zero (padding) are skipped, so the loop does the
    // same work as torch's variable-length window.
    float acc = 0.f;
    #pragma unroll 1
    for (int j = 0; j < MAX_TAPS; ++j) {
        const float wyj = wy[j];
        if (wyj == 0.f) continue;
        const uint8_t* row = plane + (size_t)(ys + j) * IN_W + xs;
        float racc = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_TAPS; ++i) {
            racc += wx[i] * (float)__ldg(row + i);   // zero weights read a valid in-window byte and add nothing
        }
        acc += wyj * racc;
    }

    // torchvision resizes uint8 input in float and rounds back to uint8; the
    // processor then rescales and normalizes. Reproduce both so the comparison
    // against the real processor is exact up to summation order.
    const float q = fminf(fmaxf(rintf(acc), 0.f), 255.f);
    const float v = q * (1.f / 127.5f) - 1.f;

    // Scatter into the patch layout (see header), once per temporal slot.
    const int bh = y / (PATCH * MERGE), mh = (y % (PATCH * MERGE)) / PATCH, py = y % PATCH;
    const int bw = x / (PATCH * MERGE), mw = (x % (PATCH * MERGE)) / PATCH, px = x % PATCH;
    const size_t patch_row = (size_t)img * PATCHES_PER_IMAGE
                           + (((size_t)bh * (GRID_W / MERGE) + bw) * MERGE + mh) * MERGE + mw;
    float* dst = out + patch_row * ROW_LEN;
    #pragma unroll
    for (int t = 0; t < TEMPORAL; ++t) {
        dst[((c * TEMPORAL + t) * PATCH + py) * PATCH + px] = v;
    }
}

}  // namespace

// frames: uint8 CUDA tensor [N, 3, 1080, 1920], contiguous.
// Returns float32 [N * 720, 1536], the processor's pixel_values for these frames.
torch::Tensor preprocess(torch::Tensor frames,
                         torch::Tensor y_start, torch::Tensor y_weight,
                         torch::Tensor x_start, torch::Tensor x_weight)
{
    TORCH_CHECK(frames.is_cuda() && frames.dtype() == torch::kUInt8 && frames.is_contiguous(), "frames: contiguous uint8 CUDA tensor");
    TORCH_CHECK(frames.dim() == 4 && frames.size(1) == CHANNELS && frames.size(2) == IN_H && frames.size(3) == IN_W,
                "frames must be [N, 3, 1080, 1920]; this kernel is specialised to the camera geometry");
    TORCH_CHECK(y_weight.size(0) == OUT_H && y_weight.size(1) == MAX_TAPS && x_weight.size(0) == OUT_W && x_weight.size(1) == MAX_TAPS,
                "weight tables must be [320, 16] and [576, 16]");
    const int n = (int)frames.size(0);
    auto out = torch::empty({(int64_t)n * PATCHES_PER_IMAGE, ROW_LEN}, frames.options().dtype(torch::kFloat32));
    const dim3 block(32, 8);
    const dim3 grid((OUT_W + block.x - 1) / block.x, (OUT_H + block.y - 1) / block.y, n * CHANNELS);
    preprocess_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        frames.data_ptr<uint8_t>(),
        y_start.data_ptr<int>(), y_weight.data_ptr<float>(),
        x_start.data_ptr<int>(), x_weight.data_ptr<float>(),
        out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess", &preprocess, "Alpamayo camera preprocessing: resize, normalize and patch layout in one pass");
}
