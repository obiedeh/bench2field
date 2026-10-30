// Alpamayo camera preprocessing kernels: the C++ host interface.
//
// This header is shared by the PyTorch binding (preprocess_torch.cpp), the
// standalone driver (standalone/main.cpp) and the device code (preprocess.cu).
// It holds the fixed geometry of the case study and the two launch functions.
// Nothing here depends on PyTorch, so the kernels can be built and run with
// only the CUDA toolkit.
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace alpamayo_pre {

// Geometry of the model's camera input, from the processor's own settings.
constexpr int IN_H = 1080, IN_W = 1920;          // camera frame
constexpr int OUT_H = 320, OUT_W = 576;          // smart_resize of 1080x1920 at the processor's MIN/MAX_PIXELS
constexpr int PATCH = 16, MERGE = 2, TEMPORAL = 2, CHANNELS = 3;
constexpr int GRID_H = OUT_H / PATCH;            // 20
constexpr int GRID_W = OUT_W / PATCH;            // 36
constexpr int PATCHES_PER_IMAGE = GRID_H * GRID_W;                   // 720
constexpr int ROW_LEN = CHANNELS * TEMPORAL * PATCH * PATCH;         // 1536
constexpr int MAX_TAPS = 16;                     // antialiased bicubic at 3.375x needs 14 taps per axis

// Bytes of the intermediate buffer the separable pair needs for n images:
// float32 [n * 3][IN_H][OUT_W], rows resampled horizontally, not yet vertically.
inline std::size_t separable_tmp_bytes(int n) {
    return static_cast<std::size_t>(n) * CHANNELS * IN_H * OUT_W * sizeof(float);
}

// Bytes of the output for n images: float32 [n * PATCHES_PER_IMAGE][ROW_LEN].
inline std::size_t output_bytes(int n) {
    return static_cast<std::size_t>(n) * PATCHES_PER_IMAGE * ROW_LEN * sizeof(float);
}

// One kernel: resample, normalize and scatter in a single pass (the first,
// readable version). frames: uint8 [n][3][IN_H][IN_W] on the device.
// y_start/x_start: int32 [OUT_H]/[OUT_W], first input index of each window.
// y_weight/x_weight: float32 [OUT_H][MAX_TAPS]/[OUT_W][MAX_TAPS], zero padded.
// out: float32 [n * PATCHES_PER_IMAGE][ROW_LEN]. Launches on `stream`; the
// caller checks cudaGetLastError afterwards.
void launch_preprocess_fused(const uint8_t* frames,
                             const int* y_start, const float* y_weight,
                             const int* x_start, const float* x_weight,
                             float* out, int n, cudaStream_t stream);

// Two kernels: a horizontal pass into `tmp` (separable_tmp_bytes(n) bytes),
// then a vertical pass that also normalizes and scatters into `out`.
// Same arguments otherwise.
void launch_preprocess_separable(const uint8_t* frames,
                                 const int* y_start, const float* y_weight,
                                 const int* x_start, const float* x_weight,
                                 float* tmp, float* out, int n, cudaStream_t stream);

}  // namespace alpamayo_pre
