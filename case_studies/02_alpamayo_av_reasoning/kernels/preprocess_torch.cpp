// PyTorch binding for the Alpamayo preprocessing kernels: the C++ host side.
//
// This translation unit is compiled by the host C++ compiler, not nvcc. It
// checks every input the way a library entry point should (device, dtype,
// contiguity, exact shapes, matching devices), allocates the output and the
// intermediate buffer through PyTorch's caching allocator so they are reused
// across calls, launches on the stream PyTorch is currently using so the
// kernel is ordered correctly with the surrounding PyTorch work, and turns a
// launch error into a Python exception.
//
// Exposed to Python as `alpamayo_preprocess.preprocess(frames, y_start,
// y_weight, x_start, x_weight, variant)`, with variant "fused" or
// "separable". preprocess_ext.py wraps it.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <string>

#include "preprocess.h"

namespace {

using namespace alpamayo_pre;

void check_table(const torch::Tensor& t, const char* name, int rows, torch::ScalarType dtype, const torch::Device& dev) {
    TORCH_CHECK(t.device() == dev, name, " must be on the same device as frames");
    TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    if (dtype == torch::kFloat32) {
        TORCH_CHECK(t.dim() == 2 && t.size(0) == rows && t.size(1) == MAX_TAPS, name, " must be [", rows, ", ", MAX_TAPS, "]");
    } else {
        TORCH_CHECK(t.dim() == 1 && t.size(0) == rows, name, " must be [", rows, "]");
    }
}

}  // namespace

// frames: uint8 CUDA tensor [N, 3, 1080, 1920], contiguous.
// Returns float32 [N * 720, 1536], the processor's pixel_values for these frames.
torch::Tensor preprocess(torch::Tensor frames,
                         torch::Tensor y_start, torch::Tensor y_weight,
                         torch::Tensor x_start, torch::Tensor x_weight,
                         const std::string& variant)
{
    TORCH_CHECK(frames.is_cuda(), "frames must be a CUDA tensor");
    TORCH_CHECK(frames.scalar_type() == torch::kUInt8, "frames must be uint8");
    TORCH_CHECK(frames.is_contiguous(), "frames must be contiguous");
    TORCH_CHECK(frames.dim() == 4 && frames.size(1) == CHANNELS && frames.size(2) == IN_H && frames.size(3) == IN_W,
                "frames must be [N, 3, 1080, 1920]; the kernels are specialised to the camera geometry");
    const int n = static_cast<int>(frames.size(0));
    TORCH_CHECK(n >= 1, "frames must hold at least one image");
    const torch::Device dev = frames.device();
    check_table(y_start, "y_start", OUT_H, torch::kInt32, dev);
    check_table(y_weight, "y_weight", OUT_H, torch::kFloat32, dev);
    check_table(x_start, "x_start", OUT_W, torch::kInt32, dev);
    check_table(x_weight, "x_weight", OUT_W, torch::kFloat32, dev);
    TORCH_CHECK(variant == "fused" || variant == "separable", "variant must be 'fused' or 'separable', got '", variant, "'");

    const c10::cuda::CUDAGuard guard(dev);
    auto out = torch::empty({static_cast<int64_t>(n) * PATCHES_PER_IMAGE, ROW_LEN}, frames.options().dtype(torch::kFloat32));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (variant == "fused") {
        launch_preprocess_fused(frames.data_ptr<uint8_t>(),
                                y_start.data_ptr<int>(), y_weight.data_ptr<float>(),
                                x_start.data_ptr<int>(), x_weight.data_ptr<float>(),
                                out.data_ptr<float>(), n, stream);
    } else {
        auto tmp = torch::empty({static_cast<int64_t>(n) * CHANNELS, IN_H, OUT_W}, frames.options().dtype(torch::kFloat32));
        launch_preprocess_separable(frames.data_ptr<uint8_t>(),
                                    y_start.data_ptr<int>(), y_weight.data_ptr<float>(),
                                    x_start.data_ptr<int>(), x_weight.data_ptr<float>(),
                                    tmp.data_ptr<float>(), out.data_ptr<float>(), n, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess", &preprocess,
          "Alpamayo camera preprocessing: resize, normalize and patch layout on the GPU",
          pybind11::arg("frames"), pybind11::arg("y_start"), pybind11::arg("y_weight"),
          pybind11::arg("x_start"), pybind11::arg("x_weight"), pybind11::arg("variant") = "separable");
}
