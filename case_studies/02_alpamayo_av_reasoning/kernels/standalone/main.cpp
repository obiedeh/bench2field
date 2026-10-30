// Standalone driver for the Alpamayo preprocessing kernels: no PyTorch, no
// Python, only the CUDA toolkit.
//
// Reads raw uint8 frames (N x 3 x 1080 x 1920 bytes, planar, as written by
// `bench_preprocess.py --dump-frames`), builds the resampling tables in C++,
// runs the chosen kernel on the device, times it with CUDA events, writes the
// float32 output (N * 720 x 1536) and the tables to files, and prints one JSON
// line with the timings. `compare_standalone.py` checks the output and the
// tables against the PyTorch extension bit for bit.
//
// Usage:
//   preprocess_standalone <frames.u8> <N> <variant: fused|separable> <out.f32>
//                         [reps=50] [warmup=5] [tables_prefix]
//
// Every CUDA call is checked; a failure prints the call and the error string
// and exits non-zero, so a partial result is never mistaken for a good one.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#include "../preprocess.h"
#include "tables.h"

using namespace alpamayo_pre;

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err__ = (call);                                                           \
        if (err__ != cudaSuccess) {                                                           \
            std::fprintf(stderr, "CUDA error: %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                                          \
            std::exit(2);                                                                     \
        }                                                                                     \
    } while (0)

namespace {

std::vector<uint8_t> read_file(const std::string& path, std::size_t expect) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    std::vector<uint8_t> buf(expect);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(expect));
    if (static_cast<std::size_t>(f.gcount()) != expect) {
        std::fprintf(stderr, "%s holds %lld bytes, expected %zu\n", path.c_str(), static_cast<long long>(f.gcount()), expect);
        std::exit(2);
    }
    return buf;
}

template <typename T>
void write_file(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); std::exit(2); }
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}

template <typename T>
T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

double percentile(std::vector<double> xs, double p) {
    std::sort(xs.begin(), xs.end());
    const std::size_t idx = static_cast<std::size_t>(std::lround(p * (xs.size() - 1)));
    return xs[std::min(idx, xs.size() - 1)];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <frames.u8> <N> <fused|separable> <out.f32> [reps] [warmup] [tables_prefix]\n", argv[0]);
        return 1;
    }
    const std::string frames_path = argv[1];
    const int n = std::atoi(argv[2]);
    const std::string variant = argv[3];
    const std::string out_path = argv[4];
    const int reps = argc > 5 ? std::atoi(argv[5]) : 50;
    const int warmup = argc > 6 ? std::atoi(argv[6]) : 5;
    const std::string tables_prefix = argc > 7 ? argv[7] : "";
    if (n < 1 || (variant != "fused" && variant != "separable")) {
        std::fprintf(stderr, "N must be >= 1 and variant fused or separable\n");
        return 1;
    }

    // Input and tables on the host.
    const std::size_t frame_bytes = static_cast<std::size_t>(n) * CHANNELS * IN_H * IN_W;
    const std::vector<uint8_t> frames = read_file(frames_path, frame_bytes);
    const AxisTable ty = bicubic_aa_table(IN_H, OUT_H, MAX_TAPS);
    const AxisTable tx = bicubic_aa_table(IN_W, OUT_W, MAX_TAPS);
    if (!tables_prefix.empty()) {
        write_file(tables_prefix + "_y_start.i32", ty.start);
        write_file(tables_prefix + "_y_weight.f32", ty.weight);
        write_file(tables_prefix + "_x_start.i32", tx.start);
        write_file(tables_prefix + "_x_weight.f32", tx.weight);
    }

    // Device buffers.
    uint8_t* d_frames = upload(frames);
    int* d_ys = upload(ty.start);
    float* d_yw = upload(ty.weight);
    int* d_xs = upload(tx.start);
    float* d_xw = upload(tx.weight);
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, output_bytes(n)));
    float* d_tmp = nullptr;
    if (variant == "separable") CUDA_CHECK(cudaMalloc(&d_tmp, separable_tmp_bytes(n)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto launch = [&]() {
        if (variant == "fused") {
            launch_preprocess_fused(d_frames, d_ys, d_yw, d_xs, d_xw, d_out, n, stream);
        } else {
            launch_preprocess_separable(d_frames, d_ys, d_yw, d_xs, d_xw, d_tmp, d_out, n, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    };

    // Warm-up, then timed repetitions with events on the same stream.
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    std::vector<double> ms;
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(e0, stream));
        launch();
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, e0, e1));
        ms.push_back(t);
    }

    // Output back to the host and to disk.
    std::vector<float> out(static_cast<std::size_t>(n) * PATCHES_PER_IMAGE * ROW_LEN);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, output_bytes(n), cudaMemcpyDeviceToHost));
    write_file(out_path, out);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int rt = 0, drv = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    std::printf("{\"variant\": \"%s\", \"images\": %d, \"reps\": %d, \"warmup\": %d, "
                "\"median_ms\": %.4f, \"p95_ms\": %.4f, \"min_ms\": %.4f, \"max_ms\": %.4f, "
                "\"device\": \"%s\", \"compute_capability\": [%d, %d], \"cuda_runtime\": %d, \"cuda_driver\": %d}\n",
                variant.c_str(), n, reps, warmup,
                percentile(ms, 0.5), percentile(ms, 0.95), *std::min_element(ms.begin(), ms.end()),
                *std::max_element(ms.begin(), ms.end()), prop.name, prop.major, prop.minor, rt, drv);

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_yw));
    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_xw));
    CUDA_CHECK(cudaFree(d_out));
    if (d_tmp) CUDA_CHECK(cudaFree(d_tmp));
    return 0;
}
