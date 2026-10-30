// See tables.h. Mirrors preprocess_ext.bicubic_aa_table line for line so the
// two can be compared; keep them in step.
#include "tables.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace alpamayo_pre {

namespace {

// The bicubic convolution kernel torch and PIL use (Keys, a = -0.5).
double cubic(double x, double a = -0.5) {
    x = std::fabs(x);
    if (x < 1.0) return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
    if (x < 2.0) return a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a;
    return 0.0;
}

}  // namespace

AxisTable bicubic_aa_table(int in_size, int out_size, int max_taps) {
    const double scale = static_cast<double>(in_size) / out_size;
    const double support = 2.0 * scale;          // bicubic has support 2 on each side
    const double invscale = 1.0 / scale;
    AxisTable t;
    t.start.assign(out_size, 0);
    t.weight.assign(static_cast<std::size_t>(out_size) * max_taps, 0.0f);
    std::vector<double> w;
    for (int i = 0; i < out_size; ++i) {
        const double center = scale * (i + 0.5);
        const int xmin = std::max(static_cast<int>(center - support + 0.5), 0);
        const int xmax = std::min(static_cast<int>(center + support + 0.5), in_size);
        const int n = xmax - xmin;
        if (n > max_taps) {
            throw std::runtime_error(std::to_string(n) + " taps exceed MAX_TAPS=" + std::to_string(max_taps));
        }
        w.assign(n, 0.0);
        double total = 0.0;
        for (int j = 0; j < n; ++j) {
            w[j] = cubic((j + xmin - center + 0.5) * invscale);
            total += w[j];
        }
        t.start[i] = xmin;
        for (int j = 0; j < n; ++j) {
            t.weight[static_cast<std::size_t>(i) * max_taps + j] = total != 0.0 ? static_cast<float>(w[j] / total) : 0.0f;
        }
    }
    return t;
}

}  // namespace alpamayo_pre
