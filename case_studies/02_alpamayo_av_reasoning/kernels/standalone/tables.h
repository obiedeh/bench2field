// Resampling tables for the Alpamayo preprocessing kernels, in C++.
//
// The same computation as preprocess_ext.bicubic_aa_table: for every output
// index, the first input index of its window and up to MAX_TAPS normalized
// antialiased bicubic weights (Keys cubic, a = -0.5, stretched by the scale
// factor), the way torch computes them for an antialiased downscale. The
// standalone driver builds its tables here; a Python check compares them with
// the Python tables element by element.
#pragma once

#include <vector>

namespace alpamayo_pre {

struct AxisTable {
    std::vector<int> start;      // [out_size]
    std::vector<float> weight;   // [out_size * max_taps], zero padded
};

// Throws std::runtime_error if any window needs more than max_taps taps.
AxisTable bicubic_aa_table(int in_size, int out_size, int max_taps);

}  // namespace alpamayo_pre
