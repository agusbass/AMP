// quant.hpp - INT8/INT4 quantization types and utilities
// Supports:
//   - Per-tensor symmetric INT8 (scale only)
//   - Per-channel symmetric INT8 (scale per row/col)
//   - INT4 packed (2 values per byte, nibble packing)
//   - AWQ/GPTQ-style INT4 with group scales
// Device kernels (quantize/dequantize) in quant.cpp or inline here.
#pragma once
#include "portable.hpp"
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace AMP {

// ---- Quantization config ----
enum class QuantMode : uint8_t {
    INT8_PER_TENSOR  = 0,  // single scale for the entire tensor
    INT8_PER_CHANNEL = 1,  // scale per row (weight quantization)
    INT4_PER_GROUP   = 2,  // AWQ/GPTQ-style: group_size elements per scale
    INT8_DYNAMIC     = 3,  // scale computed per-token at inference time
};

struct QuantConfig {
    QuantMode mode       = QuantMode::INT8_PER_TENSOR;
    int       group_size = 128;    // only for INT4_PER_GROUP
    bool      symmetric  = true;   // symmetric: zero_point = 0
    DataType  compute_dtype = DataType::FP32;  // dtype for dequant
};

// Scale + zero_point for one quantization group / channel
struct QuantScale {
    float   scale      = 1.0f;
    int32_t zero_point = 0;    // 0 for symmetric
};

// ---- INT4 nibble packing (2 INT4 per byte, lower nibble first) ----
struct Int4Pack {
    static uint8_t pack(int8_t lo, int8_t hi) {
        return ((uint8_t)(lo & 0xF)) | (uint8_t)((hi & 0xF) << 4);
    }
    static void unpack(uint8_t v, int8_t& lo, int8_t& hi) {
        lo = (int8_t)((v & 0xF) << 4) >> 4;   // sign-extend 4→8 bit
        hi = (int8_t)((v >> 4) << 4) >> 4;
    }
};

// ---- Host-side quantize / dequantize ----
namespace detail {

// Per-tensor: compute scale from abs-max, quantize to INT8 symmetric [-127, 127]
// (not [-128, 127] — avoids the asymmetry that complicates DP4A hardware)
inline QuantScale compute_int8_scale(const float* data, size_t n) {
    float amax = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float a = std::fabs(data[i]);
        if (a > amax) amax = a;
    }
    QuantScale s;
    s.scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
    return s;
}

inline void quantize_int8(const float* src, int8_t* dst, size_t n,
                           const QuantScale& s) {
    float inv = 1.0f / s.scale;
    for (size_t i = 0; i < n; ++i) {
        float v = src[i] * inv + s.zero_point;
        v = v < -127.0f ? -127.0f : (v > 127.0f ? 127.0f : v);
        dst[i] = (int8_t)std::round(v);
    }
}

inline void dequantize_int8(const int8_t* src, float* dst, size_t n,
                              const QuantScale& s) {
    for (size_t i = 0; i < n; ++i)
        dst[i] = ((float)src[i] - s.zero_point) * s.scale;
}

// Per-channel: each row M has its own scale (weight quant for Linear)
inline std::vector<QuantScale> compute_int8_per_channel(
        const float* data, size_t rows, size_t cols) {
    std::vector<QuantScale> scales(rows);
    for (size_t r = 0; r < rows; ++r) {
        float amax = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float a = std::fabs(data[r * cols + c]);
            if (a > amax) amax = a;
        }
        scales[r].scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
    }
    return scales;
}

// INT4 group quantization (AWQ/GPTQ style)
// Output: packed nibbles, shape = (rows, cols/2)
// Scales: shape = (rows, cols / group_size)
struct Int4QuantResult {
    std::vector<uint8_t>     packed;   // nibble-packed INT4
    std::vector<QuantScale>  scales;   // per-group scales
    size_t rows, cols, group_size;
};

inline Int4QuantResult quantize_int4_group(
        const float* data, size_t rows, size_t cols, size_t group_size = 128) {
    if (cols % 2 != 0) throw std::invalid_argument("cols must be even for INT4 packing");
    if (cols % group_size != 0) throw std::invalid_argument("cols must be divisible by group_size");

    size_t n_groups = cols / group_size;
    Int4QuantResult r;
    r.rows = rows; r.cols = cols; r.group_size = group_size;
    r.packed.resize(rows * cols / 2);
    r.scales.resize(rows * n_groups);

    for (size_t row = 0; row < rows; ++row) {
        for (size_t g = 0; g < n_groups; ++g) {
            size_t start = row * cols + g * group_size;
            float amax = 0.0f;
            for (size_t i = 0; i < group_size; ++i) {
                float a = std::fabs(data[start + i]);
                if (a > amax) amax = a;
            }
            float scale = (amax > 0.0f) ? amax / 7.0f : 1.0f;  // INT4: [-7, 7]
            r.scales[row * n_groups + g].scale = scale;

            float inv = 1.0f / scale;
            for (size_t i = 0; i < group_size; i += 2) {
                int8_t lo = (int8_t)std::round(
                    std::max(-7.0f, std::min(7.0f, data[start + i]     * inv)));
                int8_t hi = (int8_t)std::round(
                    std::max(-7.0f, std::min(7.0f, data[start + i + 1] * inv)));
                size_t pack_idx = (row * cols + g * group_size + i) / 2;
                r.packed[pack_idx] = Int4Pack::pack(lo, hi);
            }
        }
    }
    return r;
}

} // namespace detail

// ---- GPU quantization descriptor (for kernel dispatch) ----
// Used by the matmul dispatcher to select the INT8 GEMM path
struct QuantizedTensor {
    void*    data_dev   = nullptr;  // int8_t* on device
    float*   scales_dev = nullptr;  // scale array on device (nullptr = per-tensor)
    float    global_scale = 1.0f;   // per-tensor scale
    size_t   rows = 0, cols = 0;
    DataType dtype = DataType::INT8;
    int      group_size = 0;        // 0 = per-tensor / per-channel
};

} // namespace AMP
