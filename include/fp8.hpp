// fp8.hpp - host-side FP8 conversion utilities
// FP8 E4M3: range ~[-448, 448], mantissa 3 bit, exponent 4 bit (bias=7)
// FP8 E5M2: range ~[-57344, 57344], mantissa 2 bit, exponent 5 bit (bias=15)
// Used for test data initialization and round-trip checks.
// Device-side casting is performed by hardware WMMA intrinsics.
#pragma once
#include <cstdint>
#include <cmath>
#include <limits>

namespace AMP {
namespace fp8 {

// ---- E4M3: bias=7, max_exp=14 (15 reserved for NaN/Inf), no Inf ----
inline float e4m3_to_f32(uint8_t v) {
    uint8_t sign = (v >> 7) & 1;
    uint8_t exp  = (v >> 3) & 0xF;
    uint8_t mant = v & 0x7;
    float result;
    if (exp == 0 && mant == 0) {
        result = 0.0f;
    } else if (exp == 0xF && mant == 0x7) {
        result = std::numeric_limits<float>::quiet_NaN();
    } else if (exp == 0) {
        // subnormal: value = (-1)^s * 2^(-6) * (0.mant)
        result = std::ldexp((float)mant / 8.0f, -6);
    } else {
        // normal: value = (-1)^s * 2^(exp-7) * (1 + mant/8)
        result = std::ldexp(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -result : result;
}

inline uint8_t f32_to_e4m3(float v) {
    if (std::isnan(v)) return 0x7F;  // NaN encoding
    uint8_t sign = (v < 0.0f) ? 0x80 : 0;
    v = std::fabs(v);
    // Clamp to E4M3 max = 448.0
    if (v > 448.0f) v = 448.0f;
    if (v == 0.0f) return sign;

    int exp;
    float frac = std::frexp(v, &exp);  // v = frac * 2^exp, frac in [0.5, 1.0)
    int biased = exp - 1 + 7;          // biased exponent (frexp gives 2^(exp-1) scale)

    if (biased <= 0) {
        // subnormal
        uint8_t mant = (uint8_t)std::round(v * std::ldexp(8.0f, 6));
        mant = mant > 7 ? 7 : mant;
        return sign | mant;
    }
    // biased==15 is a valid normal exponent in E4M3 (only E=15,M=7 is
    // reserved for NaN) — do NOT clamp here, compute the mantissa first.
    uint8_t e = (uint8_t)(biased > 15 ? 15 : biased);
    // mantissa: frac in [0.5, 1.0) → normalized mantissa = frac*2 - 1 in [0,1)
    float norm_mant = frac * 2.0f - 1.0f;
    uint8_t m = (uint8_t)std::round(norm_mant * 8.0f);
    if (m >= 8) { m = 0; e++; }
    // E=15,M=7 is the reserved NaN pattern — clamp any overflow into it
    // (or true exponent overflow) to the max finite value, 448.
    if (e > 15 || (e == 15 && m >= 7)) return sign | 0x7E;
    return sign | (e << 3) | m;
}

// ---- E5M2: bias=15, supports Inf/NaN ----
inline float e5m2_to_f32(uint8_t v) {
    uint8_t sign = (v >> 7) & 1;
    uint8_t exp  = (v >> 2) & 0x1F;
    uint8_t mant = v & 0x3;
    float result;
    if (exp == 0x1F) {
        result = (mant == 0)
            ? std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::quiet_NaN();
    } else if (exp == 0 && mant == 0) {
        result = 0.0f;
    } else if (exp == 0) {
        result = std::ldexp((float)mant / 4.0f, -14);
    } else {
        result = std::ldexp(1.0f + (float)mant / 4.0f, (int)exp - 15);
    }
    return sign ? -result : result;
}

inline uint8_t f32_to_e5m2(float v) {
    if (std::isnan(v))  return 0x7F;
    uint8_t sign = (v < 0.0f) ? 0x80 : 0;
    if (std::isinf(v))  return sign | 0x7C;
    v = std::fabs(v);
    // Clamp to E5M2 max = 57344
    if (v > 57344.0f) v = 57344.0f;
    if (v == 0.0f) return sign;

    int exp;
    float frac = std::frexp(v, &exp);
    int biased = exp - 1 + 15;

    if (biased <= 0) {
        uint8_t mant = (uint8_t)std::round(v * std::ldexp(4.0f, 14));
        mant = mant > 3 ? 3 : mant;
        return sign | mant;
    }
    if (biased >= 31) return sign | 0x7C;  // Inf

    uint8_t e = (uint8_t)biased;
    float norm_mant = frac * 2.0f - 1.0f;
    uint8_t m = (uint8_t)std::round(norm_mant * 4.0f);
    if (m >= 4) { m = 0; e++; }
    if (e >= 31) return sign | 0x7C;
    return sign | (e << 2) | m;
}

} // namespace fp8
} // namespace AMP
