// verify_fixes.cpp - standalone host-side checks for bugs fixed in this session.
//
// This target is meant to be backend-agnostic (no GPU touched), but CMake's
// add_compile_definitions(AMP_BACKEND_CUDA/HIP/...) is a directory-scoped
// property that applies to every target in this CMakeLists.txt regardless
// of where add_executable() is called relative to it. #undef here forces
// portable.hpp's plain CPU stub path no matter what other targets need.
#ifdef AMP_BACKEND_CUDA
#undef AMP_BACKEND_CUDA
#endif
#ifdef AMP_BACKEND_HIP
#undef AMP_BACKEND_HIP
#endif
#ifdef AMP_BACKEND_SYCL
#undef AMP_BACKEND_SYCL
#endif

#include "fp8.hpp"
#include "quant.hpp"
#include "autotune_sh.hpp"
#include <cstdio>
#include <cmath>

int main() {
    int failures = 0;

    // ---- fp8.hpp: E4M3 round-trip across the [256,448) range ----
    // Before the fix, every value in this range encoded to exactly 448.
    {
        float probes[] = {256.f, 260.f, 288.f, 300.f, 320.f, 350.f, 384.f, 400.f, 416.f, 440.f, 448.f};
        bool all_distinct_ok = true;
        uint8_t prev_code = 0xFF;
        int distinct_codes = 0;
        for (float v : probes) {
            uint8_t code = AMP::fp8::f32_to_e4m3(v);
            float back = AMP::fp8::e4m3_to_f32(code);
            float err_pct = std::fabs(back - v) / v * 100.0f;
            if (code != prev_code) { distinct_codes++; prev_code = code; }
            printf("  e4m3(%6.1f) -> 0x%02x -> %6.1f  (err %.1f%%)\n", v, code, back, err_pct);
            if (err_pct > 10.0f) all_distinct_ok = false;  // >10% would indicate the old clamp-to-448 bug
        }
        if (!all_distinct_ok || distinct_codes < 5) {
            printf("FAIL: fp8 E4M3 [256,448) range still collapses to one code or has >10%% error\n");
            failures++;
        } else {
            printf("PASS: fp8 E4M3 [256,448) round-trip (%d distinct codes, max err <=10%%)\n", distinct_codes);
        }
    }

    // ---- quant.hpp: confirm std::max/std::min (via <algorithm>) resolve ----
    {
        auto cfg = AMP::QuantMode::INT8_PER_TENSOR;
        (void)cfg;
        printf("PASS: quant.hpp compiles and links (std::max/min via <algorithm> resolved)\n");
    }

    // ---- autotune_sh.hpp: empty candidates must throw, not UB-crash ----
    {
        std::vector<int> empty_candidates;
        bool threw = false;
        try {
            AMP::successive_halving<int>(empty_candidates,
                [](const int&, int) { return 0.0; });
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        if (threw) {
            printf("PASS: successive_halving() throws on empty candidates instead of UB\n");
        } else {
            printf("FAIL: successive_halving() did not throw on empty candidates\n");
            failures++;
        }
    }

    printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
