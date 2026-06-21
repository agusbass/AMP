// amp_validate_kernel.cpp - validate a USER-supplied GEMM kernel (CUDA or
// HIP, AMP doesn't care which) against a CPU reference, and dump the result
// in the same JSON schema amp_parity_dump uses so scripts/parity_check.py
// can diff a CUDA-side run against a HIP-side run of the user's own kernel.
//
// Deliberately links against neither CUDA nor HIP: the user's library
// already contains whatever device code it needs, and is loaded at
// runtime via dlopen(). This binary only ever touches host memory.
#include "amp_plugin.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <dlfcn.h>

static void cpu_reference_gemm(const std::vector<float>& A, const std::vector<float>& B,
                                std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

static void write_json(const std::string& path, const std::string& vendor_tag,
                        int M, int N, int K, unsigned seed,
                        double gflops, double latency_ms, float max_rel_err_vs_cpu,
                        const std::vector<float>& output) {
    std::ofstream f(path);
    f << "{\n  \"vendor\": \"" << vendor_tag << "\",\n"
      << "  \"M\": " << M << ", \"N\": " << N << ", \"K\": " << K << ", \"seed\": " << seed << ",\n"
      // A single synthetic "tile" entry (bm=bn=bk=0) keeps this dump
      // readable by parity_check.py unmodified -- a user plugin is a
      // black box with no tile-config sweep, but the comparison logic
      // (output diff, GFLOPS ratio) is identical either way.
      << "  \"tiles\": [\n    {\"bm\": 0, \"bn\": 0, \"bk\": 0, "
      << "\"gflops\": " << gflops << ", \"latency_ms\": " << latency_ms << ", "
      << "\"max_rel_err_vs_cpu\": " << max_rel_err_vs_cpu << ", \"output\": [";
    for (size_t i = 0; i < output.size(); ++i) {
        if (i) f << ",";
        f << output[i];
    }
    f << "]}\n  ]\n}\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <plugin.so> <out_dump.json> [M N K] [seed]\n"
            "  plugin.so must export: " AMP_PLUGIN_GEMM_FP32_SYMBOL "\n"
            "  (see include/amp_plugin.h and examples/user_plugin_example.cu)\n",
            argv[0]);
        return 2;
    }
    const std::string plugin_path = argv[1];
    const std::string out_path = argv[2];
    const int M = argc > 5 ? std::atoi(argv[3]) : 96;
    const int N = argc > 5 ? std::atoi(argv[4]) : 96;
    const int K = argc > 5 ? std::atoi(argv[5]) : 96;
    const unsigned seed = argc > 6 ? (unsigned)std::atoi(argv[6]) : 42u;

    void* handle = dlopen(plugin_path.c_str(), RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "Failed to load %s: %s\n", plugin_path.c_str(), dlerror());
        return 1;
    }
    dlerror();
    auto* fn = (amp_user_gemm_fp32_fn)dlsym(handle, AMP_PLUGIN_GEMM_FP32_SYMBOL);
    const char* dlsym_err = dlerror();
    if (!fn || dlsym_err) {
        fprintf(stderr, "Failed to find symbol '%s' in %s: %s\n",
                AMP_PLUGIN_GEMM_FP32_SYMBOL, plugin_path.c_str(),
                dlsym_err ? dlsym_err : "unknown error");
        dlclose(handle);
        return 1;
    }

    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f), ref(M * N);
    srand(seed);
    for (auto& v : A) v = (float)(rand() % 2000 - 1000) / 100.0f;
    for (auto& v : B) v = (float)(rand() % 2000 - 1000) / 100.0f;
    cpu_reference_gemm(A, B, ref, M, N, K);

    printf("Loaded plugin: %s\n", plugin_path.c_str());
    printf("Validating M=%d N=%d K=%d seed=%u against CPU reference...\n", M, N, K, seed);

    auto t0 = std::chrono::high_resolution_clock::now();
    int rc = fn(A.data(), B.data(), C.data(), M, N, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (rc != 0) {
        fprintf(stderr, "Plugin returned nonzero (%d) -- kernel launch/sync failed\n", rc);
        dlclose(handle);
        return 1;
    }
    double latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gflops = (2.0 * M * N * K) / (latency_ms * 1e6);

    float max_abs_err = 0.0f, max_rel_err = 0.0f;
    for (size_t i = 0; i < C.size(); ++i) {
        float abs_err = std::fabs(C[i] - ref[i]);
        float rel_err = abs_err / std::max(1.0f, std::fabs(ref[i]));
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    printf("  GFLOPS: %.1f   latency: %.4f ms\n", gflops, latency_ms);
    printf("  max_abs_err=%.6f max_rel_err=%.6f vs CPU reference  %s\n",
           max_abs_err, max_rel_err, max_rel_err < 1e-3f ? "PASS" : "FAIL");

    write_json(out_path, plugin_path, M, N, K, seed, gflops, latency_ms, max_rel_err, C);
    printf("\nWrote %s -- run this same tool with your HIP-side build of the\n", out_path.c_str());
    printf("same kernel, then: python3 scripts/parity_check.py <a>.json <b>.json\n");

    dlclose(handle);
    return max_rel_err < 1e-3f ? 0 : 1;
}
