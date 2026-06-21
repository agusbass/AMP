// verify_matmul_correctness.cpp - numerical correctness check for matmul_fp32_kernel
// across ALL tile configs, not just the one the autotuner happens to pick.
//
// Context: the (32,16,16) and (32,32,16) tile configs used to read/write out
// of shared-memory bounds (fixed in matmul.cuh). A plain speed benchmark
// (test_triple's "Modul 1") can't catch that — a kernel can produce wrong
// numbers without crashing. This test compares each tile config's GPU output
// against a naive CPU reference GEMM.
#include "../kernels/matmul.cuh"
#include "portable.hpp"
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#if defined(AMP_BACKEND_CUDA) || defined(AMP_BACKEND_HIP)

static void cpu_reference_gemm(const std::vector<float>& A, const std::vector<float>& B,
                                std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

static bool check_tile(int BM, int BN, int BK, int M, int N, int K,
                        const std::vector<float>& A, const std::vector<float>& B,
                        const std::vector<float>& ref) {
    float *dA, *dB, *dC;
    AMP_MALLOC(&dA, (size_t)M * K * sizeof(float));
    AMP_MALLOC(&dB, (size_t)K * N * sizeof(float));
    AMP_MALLOC(&dC, (size_t)M * N * sizeof(float));
    AMP_MEMCPY_HD(dA, A.data(), (size_t)M * K * sizeof(float));
    AMP_MEMCPY_HD(dB, B.data(), (size_t)K * N * sizeof(float));

    AMP::TileCfg cfg;
    cfg.BM = BM; cfg.BN = BN; cfg.BK = BK;
    bool launch_ok = true;
    try {
        AMP::launch_matmul(cfg, dA, dB, dC, M, N, K);
        AMP_SYNC();
    } catch (const std::exception& ex) {
        printf("  tile (%d,%d,%d): launch threw: %s\n", BM, BN, BK, ex.what());
        launch_ok = false;
    }

    std::vector<float> out(M * N, 0.0f);
    if (launch_ok) AMP_MEMCPY_DH(out.data(), dC, (size_t)M * N * sizeof(float));
    AMP_FREE(dA); AMP_FREE(dB); AMP_FREE(dC);
    if (!launch_ok) return false;

    float max_abs_err = 0.0f, max_rel_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float abs_err = std::fabs(out[i] - ref[i]);
        float rel_err = abs_err / std::max(1.0f, std::fabs(ref[i]));
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
    }
    bool ok = max_rel_err < 1e-3f;
    printf("  tile (%2d,%2d,%2d): max_abs_err=%.6f max_rel_err=%.6f  %s\n",
           BM, BN, BK, max_abs_err, max_rel_err, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    const int M = 96, N = 96, K = 96;  // not a multiple of all tile sizes on purpose
    std::vector<float> A(M * K), B(K * N), ref(M * N);
    srand(42);
    for (auto& v : A) v = (float)(rand() % 2000 - 1000) / 100.0f;
    for (auto& v : B) v = (float)(rand() % 2000 - 1000) / 100.0f;
    cpu_reference_gemm(A, B, ref, M, N, K);

    printf("Matmul FP32 correctness check (M=%d,N=%d,K=%d) vs CPU reference:\n", M, N, K);
    int failures = 0;
    struct { int bm, bn, bk; } tiles[] = {{16,16,16}, {32,16,16}, {32,32,16}, {32,32,32}};
    for (auto& t : tiles)
        if (!check_tile(t.bm, t.bn, t.bk, M, N, K, A, B, ref)) failures++;

    printf("\n%s\n", failures == 0 ? "ALL TILE CONFIGS NUMERICALLY CORRECT"
                                    : "SOME TILE CONFIGS PRODUCED WRONG RESULTS");
    return failures == 0 ? 0 : 1;
}

#else
int main() { printf("SKIP: requires AMP_BACKEND_CUDA or AMP_BACKEND_HIP\n"); return 0; }
#endif
