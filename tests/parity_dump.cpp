// parity_dump.cpp - dumps GEMM kernel output + timing to JSON so it can be
// diffed against a dump from a *different vendor's* hardware.
//
// Why this exists: verify_matmul_correctness.cpp proves a kernel is correct
// against a CPU reference, on whichever single machine it runs on. It can't
// tell you whether a CUDA build and a HIP build of the *same* kernel produce
// the same numbers on real NVIDIA vs real AMD hardware, because the two
// builds never run in the same process (different runtimes, usually
// different machines). This tool runs once per vendor, with a fixed seed and
// fixed shapes, and writes the full output matrix + GFLOPS to JSON. A
// separate script (scripts/parity_check.py) then loads two such dumps —
// produced independently on a CUDA machine and a HIP machine — and diffs
// them directly: cross-vendor numerical parity, not vendor-vs-CPU.
#include "../kernels/matmul.cuh"
#include "portable.hpp"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <sstream>

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

struct TileRun {
    int bm, bn, bk;
    double gflops;
    double latency_ms;
    float max_rel_err_vs_cpu;
    std::vector<float> output;
};

static TileRun run_tile(int BM, int BN, int BK, int M, int N, int K,
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

    const int WARMUP = 3, REPS = 20;
    for (int i = 0; i < WARMUP; ++i) AMP::launch_matmul(cfg, dA, dB, dC, M, N, K);
    AMP_SYNC();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < REPS; ++i) AMP::launch_matmul(cfg, dA, dB, dC, M, N, K);
    AMP_SYNC();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms_total = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_per_call = ms_total / REPS;
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (flops / (ms_per_call / 1000.0)) / 1e9;

    std::vector<float> out(M * N, 0.0f);
    AMP_MEMCPY_DH(out.data(), dC, (size_t)M * N * sizeof(float));
    AMP_FREE(dA); AMP_FREE(dB); AMP_FREE(dC);

    float max_rel_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float abs_err = std::fabs(out[i] - ref[i]);
        float rel_err = abs_err / std::max(1.0f, std::fabs(ref[i]));
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    return TileRun{BM, BN, BK, gflops, ms_per_call, max_rel_err, std::move(out)};
}

static void write_json(const std::string& path, int M, int N, int K, unsigned seed,
                        const std::vector<TileRun>& runs) {
    std::ofstream f(path);
    f << "{\n";
    f << "  \"vendor\": \"" << AMP_VENDOR << "\",\n";
    f << "  \"M\": " << M << ", \"N\": " << N << ", \"K\": " << K << ", \"seed\": " << seed << ",\n";
    f << "  \"tiles\": [\n";
    for (size_t t = 0; t < runs.size(); ++t) {
        const auto& r = runs[t];
        f << "    {\"bm\": " << r.bm << ", \"bn\": " << r.bn << ", \"bk\": " << r.bk
          << ", \"gflops\": " << r.gflops << ", \"latency_ms\": " << r.latency_ms
          << ", \"max_rel_err_vs_cpu\": " << r.max_rel_err_vs_cpu
          << ", \"output\": [";
        for (size_t i = 0; i < r.output.size(); ++i) {
            f << r.output[i];
            if (i + 1 != r.output.size()) f << ",";
        }
        f << "]}";
        if (t + 1 != runs.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n}\n";
}

int main(int argc, char** argv) {
    const std::string out_path = argc > 1 ? argv[1] : "parity_dump.json";
    const int M = 96, N = 96, K = 96;
    const unsigned seed = 42;

    std::vector<float> A(M * K), B(K * N), ref(M * N);
    srand(seed);
    for (auto& v : A) v = (float)(rand() % 2000 - 1000) / 100.0f;
    for (auto& v : B) v = (float)(rand() % 2000 - 1000) / 100.0f;
    cpu_reference_gemm(A, B, ref, M, N, K);

    printf("Parity dump — vendor=%s M=%d N=%d K=%d seed=%u\n", AMP_VENDOR, M, N, K, seed);

    struct { int bm, bn, bk; } tiles[] = {{16,16,16}, {32,16,16}, {32,32,16}, {32,32,32}};
    std::vector<TileRun> runs;
    for (auto& t : tiles) {
        TileRun r = run_tile(t.bm, t.bn, t.bk, M, N, K, A, B, ref);
        printf("  tile (%2d,%2d,%2d): %8.1f GFLOPS  %6.4f ms  max_rel_err_vs_cpu=%.6f\n",
               r.bm, r.bn, r.bk, r.gflops, r.latency_ms, r.max_rel_err_vs_cpu);
        runs.push_back(std::move(r));
    }

    write_json(out_path, M, N, K, seed, runs);
    printf("\nWrote %s — run this same binary on the other vendor's hardware,\n", out_path.c_str());
    printf("then: python3 scripts/parity_check.py <cuda_dump.json> <hip_dump.json>\n");
    return 0;
}

#else
int main() { printf("SKIP: requires AMP_BACKEND_CUDA or AMP_BACKEND_HIP\n"); return 0; }
#endif
