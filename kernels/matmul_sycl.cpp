// matmul_sycl.cpp - Intel SYCL GEMM using joint_matrix (XMX)
// Prerequisites: Intel oneAPI 2024.1+, GPU with XMX (Xe-HPC / Gaudi 3 / Arc)
// Build: icpx -fsycl + add_sycl_to_target in CMake
//
// Fallback: without -DAMP_HAVE_XMX, use tiled SYCL (ND_range, shared local mem)
#include "portable.hpp"

#ifdef AMP_BACKEND_SYCL
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include <chrono>

#ifdef AMP_HAVE_XMX
#include <sycl/ext/intel/experimental/matrix/matrix.hpp>
using namespace sycl::ext::intel::experimental::matrix;
#endif

namespace AMP {

// ----------------------------------------------------------------
// XMX joint_matrix GEMM (Intel Xe-HPC / Gaudi3)
// M, N, K: GEMM dimensions (must be multiples of tile sizes)
// A[M×K] BF16, B[K×N] BF16, C[M×N] FP32
// ----------------------------------------------------------------
#ifdef AMP_HAVE_XMX

static void gemm_xmx_bf16(sycl::queue& q,
                            const sycl::bfloat16* A,
                            const sycl::bfloat16* B,
                            float* C,
                            int M, int N, int K)
{
    // XMX tile: 8×16×16 (varies by hardware — use conservative 8x16x16)
    constexpr int TM = 8, TN = 16, TK = 16;
    if (M % TM || N % TN || K % TK)
        throw std::invalid_argument("M,N,K must be multiples of XMX tile (8,16,16)");

    sycl::range<2> global{(size_t)(M / TM), (size_t)(N / TN)};
    sycl::range<2> local{1, 1};   // 1 subgroup per work-group

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>{global, local},
            [=](sycl::nd_item<2> item)
            [[intel::reqd_sub_group_size(16)]]  // XMX needs SG size 16
            {
                auto sg = item.get_sub_group();
                int m_tile = item.get_global_id(0);
                int n_tile = item.get_global_id(1);

                // joint_matrix accumulator (FP32)
                joint_matrix<float, use::accumulator, TM, TN,
                             layout::row_major> c_mat;
                joint_matrix_fill(sg, c_mat, 0.0f);

                for (int k = 0; k < K; k += TK) {
                    // Load A tile [m_tile*TM, k] → [+TM, +TK]
                    joint_matrix<sycl::bfloat16, use::a, TM, TK,
                                 layout::row_major> a_mat;
                    joint_matrix_load(sg, a_mat,
                        A + m_tile * TM * K + k, K,
                        layout::row_major);

                    // Load B tile [k, n_tile*TN] → [+TK, +TN]
                    joint_matrix<sycl::bfloat16, use::b, TK, TN,
                                 layout::row_major> b_mat;
                    joint_matrix_load(sg, b_mat,
                        B + k * N + n_tile * TN, N,
                        layout::row_major);

                    // MMA: C += A × B
                    joint_matrix_mad(sg, c_mat, a_mat, b_mat, c_mat);
                }

                // Store result
                joint_matrix_store(sg, c_mat,
                    C + m_tile * TM * N + n_tile * TN, N,
                    layout::row_major);
            });
    });
}

#else

// ----------------------------------------------------------------
// Fallback: tiled SYCL GEMM with local memory (no XMX)
// Analogous to matmul_fp32_kernel in CUDA, but for SYCL BF16→FP32
// ----------------------------------------------------------------
static void gemm_tiled_sycl(sycl::queue& q,
                              const sycl::bfloat16* A,
                              const sycl::bfloat16* B,
                              float* C,
                              int M, int N, int K)
{
    constexpr int TILE = 16;
    sycl::range<2> global{(size_t)M, (size_t)N};
    sycl::range<2> local{TILE, TILE};

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> lA({TILE, TILE}, h);
        sycl::local_accessor<float, 2> lB({TILE, TILE}, h);

        h.parallel_for(
            sycl::nd_range<2>{global, local},
            [=](sycl::nd_item<2> item) {
                int row = item.get_global_id(0);
                int col = item.get_global_id(1);
                int lr  = item.get_local_id(0);
                int lc  = item.get_local_id(1);

                float acc = 0.0f;
                for (int t = 0; t < K; t += TILE) {
                    lA[lr][lc] = (row < M && t + lc < K)
                        ? (float)A[row * K + t + lc] : 0.0f;
                    lB[lr][lc] = (t + lr < K && col < N)
                        ? (float)B[(t + lr) * N + col] : 0.0f;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int k = 0; k < TILE; ++k)
                        acc += lA[lr][k] * lB[k][lc];
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (row < M && col < N) C[row * N + col] = acc;
            });
    });
}

#endif  // AMP_HAVE_XMX

// ----------------------------------------------------------------
// Public API (matching CUDA/HIP launch signature style)
// ----------------------------------------------------------------
struct SyclGemmResult {
    double ms;
    double gflops;
};

SyclGemmResult launch_matmul_sycl_bf16(
    const sycl::bfloat16* dA, const sycl::bfloat16* dB, float* dC,
    int M, int N, int K, sycl::queue* q_ptr)
{
    sycl::queue& q = q_ptr ? *q_ptr : amp::detail::default_q();

    auto t0 = std::chrono::steady_clock::now();
#ifdef AMP_HAVE_XMX
    gemm_xmx_bf16(q, dA, dB, dC, M, N, K);
#else
    gemm_tiled_sycl(q, dA, dB, dC, M, N, K);
#endif
    q.wait();
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    return {ms, gflops};
}

SyclGemmResult autotune_matmul_sycl_bf16(
    int M, int N, int K,
    const sycl::bfloat16* dA, const sycl::bfloat16* dB, float* dC,
    sycl::queue* q_ptr)
{
    sycl::queue& q = q_ptr ? *q_ptr : amp::detail::default_q();
    // Warmup
    launch_matmul_sycl_bf16(dA, dB, dC, M, N, K, &q);

    const int reps = 10;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i)
        launch_matmul_sycl_bf16(dA, dB, dC, M, N, K, &q);
    q.wait();
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count() / reps;
    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    return {ms, gflops};
}

bool has_xmx_support() {
    try {
        auto dev = amp::detail::default_q().get_device();
        return dev.has(sycl::aspect::ext_intel_matrix);
    } catch (...) { return false; }
}

} // namespace AMP

#endif // AMP_BACKEND_SYCL
