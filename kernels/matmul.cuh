// matmul.cuh - GEMM kernels: FP32 tiled SHMEM + BF16/FP8 WMMA/MFMA tensor core
// FP32:  all SM/CU, tile size as template param → autotuner.
// BF16:  WMMA (CUDA SM>=80) or rocWMMA (HIP gfx90a+), FP32 accumulator.
// FP8E4M3: WMMA SM>=8.9 (Ada/H100) or MFMA MI300X, FP32 accumulator.
//          Throughput: ~2x BF16 per-unit on H100/MI300X.
// INT8:  DP4A (SM>=6.1) / IDOT (gfx908+), INT32 accumulator.
#pragma once
#include "portable.hpp"

#if defined(AMP_BACKEND_CUDA) || defined(AMP_BACKEND_HIP)

// ---- dtype tag (extends DataType for backward compat) ----
enum class MatmulDtype {
    FP32    = 0,
    BF16    = 1,
    FP8E4M3 = 2,   // 8-bit E4M3, WMMA SM>=8.9 / MFMA gfx940+
    FP8E5M2 = 3,   // 8-bit E5M2, WMMA SM>=8.9
    INT8    = 4,   // INT8→INT32 accumulate, DP4A
};

namespace AMP {

// ---- FP32 kernel (all hardware) ----
template<int BM, int BN, int BK>
__global__ void matmul_fp32_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];
    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = by * BM + ty;
    int col = bx * BN + tx;
    float acc = 0.0f;

    // Cooperative load: thread mapping for sA/sB is independent of the
    // output-tile thread mapping above, so this is correct for any
    // BM/BN/BK combination (not just square tiles where BM==BN==BK).
    const int tid = ty * BN + tx;
    const int threads_per_block = BM * BN;

    for (int kb = 0; kb < K; kb += BK) {
        for (int idx = tid; idx < BM * BK; idx += threads_per_block) {
            int r = idx / BK, c = idx % BK;
            int a_row = by * BM + r, a_col = kb + c;
            sA[r][c] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }
        for (int idx = tid; idx < BK * BN; idx += threads_per_block) {
            int r = idx / BN, c = idx % BN;
            int b_row = kb + r, b_col = bx * BN + c;
            sB[r][c] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) acc += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N + col] = acc;
}

// ---- BF16 WMMA kernel (CUDA SM>=80 / HIP gfx90a+) ----
// Each warp (32 threads) computes 1 output tile of 16x16.
// A and B must already be converted to BF16 by the caller.
// Output C is in FP32 (mixed-precision accumulation).
#if defined(AMP_BACKEND_CUDA)
  #include <mma.h>
  using namespace nvcuda;
  using AMP_bf16 = __nv_bfloat16;

  __global__ void matmul_bf16_wmma(const AMP_bf16* A, const AMP_bf16* B,
                                    float* C, int M, int N, int K);

  // FP8 kernel: implemented via cuBLASLt in gemm_vendor.cpp

#elif defined(AMP_BACKEND_HIP) && defined(AMP_HAVE_ROCWMMA)
  #include <rocwmma/rocwmma.hpp>
  using namespace rocwmma;
  using AMP_bf16 = hip_bfloat16;

  __global__ void matmul_bf16_wmma(const AMP_bf16* A, const AMP_bf16* B,
                                    float* C, int M, int N, int K);
  // FP8 MFMA: implemented via rocBLAS in gemm_vendor.cpp
#endif

// ---- Tile config + dispatch ----
struct TileCfg {
    int BM = 64, BN = 64, BK = 16;
    MatmulDtype dtype = MatmulDtype::FP32;
    // FP8: per-tensor scale factors (dequant at the accumulator)
    float scale_a = 1.0f;
    float scale_b = 1.0f;
};

struct TuneResult { TileCfg cfg; double ms = 1e18; double gflops = 0.0; };

// ---- FP32 launcher (all SMs, tile from autotuner) ----
void launch_matmul(const TileCfg& cfg,
                   const float* dA, const float* dB, float* dC,
                   int M, int N, int K, gpu_stream_t stream = 0);

// ---- BF16 launcher — SM>=80 (CUDA) or gfx90a+ (HIP+rocWMMA) ----
#if defined(AMP_BACKEND_CUDA) || (defined(AMP_BACKEND_HIP) && defined(AMP_HAVE_ROCWMMA))
  #if defined(AMP_BACKEND_CUDA)
  void launch_matmul_bf16(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
                           float* dC, int M, int N, int K,
                           gpu_stream_t stream = 0);
  #else
  void launch_matmul_bf16(const hip_bfloat16* dA, const hip_bfloat16* dB,
                           float* dC, int M, int N, int K,
                           gpu_stream_t stream = 0);
  #endif
  TuneResult autotune_matmul_bf16(int M, int N, int K,
                                   const AMP_bf16* dA, const AMP_bf16* dB,
                                   float* dC);
#endif

// ---- FP8 E4M3 launcher — SM>=8.9 (CUDA) or gfx940+ (HIP) ----
// scale_a, scale_b: descaling before accumulation (per-tensor)
// Output C is in FP32.
#if defined(AMP_HAVE_FP8)
void launch_matmul_fp8e4m3(const AMP_fp8_e4m3* dA, const AMP_fp8_e4m3* dB,
                             float* dC, int M, int N, int K,
                             float scale_a, float scale_b,
                             gpu_stream_t stream = 0);

TuneResult autotune_matmul_fp8(int M, int N, int K,
                                 const AMP_fp8_e4m3* dA,
                                 const AMP_fp8_e4m3* dB,
                                 float* dC,
                                 float scale_a = 1.0f, float scale_b = 1.0f);

bool has_tensor_core_fp8();  // SM>=8.9 or gfx940+
#endif

// ---- Runtime capability checks ----
bool has_tensor_core_bf16();  // SM>=80 or gfx90a+

// ---- FP32 autotuner ----
TuneResult autotune_matmul(int M, int N, int K,
                            const float* dA, const float* dB, float* dC);

} // namespace AMP
#endif
