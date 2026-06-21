// matmul.cu - FP32 tile dispatch + BF16 WMMA kernel + autotuners
#include "matmul.cuh"
#include "gemm_vendor.hpp"
#include "profiler.hpp"
#include <chrono>
#include <vector>
#include <stdexcept>

#if defined(AMP_BACKEND_CUDA) || defined(AMP_BACKEND_HIP)

namespace AMP {

// =====================================================================
// FP32 dispatch table (9 tile configs, all compiled in)
// =====================================================================
#define DISPATCH_FP32(bm, bn, bk) \
    if (cfg.BM == bm && cfg.BN == bn && cfg.BK == bk) { \
        dim3 block(bn, bm); \
        dim3 grid((N + bn - 1)/bn, (M + bm - 1)/bm); \
        matmul_fp32_kernel<bm,bn,bk><<<grid, block, 0, stream>>>(dA,dB,dC,M,N,K); \
        return; \
    }

void launch_matmul(const TileCfg& cfg,
                   const float* dA, const float* dB, float* dC,
                   int M, int N, int K, gpu_stream_t stream) {
    AMP_RANGE_PUSH("amp::matmul_fp32");
    // Only valid configs: BM*BN <= 1024 (max threads per block)
    DISPATCH_FP32(16,16,16);   // 256 threads
    DISPATCH_FP32(32,16,16);   // 512 threads
    DISPATCH_FP32(32,32,16);   // 1024 threads
    DISPATCH_FP32(32,32,32);   // 1024 threads
    AMP_RANGE_POP();
    throw std::runtime_error("unknown FP32 tile cfg");
}

// FP8 hand-rolled kernels removed — use cuBLASLt/rocBLAS via gemm_vendor.
// cuBLASLt FP8 (CUDA 12+) is much faster than hand-rolled WMMA.

// =====================================================================
// BF16 WMMA kernel implementation
// =====================================================================
#if defined(AMP_BACKEND_CUDA)

// Kernel is compiled for all target archs in CMAKE_CUDA_ARCHITECTURES.
// Runtime check has_tensor_core_bf16() prevents launching on SM<80.
__global__ void matmul_bf16_wmma(const __nv_bfloat16* A, const __nv_bfloat16* B,
                                  float* C, int M, int N, int K) {
#if __CUDA_ARCH__ >= 800
    using frag_a = wmma::fragment<wmma::matrix_a, 16, 16, 16,
                                  __nv_bfloat16, wmma::row_major>;
    using frag_b = wmma::fragment<wmma::matrix_b, 16, 16, 16,
                                  __nv_bfloat16, wmma::row_major>;
    using frag_c = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    // Staging tiles in shared mem to handle partial boundary tiles
    __shared__ __nv_bfloat16 sA[16][16];
    __shared__ __nv_bfloat16 sB[16][16];
    __shared__ float          sC[16][16];

    frag_a fa; frag_b fb; frag_c fc;
    wmma::fill_fragment(fc, 0.0f);

    int wM = blockIdx.y;   // row tile index
    int wN = blockIdx.x;   // col tile index
    int lane = threadIdx.x; // 0..31

    for (int k = 0; k < K; k += 16) {
        // Stage A[wM*16..+16, k..+16] — 32 threads fill 256 elements
        for (int i = lane; i < 256; i += 32) {
            int r = i >> 4, c = i & 15;
            int gR = wM*16 + r, gC = k + c;
            sA[r][c] = (gR < M && gC < K) ? A[gR*K + gC] : __nv_bfloat16(0.f);
        }
        // Stage B[k..+16, wN*16..+16]
        for (int i = lane; i < 256; i += 32) {
            int r = i >> 4, c = i & 15;
            int gR = k + r, gC = wN*16 + c;
            sB[r][c] = (gR < K && gC < N) ? B[gR*N + gC] : __nv_bfloat16(0.f);
        }
        __syncthreads();

        wmma::load_matrix_sync(fa, &sA[0][0], 16);
        wmma::load_matrix_sync(fb, &sB[0][0], 16);
        wmma::mma_sync(fc, fa, fb, fc);
        __syncthreads();
    }

    // Scatter output via shared mem to handle partial C tiles
    if (wM*16 < M && wN*16 < N) {
        wmma::store_matrix_sync(&sC[0][0], fc, 16, wmma::mem_row_major);
        __syncthreads();
        for (int i = lane; i < 256; i += 32) {
            int r = i >> 4, c = i & 15;
            int gR = wM*16 + r, gC = wN*16 + c;
            if (gR < M && gC < N) C[gR*N + gC] = sC[r][c];
        }
    }
#endif // __CUDA_ARCH__ >= 800
}

void launch_matmul_bf16(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
                         float* dC, int M, int N, int K, gpu_stream_t stream) {
    AMP_RANGE_PUSH("amp::matmul_bf16_wmma");
    // 1 warp per block, each block = 1 output tile 16x16
    dim3 block(32, 1, 1);
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    matmul_bf16_wmma<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    AMP_RANGE_POP();
}

bool has_tensor_core_bf16() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
    return p.major >= 8;  // Ampere (A100) and above
}

TuneResult autotune_matmul_bf16(int M, int N, int K,
                                  const __nv_bfloat16* dA,
                                  const __nv_bfloat16* dB, float* dC) {
    AMP_RANGE_PUSH("amp::autotune_bf16");
    // BF16 WMMA tile size is fixed at 16x16x16 — the only valid config for wmma.
    // What can be tuned: number of warps per block (multi-warp tiling).
    // For simplicity, return the fixed config directly + measure real throughput.
    // 3 warmup + 20 reps for a stable thermal/boost state
    for (int w = 0; w < 3; ++w) launch_matmul_bf16(dA, dB, dC, M, N, K);
    AMP_SYNC();

    const int reps = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i)
        launch_matmul_bf16(dA, dB, dC, M, N, K);
    AMP_SYNC();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count() / reps;
    // BF16 FLOPS: 2*M*N*K (same as FP32, but tensor core throughput is much higher)
    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    AMP_RANGE_POP();
    return {{16, 16, 16, MatmulDtype::BF16}, ms, gflops};
}

#elif defined(AMP_BACKEND_HIP) && defined(AMP_HAVE_ROCWMMA)

__global__ void matmul_bf16_wmma(const hip_bfloat16* A, const hip_bfloat16* B,
                                  float* C, int M, int N, int K) {
    using frag_a = fragment<matrix_a, 16, 16, 16, hip_bfloat16, row_major>;
    using frag_b = fragment<matrix_b, 16, 16, 16, hip_bfloat16, row_major>;
    using frag_c = fragment<accumulator, 16, 16, 16, float>;

    __shared__ hip_bfloat16 sA[16][16];
    __shared__ hip_bfloat16 sB[16][16];
    __shared__ float         sC[16][16];

    frag_a fa; frag_b fb; frag_c fc;
    fill_fragment(fc, 0.0f);

    int wM = blockIdx.y, wN = blockIdx.x;
    int lane = threadIdx.x;

    for (int k = 0; k < K; k += 16) {
        for (int i = lane; i < 256; i += 64) { // wave64
            int r = i >> 4, c = i & 15;
            int gR = wM*16 + r, gC = k + c;
            sA[r][c] = (gR < M && gC < K) ? A[gR*K + gC] : hip_bfloat16(0.f);
        }
        for (int i = lane; i < 256; i += 64) {
            int r = i >> 4, c = i & 15;
            int gR = k + r, gC = wN*16 + c;
            sB[r][c] = (gR < K && gC < N) ? B[gR*N + gC] : hip_bfloat16(0.f);
        }
        __syncthreads();
        load_matrix_sync(fa, &sA[0][0], 16);
        load_matrix_sync(fb, &sB[0][0], 16);
        mma_sync(fc, fa, fb, fc);
        __syncthreads();
    }

    if (wM*16 < M && wN*16 < N) {
        store_matrix_sync(&sC[0][0], fc, 16, mem_row_major);
        __syncthreads();
        for (int i = lane; i < 256; i += 64) {
            int r = i >> 4, c = i & 15;
            int gR = wM*16 + r, gC = wN*16 + c;
            if (gR < M && gC < N) C[gR*N + gC] = sC[r][c];
        }
    }
}

void launch_matmul_bf16(const hip_bfloat16* dA, const hip_bfloat16* dB,
                         float* dC, int M, int N, int K, gpu_stream_t stream) {
    AMP_RANGE_PUSH("amp::matmul_bf16_mfma");
    dim3 block(64, 1, 1);  // wave64 for AMD
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    matmul_bf16_wmma<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    AMP_RANGE_POP();
}

bool has_tensor_core_bf16() {
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, 0) != hipSuccess) return false;
    std::string arch = p.gcnArchName;
    return arch.find("gfx90a") != std::string::npos ||
           arch.find("gfx940") != std::string::npos ||
           arch.find("gfx941") != std::string::npos ||
           arch.find("gfx942") != std::string::npos;
}

TuneResult autotune_matmul_bf16(int M, int N, int K,
                                  const hip_bfloat16* dA,
                                  const hip_bfloat16* dB, float* dC) {
    const int reps = 10;
    launch_matmul_bf16(dA, dB, dC, M, N, K);
    AMP_SYNC();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) launch_matmul_bf16(dA, dB, dC, M, N, K);
    AMP_SYNC();
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now()-t0).count() / reps;
    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    return {{16, 16, 16, MatmulDtype::BF16}, ms, gflops};
}

#else
// Fallback: BF16 is not available in this configuration
bool has_tensor_core_bf16() { return false; }
#endif

// =====================================================================
// FP8 launchers + autotuner
// =====================================================================
#if defined(AMP_HAVE_FP8)

void launch_matmul_fp8e4m3(const AMP_fp8_e4m3* dA, const AMP_fp8_e4m3* dB,
                             float* dC, int M, int N, int K,
                             float scale_a, float scale_b,
                             gpu_stream_t stream)
{
    AMP_RANGE_PUSH("amp::matmul_fp8e4m3");
    // cuBLASLt FP8 REQUIRES: scale pointers must be device-side float*
    float *d_sa = nullptr, *d_sb = nullptr;
#if defined(AMP_BACKEND_CUDA)
    cudaMalloc(&d_sa, sizeof(float));
    cudaMalloc(&d_sb, sizeof(float));
    cudaMemcpy(d_sa, &scale_a, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sb, &scale_b, sizeof(float), cudaMemcpyHostToDevice);
#elif defined(AMP_BACKEND_HIP)
    hipMalloc(&d_sa, sizeof(float));
    hipMalloc(&d_sb, sizeof(float));
    hipMemcpy(d_sa, &scale_a, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_sb, &scale_b, sizeof(float), hipMemcpyHostToDevice);
#endif
    GemmDesc desc;
    desc.M = M; desc.N = N; desc.K = K;
    desc.dtype_a   = DataType::FP8E4M3;
    desc.dtype_b   = DataType::FP8E4M3;
    desc.dtype_c   = DataType::FP32;
    desc.alpha     = 1.0f;  // scaling via d_scale_a/b, not alpha
    desc.beta      = 0.0f;
    desc.d_scale_a = d_sa;
    desc.d_scale_b = d_sb;
    global_gemm().gemm(desc, dA, dB, dC, stream);
#if defined(AMP_BACKEND_CUDA)
    cudaFree(d_sa); cudaFree(d_sb);
#elif defined(AMP_BACKEND_HIP)
    hipFree(d_sa);  hipFree(d_sb);
#endif
    AMP_RANGE_POP();
}

bool has_tensor_core_fp8() {
#if defined(AMP_BACKEND_CUDA)
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
    return (p.major > 8) || (p.major == 8 && p.minor >= 9);  // SM 8.9+
#elif defined(AMP_BACKEND_HIP)
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, 0) != hipSuccess) return false;
    std::string arch = p.gcnArchName;
    return arch.find("gfx940") != std::string::npos ||
           arch.find("gfx941") != std::string::npos ||
           arch.find("gfx942") != std::string::npos;
#else
    return false;
#endif
}

TuneResult autotune_matmul_fp8(int M, int N, int K,
                                 const AMP_fp8_e4m3* dA,
                                 const AMP_fp8_e4m3* dB,
                                 float* dC,
                                 float scale_a, float scale_b)
{
    AMP_RANGE_PUSH("amp::autotune_fp8");
    if (!has_tensor_core_fp8()) {
        AMP_RANGE_POP();
        return {{16, 16, 32, MatmulDtype::FP8E4M3}, 1e18, 0.0};
    }

    // 3 warmup runs so cuBLASLt algorithm selection stabilizes
    for (int w = 0; w < 3; ++w)
        launch_matmul_fp8e4m3(dA, dB, dC, M, N, K, scale_a, scale_b);
    AMP_SYNC();

    const int reps = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i)
        launch_matmul_fp8e4m3(dA, dB, dC, M, N, K, scale_a, scale_b);
    AMP_SYNC();
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count() / reps;
    // FP8: same FLOP count as BF16, but throughput ~2x on H100/MI300X
    double gflops = (2.0 * M * N * K) / (ms * 1e6);

    TileCfg cfg;
    cfg.BM = 16; cfg.BN = 16; cfg.BK = 32;
    cfg.dtype = MatmulDtype::FP8E4M3;
    cfg.scale_a = scale_a; cfg.scale_b = scale_b;

    AMP_RANGE_POP();
    return {cfg, ms, gflops};
}

#endif  // AMP_HAVE_FP8

// =====================================================================
// FP32 autotuner (grid search — for backward compat)
// =====================================================================
TuneResult autotune_matmul(int M, int N, int K,
                            const float* dA, const float* dB, float* dC) {
    AMP_RANGE_PUSH("amp::autotune_fp32");
    // Only valid configs: BM*BN <= 1024
    std::vector<TileCfg> space = {
        {16,16,16}, {32,16,16}, {32,32,16}, {32,32,32}
    };
    TuneResult best{{0,0,0}, 1e18, 0.0};
    const int WARMUP = 3, REPS = 20;  // 3 warmup + 20 reps for stability
    for (auto& c : space) {
        // warmup: 3x to flush cache + bring GPU to steady-state
        bool ok = true;
        for (int w = 0; w < WARMUP; ++w) {
            try { launch_matmul(c, dA, dB, dC, M, N, K); }
            catch(...) { ok = false; break; }
        }
        if (!ok) continue;
        AMP_SYNC();

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < REPS; ++i)
            launch_matmul(c, dA, dB, dC, M, N, K);
        AMP_SYNC();
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now()-t0).count() / REPS;
        double gflops = (2.0 * M * N * K) / (ms * 1e6);
        if (ms < best.ms) best = {c, ms, gflops};
    }
    AMP_RANGE_POP();
    return best;
}

} // namespace AMP
#endif
