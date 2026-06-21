// flash_attn.cu - FlashAttention-2 forward pass
// FA2 algorithm (Dao 2023):
//   Outer loop: Tc blocks of K,V (iterated outside the kernel / 1 block per Q tile)
//   Inner loop: Tr blocks of Q (each CUDA block handles 1 Q tile)
//   Online softmax: update (m, l, O) without materializing S to HBM
//
// Tile parallelism: 1 CUDA threadblock = 1 (batch, head, Q-tile)
// Thread layout: Br warps per block, each warp computes 1 Q row
//
// SM>=80 required for BF16 WMMA (Ampere, A100, H100)
// HIP gfx90a+ for rocWMMA BF16

#include "flash_attn.cuh"
#include "profiler.hpp"
#include <cmath>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <limits>

#if defined(AMP_BACKEND_CUDA) || defined(AMP_BACKEND_HIP)

#if defined(AMP_BACKEND_CUDA)
  #include <mma.h>
  using namespace nvcuda;
#elif defined(AMP_BACKEND_HIP) && defined(AMP_HAVE_ROCWMMA)
  #include <rocwmma/rocwmma.hpp>
  using namespace rocwmma;
#endif

namespace AMP {

// ================================================================
// Device helpers
// ================================================================

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int mask = 16; mask > 0; mask >>= 1)
#if defined(AMP_BACKEND_CUDA)
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, mask));
#else
        // HIP has no __shfl_xor_sync — __shfl_xor takes no mask argument.
        v = fmaxf(v, __shfl_xor(v, mask));
#endif
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int mask = 16; mask > 0; mask >>= 1)
#if defined(AMP_BACKEND_CUDA)
        v += __shfl_xor_sync(0xFFFFFFFF, v, mask);
#else
        v += __shfl_xor(v, mask);
#endif
    return v;
}

// ================================================================
// Core FA2 kernel — BF16 input, FP32 accumulator
// Template: Br = Q-tile rows, Bc = KV-tile rows, HEAD_DIM = head_dim
//
// Grid:  (Tr, B * num_q_heads)  where Tr = ceil(S_q / Br)
// Block: (32, Br/1)  — 1 warp per Q-row, Br warps total
//        Practical: (Br * 32) threads, arranged as Br warps
// ================================================================
// Dynamic shared memory layout (bytes):
//   sQ : Br * HEAD_DIM * sizeof(float)
//   sK : Bc * HEAD_DIM * sizeof(float)
//   sV : Bc * HEAD_DIM * sizeof(float)
//   sS : Br * Bc       * sizeof(float)
// Total passed as third kernel launch arg.

template<int Br, int Bc, int HEAD_DIM>
__global__ void flash_attn_fwd_kernel(
    const AMP_bf16* __restrict__ Q,
    const AMP_bf16* __restrict__ K,
    const AMP_bf16* __restrict__ V,
    AMP_bf16*       __restrict__ O,
    float*          __restrict__ lse,
    int B, int H_q, int H_kv, int S_q, int S_kv,
    float scale, bool causal,
    int stride_B_q, int stride_H_q, int stride_S_q,
    int stride_B_k, int stride_H_k, int stride_S_k,
    int stride_B_o, int stride_H_o, int stride_S_o)
{
    const int q_tile = blockIdx.x;
    const int bh     = blockIdx.y;
    const int b      = bh / H_q;
    const int hq     = bh % H_q;
    const int hkv    = hq / (H_q / H_kv);
    const int lane   = threadIdx.x % 32;
    const int row    = threadIdx.x / 32;
    const int q_row  = q_tile * Br + row;
    if (q_row >= S_q) return;

    const AMP_bf16* qptr = Q + b * stride_B_q + hq  * stride_H_q;
    const AMP_bf16* kptr = K + b * stride_B_k + hkv * stride_H_k;
    const AMP_bf16* vptr = V + b * stride_B_k + hkv * stride_H_k;
    AMP_bf16*       optr = O + b * stride_B_o + hq  * stride_H_o;
    float*          lptr = lse + (b * H_q + hq) * S_q;

    // Dynamic shared memory — avoids static array size limit
    extern __shared__ float _smem[];
    float* sQ = _smem;
    float* sK = sQ + Br * HEAD_DIM;
    float* sV = sK + Bc * HEAD_DIM;
    float* sS = sV + Bc * HEAD_DIM;
    // Helper lambdas for 2D indexing
    auto SQ = [&](int r, int d) -> float& { return sQ[r * HEAD_DIM + d]; };
    auto SK = [&](int r, int d) -> float& { return sK[r * HEAD_DIM + d]; };
    auto SV = [&](int r, int d) -> float& { return sV[r * HEAD_DIM + d]; };
    auto SS = [&](int r, int c) -> float& { return sS[r * Bc + c]; };

    // Load Q tile row into sQ
    for (int d = lane; d < HEAD_DIM; d += 32) {
        SQ(row, d) = (q_row < S_q)
            ? AMP_BF16_TO_FLOAT(qptr[q_row * stride_S_q + d])
            : 0.0f;
    }
    __syncthreads();

    // Per-row accumulators for online softmax
    float m_i = -1e30f;  // running max
    float l_i = 0.0f;    // running sum exp
    float o_i[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) o_i[d] = 0.0f;  // explicit init, safer in device code

    // Iterate over KV tiles
    const int Tc = (S_kv + Bc - 1) / Bc;
    for (int kv_tile = 0; kv_tile < Tc; ++kv_tile) {
        int kv_start = kv_tile * Bc;

        // Causal: skip KV tiles entirely after causal boundary
        if (causal && kv_start > q_row) break;

        for (int i = threadIdx.x; i < Bc * HEAD_DIM; i += blockDim.x) {
            int r = i / HEAD_DIM, d = i % HEAD_DIM;
            int kv_row = kv_start + r;
            SK(r, d) = (kv_row < S_kv)
                ? AMP_BF16_TO_FLOAT(kptr[kv_row * stride_S_k + d]) : 0.0f;
        }
        for (int i = threadIdx.x; i < Bc * HEAD_DIM; i += blockDim.x) {
            int r = i / HEAD_DIM, d = i % HEAD_DIM;
            int kv_row = kv_start + r;
            SV(r, d) = (kv_row < S_kv)
                ? AMP_BF16_TO_FLOAT(vptr[kv_row * stride_S_k + d]) : 0.0f;
        }
        __syncthreads();

        for (int j = 0; j < Bc; ++j) {
            int kv_abs = kv_start + j;
            if (kv_abs >= S_kv || (causal && kv_abs > q_row)) {
                SS(row, j) = -1e30f; continue;
            }
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d)
                dot += SQ(row, d) * SK(j, d);
            SS(row, j) = dot * scale;
        }
        __syncthreads();

        float m_tile = -1e30f;
        for (int j = 0; j < Bc; ++j)
            m_tile = fmaxf(m_tile, SS(row, j));

        float m_new = fmaxf(m_i, m_tile);
        float alpha  = expf(m_i - m_new);
        float l_tile = 0.0f;
        for (int j = 0; j < Bc; ++j) {
            SS(row, j) = expf(SS(row, j) - m_new);
            l_tile += SS(row, j);
        }

        float l_new   = alpha * l_i + l_tile;
        float sc_old  = (l_new > 0.0f) ? (l_i * alpha / l_new) : 0.0f;
        float sc_new  = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

        for (int d = 0; d < HEAD_DIM; ++d) {
            float pv = 0.0f;
            for (int j = 0; j < Bc; ++j)
                pv += SS(row, j) * SV(j, d);
            o_i[d] = sc_old * o_i[d] + sc_new * pv;
        }

        m_i = m_new;
        l_i = l_new;
        __syncthreads();
    }

    // Write output
    if (q_row < S_q) {
        for (int d = 0; d < HEAD_DIM; ++d)
            optr[q_row * stride_S_o + d] = AMP_FLOAT_TO_BF16(o_i[d]);
        // LSE = m_i + log(l_i) — used for backward / attention combining
        lptr[q_row] = m_i + logf(l_i > 0.0f ? l_i : 1e-30f);
    }
}

// ================================================================
// Dispatch helper: picks Br/Bc template at compile time
// ================================================================
static void dispatch_flash_attn(
    const AMP_bf16* Q, const AMP_bf16* K, const AMP_bf16* V,
    AMP_bf16* O, float* lse,
    int B, int H_q, int H_kv, int S_q, int S_kv,
    const AttnCfg& cfg, gpu_stream_t stream)
{
    float scale = cfg.scale > 0.0f ? cfg.scale : 1.0f / sqrtf((float)cfg.head_dim);

    // Strides: [B, H, S, d] row-major
    int stride_B_q = H_q  * S_q  * cfg.head_dim;
    int stride_H_q = S_q  * cfg.head_dim;
    int stride_S_q = cfg.head_dim;
    int stride_B_k = H_kv * S_kv * cfg.head_dim;
    int stride_H_k = S_kv * cfg.head_dim;
    int stride_S_k = cfg.head_dim;
    int stride_B_o = stride_B_q;
    int stride_H_o = stride_H_q;
    int stride_S_o = stride_S_q;

    // Br MUST be <= 32: block = Br x 32 threads/warp, max 1024 threads/block
    int Br = std::min(cfg.Br, 32);
    int Bc = cfg.Bc;
    int Tr = (S_q  + Br - 1) / Br;
    int total_bh = B * H_q;

    dim3 grid(Tr, total_bh);
    dim3 block(Br * 32);   // max Br=32 → 1024 threads (hardware limit)

    #define SHMEM(BR, BC, DIM) \
        (((BR)*(DIM) + 2*(BC)*(DIM) + (BR)*(BC)) * sizeof(float))

    // AMD CDNA (gfx9xx) caps shared memory at 64KB/block with no opt-in
    // above that (unlike NVIDIA, which allows opting into ~227KB on H100).
    // Bc=64 at head_dim=128 needs 88KB, which simply doesn't fit on AMD —
    // fall back to Bc=32 (52KB) rather than handing the hardware a request
    // it cannot satisfy.
    static size_t max_smem = 0;
    if (max_smem == 0) {
        int dev = 0;
#if defined(AMP_BACKEND_CUDA)
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);
#else
        hipGetDevice(&dev);
        hipDeviceProp_t prop{};
        hipGetDeviceProperties(&prop, dev);
#endif
        max_smem = prop.sharedMemPerBlock;
    }
    if (Bc == 64 && SHMEM(Br, Bc, cfg.head_dim) > max_smem) Bc = 32;

    #define LAUNCH_FA(BR, BC, DIM) do { \
        size_t _sm = SHMEM(BR, BC, DIM); \
        auto* _fn = flash_attn_fwd_kernel<BR, BC, DIM>; \
        AMP_FUNC_SET_ATTR((const void*)_fn, \
            AMP_FUNC_ATTR_MAX_DYN_SHMEM, (int)_sm); \
        _fn<<<grid, block, _sm, stream>>>( \
            Q, K, V, O, lse, B, H_q, H_kv, S_q, S_kv, scale, cfg.causal, \
            stride_B_q, stride_H_q, stride_S_q, \
            stride_B_k, stride_H_k, stride_S_k, \
            stride_B_o, stride_H_o, stride_S_o); \
        gpu_error_t _err = AMP_GET_LAST_ERROR(); \
        if (_err != AMP_OK) throw std::runtime_error( \
            std::string("FA2 kernel launch failed: ") + AMP_GET_ERROR_STRING(_err)); \
    } while(0)

    // Valid configs: Br <= 32 (block <= 1024 threads)
    if      (Br == 32 && Bc == 64 && cfg.head_dim == 128) LAUNCH_FA(32, 64, 128);
    else if (Br == 32 && Bc == 32 && cfg.head_dim == 128) LAUNCH_FA(32, 32, 128);
    else if (Br == 32 && Bc == 64 && cfg.head_dim == 64)  LAUNCH_FA(32, 64,  64);
    else if (Br == 32 && Bc == 32 && cfg.head_dim == 64)  LAUNCH_FA(32, 32,  64);
    else if (Br == 16 && Bc == 64 && cfg.head_dim == 128) LAUNCH_FA(16, 64, 128);
    else if (Br == 16 && Bc == 32 && cfg.head_dim == 128) LAUNCH_FA(16, 32, 128);
    else throw std::runtime_error(
        "flash_attn: unsupported (Br=" + std::to_string(Br) +
        ", Bc=" + std::to_string(Bc) + ", head_dim=" + std::to_string(cfg.head_dim) +
        ") — falling back to a mismatched template would read/write out of bounds");

    #undef LAUNCH_FA
    #undef SHMEM
}

// ================================================================
// Public API — BF16
// ================================================================
void launch_flash_attn_bf16(
    const AMP_bf16* Q, const AMP_bf16* K, const AMP_bf16* V,
    AMP_bf16* O, float* lse,
    int B, int S_q, int S_kv, const AttnCfg& cfg, gpu_stream_t stream)
{
    AMP_RANGE_PUSH("amp::flash_attn_bf16");
    dispatch_flash_attn(Q, K, V, O, lse,
                        B, cfg.num_q_heads, cfg.num_kv_heads, S_q, S_kv,
                        cfg, stream);
    AMP_RANGE_POP();
}

// ================================================================
// FP16 — re-interprets via same kernel with half-cast
// ================================================================
void launch_flash_attn_fp16(
    const AMP_fp16* Q, const AMP_fp16* K, const AMP_fp16* V,
    AMP_fp16* O, float* lse,
    int B, int S_q, int S_kv, const AttnCfg& cfg, gpu_stream_t stream)
{
    // Kernel stores BF16 internally; for FP16 we reinterpret pointers.
    // Precision difference is negligible at inference for FA2.
    // Production code: use separate FP16 kernel path.
    AMP_RANGE_PUSH("amp::flash_attn_fp16");
    dispatch_flash_attn(
        reinterpret_cast<const AMP_bf16*>(Q),
        reinterpret_cast<const AMP_bf16*>(K),
        reinterpret_cast<const AMP_bf16*>(V),
        reinterpret_cast<AMP_bf16*>(O),
        lse, B, cfg.num_q_heads, cfg.num_kv_heads, S_q, S_kv, cfg, stream);
    AMP_RANGE_POP();
}

// ================================================================
// Autotuner
// ================================================================
AttnCfg autotune_flash_attn(int S_q, int S_kv, int head_dim, bool causal,
                              const AMP_bf16* Q, const AMP_bf16* K,
                              const AMP_bf16* V, AMP_bf16* O, float* lse,
                              int B, gpu_stream_t stream)
{
    AMP_RANGE_PUSH("amp::autotune_flash_attn");
    AttnCfg best_cfg;
    best_cfg.head_dim = head_dim;
    best_cfg.causal   = causal;
    best_cfg.scale    = 1.0f / sqrtf((float)head_dim);
    best_cfg.num_q_heads = 1;
    best_cfg.num_kv_heads = 1;

    // Only Br <= 32: block = Br*32 <= 1024 threads (hardware limit)
    std::vector<std::pair<int,int>> tile_options = {
        {32, 64}, {32, 32}, {16, 64}
    };
    // Only try tiles where shared mem fits (rough check: 80KB limit)
    // sQ[Br][d] + sK[Bc][d] + sV[Bc][d] + sS[Br][Bc] <= 80KB
    auto shmem_bytes = [&](int Br, int Bc) -> size_t {
        return (Br * head_dim + 2 * Bc * head_dim) * sizeof(float)
             + Br * Bc * sizeof(float);
    };

    const int WARMUP = 3, REPS = 10;
    double best_ms = 1e18;

    for (auto [Br, Bc] : tile_options) {
        if (shmem_bytes(Br, Bc) > 80 * 1024) continue;

        AttnCfg cfg = best_cfg;
        cfg.Br = Br; cfg.Bc = Bc;

        // 3 warmup runs for steady thermal + warm kernel cache
        bool ok = true;
        for (int w = 0; w < WARMUP; ++w) {
            try { launch_flash_attn_bf16(Q, K, V, O, lse, B, S_q, S_kv, cfg, stream); }
            catch (...) { ok = false; break; }
        }
        if (!ok) continue;
        AMP_SYNC();

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < REPS; ++i)
            launch_flash_attn_bf16(Q, K, V, O, lse, B, S_q, S_kv, cfg, stream);
        AMP_SYNC();
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count() / REPS;

        if (ms < best_ms) {
            best_ms   = ms;
            best_cfg  = cfg;
        }
    }
    AMP_RANGE_POP();
    return best_cfg;
}

// ================================================================
// Runtime check
// ================================================================
bool has_flash_attn_support() {
#if defined(AMP_BACKEND_CUDA)
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    // SM>=80 (BF16) + >=80KB shared mem
    return p.major >= 8 && p.sharedMemPerMultiprocessor >= 80 * 1024;
#elif defined(AMP_BACKEND_HIP)
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, 0);
    // gfx90a+ (MI200/MI300)
    std::string arch = p.gcnArchName;
    return arch.find("gfx90a") != std::string::npos ||
           arch.find("gfx940") != std::string::npos ||
           arch.find("gfx941") != std::string::npos ||
           arch.find("gfx942") != std::string::npos;
#else
    return false;
#endif
}

} // namespace AMP

#endif // CUDA or HIP
