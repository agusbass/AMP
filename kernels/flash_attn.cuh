// flash_attn.cuh - FlashAttention-2: fused causal/non-causal self-attention
//
// Paper: "FlashAttention-2: Faster Attention with Better Parallelism and Work
//         Partitioning" — Dao et al., 2023.
//
// Properties:
//   - O(N) SRAM usage: does not materialize the N×N attention matrix to HBM
//   - Online softmax with numerically-stable 2-pass (Tr × Tc tiles)
//   - Causal masking: O(N²/2) FLOP reduction
//   - GQA (Grouped Query Attention): num_kv_heads <= num_q_heads
//   - BF16/FP16 input, FP32 accumulator
//   - CUDA SM>=80 (Ampere) and HIP gfx90a+ (MI200/MI300)
//
// Tile sizes (compile-time template params):
//   Br = query block rows    (default 64)
//   Bc = key/value block rows (default 64)
//   Head dim d must be <= 256
#pragma once
#include "portable.hpp"

#if defined(AMP_BACKEND_CUDA) || defined(AMP_BACKEND_HIP)

namespace AMP {

// ---- Attention configuration ----
struct AttnCfg {
    int  num_q_heads   = 1;
    int  num_kv_heads  = 1;   // GQA: num_kv_heads divides num_q_heads
    int  head_dim      = 128; // must be a multiple of 16
    bool causal        = true;
    float scale        = 0.0f;  // 0 → automatically filled with 1/sqrt(head_dim)

    // Tile sizes — Br <= 32 is mandatory (block = Br*32 <= 1024 threads)
    int Br = 32;
    int Bc = 64;
};

// ---- Output descriptor ----
struct FlashAttnOut {
    double ms;       // execution time
    double tflops;   // effective TFLOPS (with causal reduction)
};

// ---- Launchers ----

// BF16 forward pass
// Q, K, V, O: device pointers, layout [B, H, S, d] (B=batch, H=heads, S=seq, d=head_dim)
// lse: log-sum-exp output [B, H, S] — required for the backward pass
// B = batch_size, S_q = query seq len, S_kv = kv seq len
void launch_flash_attn_bf16(
    const AMP_bf16* Q,  // [B, num_q_heads,  S_q,  d]
    const AMP_bf16* K,  // [B, num_kv_heads, S_kv, d]
    const AMP_bf16* V,  // [B, num_kv_heads, S_kv, d]
    AMP_bf16*       O,  // [B, num_q_heads,  S_q,  d]
    float*          lse,// [B, num_q_heads,  S_q]  — log-sum-exp
    int B, int S_q, int S_kv,
    const AttnCfg& cfg,
    gpu_stream_t stream = 0);

// FP16 forward pass
void launch_flash_attn_fp16(
    const AMP_fp16* Q,
    const AMP_fp16* K,
    const AMP_fp16* V,
    AMP_fp16*       O,
    float*          lse,
    int B, int S_q, int S_kv,
    const AttnCfg& cfg,
    gpu_stream_t stream = 0);

// Autotune: pick the best Br/Bc for (S_q, S_kv, head_dim, hardware)
AttnCfg autotune_flash_attn(int S_q, int S_kv, int head_dim, bool causal,
                              const AMP_bf16* Q, const AMP_bf16* K,
                              const AMP_bf16* V, AMP_bf16* O, float* lse,
                              int B, gpu_stream_t stream = 0);

// Runtime check: whether the device has enough shared mem for FA2
bool has_flash_attn_support();

} // namespace AMP

#endif // CUDA or HIP
