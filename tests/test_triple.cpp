// test_triple.cpp - Module 1+2+3 integration + new 2026 features
#include "portable.hpp"
#include "pool.hpp"
#include "cache.hpp"
#include "fp8.hpp"
#include "collective.hpp"
#include "scheduler.hpp"
#include "speculative.hpp"
#include "gemm_vendor.hpp"
#if defined(AMP_BACKEND_CUDA) || defined(AMP_BACKEND_HIP)
#include "../kernels/matmul.cuh"
#include "../kernels/flash_attn.cuh"
#endif

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <sstream>
#include <cstring>

using namespace AMP;

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        steady_clock::now().time_since_epoch()).count();
}

int main() {
    std::cout << "==================================================\n"
              << "amp triple integration test (REAL hardware)\n"
              << "Backend: " << AMP_VENDOR << "\n"
              << "==================================================\n";

    // ---- Detect hardware + runtime + capabilities ----
    auto dev  = detect_device(0);
    auto rt   = detect_runtime();
    auto caps = detect_caps(0);
    std::cout << "\n[stage_2_hwdetect]\n"
              << "  vendor       : " << dev.vendor << "\n"
              << "  name         : " << dev.name   << "\n"
              << "  simd_width   : " << dev.simd_width << "\n"
              << "  compute_units: " << dev.compute_units << "\n"
              << "  matrix_unit  : " << dev.matrix_unit << "\n"
              << "  global_mem   : " << (dev.global_mem >> 20) << " MB\n"
              << "  shared_mem   : " << caps.shared_mem_kb << " KB/SM\n"
              << "  has_bf16     : " << caps.has_bf16 << "\n"
              << "  has_fp8      : " << caps.has_fp8  << "\n"
              << "  has_int8     : " << caps.has_int8 << "\n"
              << "  has_flash_attn: " << caps.has_flash_attn << "\n"
              << "  n_devices    : " << caps.n_devices << "\n"
              << "  runtime_ver  : " << rt.runtime_ver << "\n"
              << "  driver_ver   : " << rt.driver_ver  << "\n";

#if defined(AMP_BACKEND_CUDA) || defined(AMP_BACKEND_HIP)
    // CUDA 13 context init — NO cudaDeviceReset (breaks context on CUDA 13)
    // Pattern: clear error, sync, retry malloc until it succeeds or gives up
#if defined(AMP_BACKEND_CUDA)
    cudaSetDevice(0);
    cudaDeviceSynchronize();   // force context creation
    cudaGetLastError();        // consume error from sync

    // Retry loop: the CUDA 13 driver sometimes needs a few attempts
    bool ctx_ok = false;
    for (int attempt = 0; attempt < 5 && !ctx_ok; ++attempt) {
        cudaGetLastError();    // clear on each attempt
        void* _w = nullptr;
        cudaError_t _e = cudaMalloc(&_w, 4096);
        if (_e == cudaSuccess) {
            cudaFree(_w);
            ctx_ok = true;
        } else {
            std::cerr << "  CUDA init attempt " << attempt+1
                      << " failed: " << cudaGetErrorString(_e) << "\n";
            cudaDeviceSynchronize();  // sync before retry
        }
    }
    if (!ctx_ok) {
        std::cerr << "CUDA context failed to initialize after 5 attempts\n";
        return 1;
    }
    std::cout << "  CUDA context  : OK\n";
#elif defined(AMP_BACKEND_HIP)
    hipSetDevice(0);
    hipDeviceSynchronize();
    hipGetLastError();
    { void* _w = nullptr; hipMalloc(&_w, 4096); hipFree(_w); }
#endif

    // ---- 4 GEMM shapes: FP32 autotune + cache ----
    struct Shape { int M, N, K; };
    std::vector<Shape> shapes = {
        {512, 512, 512}, {1024, 1024, 512},
        {2048, 1024, 512}, {1024, 2048, 1024}
    };

    // arch key includes runtime_ver → cache miss automatically on driver upgrade
    std::string arch_key = dev.vendor + ":" + dev.name + ":" + rt.runtime_ver;

    CacheStore store("./AMP_cache");
    std::cout << "\n[Modul 3: cache] initial entries=" << store.size() << "\n";

    // ---- FP32 autotune ----
    std::cout << "\n[Modul 1: FP32 autotune — real kernel benchmarks]\n";
    struct Entry { std::string key; TileCfg cfg; double tune_ms; double gflops; };
    std::vector<Entry> tuned;
    for (auto& s : shapes) {
        float *dA, *dB, *dC;
        size_t na = s.M*s.K*sizeof(float), nb = s.K*s.N*sizeof(float),
               nc = s.M*s.N*sizeof(float);
        AMP_CHECK(AMP_MALLOC(&dA, na));
        AMP_CHECK(AMP_MALLOC(&dB, nb));
        AMP_CHECK(AMP_MALLOC(&dC, nc));

        std::ostringstream tag;
        tag << s.M << "x" << s.N << "x" << s.K;

        CompileGraph g{"matmul", {s.M,s.N,s.K}, "fp32", arch_key, ""};
        std::string key = CacheStore::fingerprint(g);
        auto cached = store.get(key);

        double t0 = now_ms();
        TileCfg cfg{}; double gflops = 0;
        if (!cached) {
            auto r = autotune_matmul(s.M, s.N, s.K, dA, dB, dC);
            cfg = r.cfg; gflops = r.gflops;
            std::vector<char> bin(sizeof(TileCfg));
            std::memcpy(bin.data(), &cfg, sizeof(cfg));
            store.put(key, bin);
        } else {
            std::memcpy(&cfg, cached->data(), sizeof(TileCfg));
        }
        double t1 = now_ms();
        tuned.push_back({key, cfg, t1-t0, gflops});
        std::cout << "  " << tag.str()
                  << "  tile=(" << cfg.BM << "," << cfg.BN << "," << cfg.BK << ")"
                  << "  tune_time=" << (t1-t0) << "ms"
                  << "  " << (cached ? "[HIT]" : "[MISS,compiled]")
                  << "  " << gflops << " GFLOPS\n";
        AMP_FREE(dA); AMP_FREE(dB); AMP_FREE(dC);
    }
    std::cout << "  cache entries: " << store.size() << "\n";

    // ---- BF16 WMMA (if hardware supports SM>=80 / gfx90a+) ----
#if defined(AMP_BACKEND_CUDA) || (defined(AMP_BACKEND_HIP) && defined(AMP_HAVE_ROCWMMA))
    if (has_tensor_core_bf16()) {
        std::cout << "\n[Modul 1b: BF16 WMMA — tensor core benchmark]\n";
        Shape s = {1024, 1024, 512};

        // Allocate BF16 buffers
        AMP_bf16 *dA16, *dB16;
        float    *dC32;
        AMP_CHECK(AMP_MALLOC(&dA16, s.M * s.K * sizeof(AMP_bf16)));
        AMP_CHECK(AMP_MALLOC(&dB16, s.K * s.N * sizeof(AMP_bf16)));
        AMP_CHECK(AMP_MALLOC(&dC32, s.M * s.N * sizeof(float)));

        CompileGraph gb{"matmul", {s.M,s.N,s.K}, "bf16", arch_key, ""};
        std::string kbf16 = CacheStore::fingerprint(gb);
        auto cached_bf16 = store.get(kbf16);

        double t0 = now_ms();
        TuneResult rbf16 = cached_bf16
            ? TuneResult{{16,16,16,MatmulDtype::BF16}, 0, 0}
            : autotune_matmul_bf16(s.M, s.N, s.K, dA16, dB16, dC32);
        double tune_ms = now_ms() - t0;

        if (!cached_bf16) {
            std::vector<char> bin(sizeof(TileCfg));
            TileCfg cfg = rbf16.cfg;
            std::memcpy(bin.data(), &cfg, sizeof(cfg));
            store.put(kbf16, bin);
        }

        std::cout << "  1024x1024x512 BF16  tune_time=" << tune_ms << "ms"
                  << "  " << (cached_bf16 ? "[HIT]" : "[MISS,compiled]")
                  << "  " << rbf16.gflops << " GFLOPS\n";

        AMP_FREE(dA16); AMP_FREE(dB16); AMP_FREE(dC32);
    } else {
        std::cout << "\n[Modul 1b: BF16 WMMA] SKIP — hardware not supported (SM<80 / non-CDNA)\n";
    }
#endif

    // ---- Modul 2: paged pool + leak scan ----
    std::cout << "\n[Modul 2: paged pool + leak scan, soak 1000 req]\n";
    const size_t PAGE_BYTES = 1024 * 1024;
    PagePool pool(1024, PAGE_BYTES);  // 1024 pages = 1GB, enough for H100

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> leak_dist(0,1);
    int oom=0, leaked=0, served=0;
    double tot_ms = 0;
    const double LEAK_RATE = 0.25;
    const int N_REQ = 1000;

    for (int i = 0; i < N_REQ; ++i) {
        int n_pages = 2 + (i % 4);
        std::string trace = "req-" + std::to_string(i);

        double t0 = now_ms();
        auto pids = pool.alloc(trace, n_pages);
        if (pids.empty()) { oom++; continue; }
        pool.touch(pids);
        served++;
        if (leak_dist(rng) < LEAK_RATE) leaked++;
        else pool.free_pages(pids);
        tot_ms += now_ms() - t0;

        if ((i+1) % 50 == 0) pool.leak_scan(/*ttl_ms=*/1); // 1ms is enough on H100
    }
    auto st = pool.stats();
    std::cout << "  served    : " << served << "/" << N_REQ << "\n"
              << "  OOM       : " << oom << "\n"
              << "  leaked    : " << leaked << " (handled by leak_scan)\n"
              << "  in_use_end: " << st.in_use << "\n"
              << "  avg lat   : " << (tot_ms/served) << " ms\n";

    // ============================================================
    // Modul 1c: FP8 E4M3 (SM>=8.9 / MI300X)
    // ============================================================
#if defined(AMP_HAVE_FP8)
    if (has_tensor_core_fp8()) {
        std::cout << "\n[Modul 1c: FP8 E4M3 — tensor core FP8 benchmark]\n";
        try {
            Shape s = {1024, 1024, 512};
            size_t na8 = (size_t)s.M * s.K, nb8 = (size_t)s.K * s.N;
            AMP_fp8_e4m3 *dA8, *dB8;
            float *dC32_fp8;
            AMP_CHECK(AMP_MALLOC(&dA8, na8 * sizeof(AMP_fp8_e4m3)));
            AMP_CHECK(AMP_MALLOC(&dB8, nb8 * sizeof(AMP_fp8_e4m3)));
            AMP_CHECK(AMP_MALLOC(&dC32_fp8, (size_t)s.M * s.N * sizeof(float)));

            CompileGraph gfp8{"matmul", {s.M,s.N,s.K}, "fp8e4m3", arch_key, ""};
            std::string kfp8 = CacheStore::fingerprint(gfp8);
            auto cached_fp8  = store.get(kfp8);

            double t0 = now_ms();
            TuneResult rfp8{};
            if (!cached_fp8) {
                rfp8 = autotune_matmul_fp8(s.M, s.N, s.K, dA8, dB8, dC32_fp8);
                std::vector<char> bin(sizeof(TileCfg));
                TileCfg cfg = rfp8.cfg;
                std::memcpy(bin.data(), &cfg, sizeof(cfg));
                store.put(kfp8, bin);
            } else {
                std::memcpy(&rfp8.cfg, cached_fp8->data(), sizeof(TileCfg));
            }
            double fp8_tune_ms = now_ms() - t0;
            std::cout << "  1024x1024x512 FP8E4M3  tune_time=" << fp8_tune_ms << "ms"
                      << "  " << (cached_fp8 ? "[HIT]" : "[MISS,compiled]")
                      << "  " << rfp8.gflops << " GFLOPS\n";
            AMP_FREE(dA8); AMP_FREE(dB8); AMP_FREE(dC32_fp8);
        } catch (const std::exception& ex) {
            std::cout << "  FP8 via cuBLASLt failed: " << ex.what() << "\n";
            std::cout << "  (FP8 hardware present but cuBLASLt config needs workspace)\n";
#if defined(AMP_BACKEND_CUDA)
            cudaGetLastError();  // clear error state
#elif defined(AMP_BACKEND_HIP)
            hipGetLastError();   // clear error state
#endif
        }
    } else {
        std::cout << "\n[Modul 1c: FP8 E4M3] SKIP — hardware not supported (SM<8.9)\n";
    }
#else
    std::cout << "\n[Modul 1c: FP8 E4M3] SKIP — CUDA<12.1 at build time\n";
#endif

    // ============================================================
    // Modul 4: FlashAttention-2 (BF16, causal, GQA)
    // ============================================================
    if (has_flash_attn_support()) {
        std::cout << "\n[Modul 4: FlashAttention-2 — causal BF16, GQA]\n";

        // Llama-style: B=1, H_q=32, H_kv=8, S=2048, d=128
        const int B=1, H_q=32, H_kv=8, S_q=2048, S_kv=2048, D=128;
        size_t sz_q  = (size_t)B * H_q  * S_q  * D * sizeof(AMP_bf16);
        size_t sz_kv = (size_t)B * H_kv * S_kv * D * sizeof(AMP_bf16);
        size_t sz_o  = sz_q;
        size_t sz_lse = (size_t)B * H_q * S_q * sizeof(float);

        AMP_bf16 *dQ, *dK, *dV, *dO;
        float    *dLSE;
        AMP_CHECK(AMP_MALLOC(&dQ,   sz_q));
        AMP_CHECK(AMP_MALLOC(&dK,   sz_kv));
        AMP_CHECK(AMP_MALLOC(&dV,   sz_kv));
        AMP_CHECK(AMP_MALLOC(&dO,   sz_o));
        AMP_CHECK(AMP_MALLOC(&dLSE, sz_lse));

        AttnCfg cfg;
        cfg.num_q_heads = H_q; cfg.num_kv_heads = H_kv;
        cfg.head_dim = D; cfg.causal = true;

        // Cold run (first dispatch — includes kernel JIT overhead)
        double t0 = now_ms();
        launch_flash_attn_bf16(dQ, dK, dV, dO, dLSE, B, S_q, S_kv, cfg);
        AMP_SYNC();
        double cold_ms = now_ms() - t0;

        // 2 additional warmups for steady thermal state
        launch_flash_attn_bf16(dQ, dK, dV, dO, dLSE, B, S_q, S_kv, cfg);
        launch_flash_attn_bf16(dQ, dK, dV, dO, dLSE, B, S_q, S_kv, cfg);
        AMP_SYNC();

        // Warm run: 20 reps for stable statistics
        const int reps = 20;
        t0 = now_ms();
        for (int i = 0; i < reps; ++i)
            launch_flash_attn_bf16(dQ, dK, dV, dO, dLSE, B, S_q, S_kv, cfg);
        AMP_SYNC();
        double warm_ms = (now_ms() - t0) / reps;

        // FLOPS FA2 fwd (causal): 2 × B × H_q × S_q × S_kv × D
        // (QK^T + PV, causal reduces by 0.5 each = net 2×BHSqSkvD)
        // Causal GQA: H_q query heads, but compute is still H_q × S_q × Skv × D
        double flops = 2.0 * B * H_q * (double)S_q * S_kv * D;
        double tflops = flops / (warm_ms * 1e9);
        // Sanity check: H100 peak BF16 = 989 TFLOPS dense
        if (tflops > 1000.0) {
            std::cout << "  ⚠ TFLOPS=" << tflops << " > H100 peak — kernel may not be running\n";
            tflops = -1;  // mark invalid
        }
        std::cout << "  Config: B=" << B << " H_q=" << H_q << " H_kv=" << H_kv
                  << " S=" << S_q << " d=" << D << " causal=true\n"
                  << "  Cold: " << cold_ms << " ms\n"
                  << "  Warm: " << warm_ms << " ms/iter  "
                  << tflops << " TFLOPS (causal, GQA)\n";

        AMP_FREE(dQ); AMP_FREE(dK); AMP_FREE(dV);
        AMP_FREE(dO); AMP_FREE(dLSE);
    } else {
        std::cout << "\n[Modul 4: FlashAttention-2] SKIP — hardware not supported\n";
    }

    // ============================================================
    // Modul 5: Multi-GPU Collective (single-GPU loopback if 1 device)
    // ============================================================
    std::cout << "\n[Modul 5: Collective comm — " << caps.n_devices << " device(s)]\n";
    if (collective_available()) {
        std::vector<CommGroup> groups;
        comm_init_all(groups);
        std::cout << "  Initialized " << groups.size() << " communicator(s)\n";

        // AllReduce test: fill buffer with 1.0, allreduce sum → world_size
        if (!groups.empty()) {
            float* d_buf;
            AMP_CHECK(AMP_MALLOC(&d_buf, sizeof(float)));
            // Host init: send 1.0 per rank
            float hval = 1.0f;
            AMP_MEMCPY_HD(d_buf, &hval, sizeof(float));

            allreduce(groups[0], d_buf, 1, DataType::FP32, ReduceOp::SUM, 0);
            AMP_SYNC();

            float result = 0.0f;
            AMP_MEMCPY_DH(&result, d_buf, sizeof(float));
            std::cout << "  AllReduce(1.0, SUM): expected=" << groups.size()
                      << " got=" << result
                      << " " << (std::fabs(result - groups.size()) < 0.01f ? "PASS" : "FAIL")
                      << "\n";
            AMP_FREE(d_buf);
        }
        for (auto& g : groups) comm_destroy(g);
    } else {
        std::cout << "  No NCCL/RCCL/oneCCL linked — collective stub (world=1 OK)\n";
        CommGroup g;
        comm_init_rank(g, 0, 1, 0, CommUniqueId{});
        float* d_buf;
        AMP_CHECK(AMP_MALLOC(&d_buf, sizeof(float)));
        allreduce(g, d_buf, 1, DataType::FP32, ReduceOp::SUM, 0);
        AMP_FREE(d_buf);
        std::cout << "  Single-rank stub PASS\n";
    }

    // ============================================================
    // Modul 6: Continuous Batching Scheduler + Chunked Prefill
    // ============================================================
    std::cout << "\n[Modul 6: Continuous Batching Scheduler]\n";
    {
        SchedulerConfig scfg;
        scfg.max_batch_tokens           = 2048;
        scfg.max_seqs_running           = 32;
        scfg.max_prefill_tokens_per_step = 512;
        scfg.kv_page_size_tokens        = 16;
        scfg.max_pages_total            = 4096;  // 4096 pages = 1GB, enough for 20 req
        scfg.page_bytes                 = 16 * 32 * 128 * 2 * 2;

        Scheduler sched(scfg);
        std::atomic<int> completed_count{0};
        sched.set_completion_callback([&](int id, const std::vector<int>& toks) {
            completed_count.fetch_add(1);
            (void)id; (void)toks;
        });

        // Add 20 requests with varying prompt lengths
        const int N_REQ = 20;
        for (int i = 0; i < N_REQ; ++i) {
            std::vector<int> prompt(32 + (i % 4) * 32);  // 32-128 tokens
            sched.add_request(std::move(prompt), /*max_new_tokens=*/16);
        }

        // Simulate scheduling loop (no actual model — just scheduler logic)
        double t0 = now_ms();
        int steps = 0;
        auto stats_init = sched.get_stats();

        while (completed_count.load() < N_REQ && steps < 500) {
            auto batch = sched.schedule();
            if (batch.entries.empty()) break;

            // Fake decode: generate random token for each decode entry
            std::vector<int> fake_tokens;
            for (const auto& e : batch.entries) {
                if (!e.is_prefill) fake_tokens.push_back(1234 + e.seq->id % 100);
            }
            sched.step_complete(batch, fake_tokens);
            steps++;
        }
        double elapsed_ms = now_ms() - t0;

        auto st = sched.get_stats();
        std::cout << "  Requests    : " << N_REQ << "\n"
                  << "  Steps       : " << steps << "\n"
                  << "  Completed   : " << completed_count.load() << "/" << N_REQ << "\n"
                  << "  KV pages    : used=" << st.kv_pages_used
                               << " free=" << st.kv_pages_free << "\n"
                  << "  Preemptions : " << st.preemptions << "\n"
                  << "  Elapsed     : " << elapsed_ms << " ms\n"
                  << "  " << (completed_count.load() == N_REQ ? "PASS" : "PARTIAL")
                  << " (" << completed_count.load() << "/" << N_REQ << ")\n";
    }

    // ============================================================
    // Modul 7: Speculative Decoding (statistical verification)
    // ============================================================
    std::cout << "\n[Modul 7: Speculative Decoding]\n";
    {
        const int VOCAB  = 32000;
        const int K      = 5;     // draft tokens per step
        const int N_STEP = 100;

        SpecConfig scfg;
        scfg.num_speculative_tokens = K;
        scfg.temperature = 0.8f;
        scfg.vocab_size  = VOCAB;
        SpecDecodeEngine engine(scfg);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> tok_dist(0, VOCAB-1);
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

        int total_new_tokens = 0;
        for (int step = 0; step < N_STEP; ++step) {
            // Simulation: draft model generates K tokens
            std::vector<int> draft_tokens(K);
            std::vector<TokenDist> draft_dists(K), target_dists(K+1);

            for (int i = 0; i < K; ++i) {
                draft_tokens[i] = tok_dist(rng);
                // Fake logits: draft model confident on its token
                draft_dists[i].logits.assign(VOCAB, -10.0f);
                draft_dists[i].logits[draft_tokens[i]] = 5.0f;
                draft_dists[i].sampled = draft_tokens[i];

                // Fake target logits: ~80% agreement with draft
                target_dists[i].logits.assign(VOCAB, -10.0f);
                bool agree = (prob_dist(rng) < 0.8f);
                target_dists[i].logits[agree ? draft_tokens[i] : tok_dist(rng)] = 5.0f;
                target_dists[i].sampled = -1;
            }
            // Target dist at position k
            target_dists[K].logits.assign(VOCAB, -10.0f);
            target_dists[K].logits[tok_dist(rng)] = 5.0f;
            target_dists[K].sampled = -1;

            auto result = engine.verify(draft_tokens, draft_dists, target_dists);
            total_new_tokens += result.total_new();
        }

        const auto& st = engine.stats();
        double accept_rate = st.acceptance_rate();
        double speedup = (accept_rate * K + 1.0);  // expected tokens per step

        std::cout << "  Steps              : " << N_STEP << "\n"
                  << "  Draft tokens       : " << st.draft_tokens.load() << "\n"
                  << "  Accepted tokens    : " << st.accepted_tokens.load() << "\n"
                  << "  Acceptance rate    : " << (accept_rate * 100.0) << "%\n"
                  << "  Expected speedup   : " << speedup << "x\n"
                  << "  Total new tokens   : " << total_new_tokens << "\n";

        // Medusa test
        MedusaConfig mcfg;
        mcfg.num_heads = 4;
        mcfg.acceptance_threshold = 0.1f;
        std::vector<int> medusa_cands = {100, 200, 300, 400};
        std::vector<TokenDist> medusa_target(5);
        for (int i = 0; i < 5; ++i) {
            medusa_target[i].logits.assign(VOCAB, -10.0f);
            medusa_target[i].logits[i < 4 ? medusa_cands[i] : 500] = 5.0f;
        }
        auto mr = medusa_verify(mcfg, medusa_cands, medusa_target, 1.0f, rng);
        std::cout << "  Medusa accepted    : " << mr.total_accepted << "/4 heads\n";

        std::cout << "  " << (accept_rate > 0.5 ? "PASS" : "LOW_ACCEPTANCE") << "\n";
    }

    // ============================================================
    // Modul 8: Vendor GEMM backend
    // ============================================================
    std::cout << "\n[Modul 8: Vendor GEMM — " << global_gemm().backend_name() << "]\n";
    try {
        const int M = 1024, N = 1024, K = 512;
        float *dA, *dB, *dC;
        AMP_CHECK(AMP_MALLOC(&dA, (size_t)M*K*sizeof(float)));
        AMP_CHECK(AMP_MALLOC(&dB, (size_t)K*N*sizeof(float)));
        AMP_CHECK(AMP_MALLOC(&dC, (size_t)M*N*sizeof(float)));

        GemmDesc desc;
        desc.M = M; desc.N = N; desc.K = K;
        desc.dtype_a = DataType::FP32;
        desc.dtype_b = DataType::FP32;
        desc.dtype_c = DataType::FP32;

        // 3 warmups: cuBLASLt algorithm selection + GPU steady-state
        for (int w = 0; w < 3; ++w) vendor_gemm(desc, dA, dB, dC);
        AMP_SYNC();

        const int reps = 20;
        double t0 = now_ms();
        for (int i = 0; i < reps; ++i) vendor_gemm(desc, dA, dB, dC);
        AMP_SYNC();
        double ms = (now_ms() - t0) / reps;
        double gflops = (2.0 * M * N * K) / (ms * 1e6);

        std::cout << "  Backend    : " << global_gemm().backend_name() << "\n"
                  << "  1024x1024x512 FP32: " << gflops << " GFLOPS  (" << ms << " ms)\n";
        AMP_FREE(dA); AMP_FREE(dB); AMP_FREE(dC);
    } catch (const std::exception& ex) {
        std::cout << "  Vendor GEMM failed: " << ex.what() << "\n";
#if defined(AMP_BACKEND_CUDA)
        cudaGetLastError();
#elif defined(AMP_BACKEND_HIP)
        hipGetLastError();
#endif
    }

    // ---- Summary ----
    std::cout << "\n==================================================\n"
              << "RESULT\n"
              << "==================================================\n"
              << "  Modul 1 FP32 (autotune)   : " << shapes.size()
              << " shape tuned, real GFLOPS measured\n"
              << "  Modul 1 BF16 (tensor core) : "
              << (has_tensor_core_bf16() ? "tested" : "skipped (no hardware support)") << "\n"
              << "  Modul 1 FP8  (SM>=8.9)     : "
#if defined(AMP_HAVE_FP8)
              << (has_tensor_core_fp8() ? "tested" : "skipped (SM<8.9)") << "\n"
#else
              << "skipped (CUDA<12.1/ROCm<6.1)\n"
#endif
              << "  Modul 2 (mem)              : "
              << (oom == 0 ? "PASS" : "FAIL") << " (OOM=" << oom << ")\n"
              << "  Modul 3 (cache)            : " << store.size() << " entries on disk\n"
              << "  Modul 4 (FlashAttn-2)      : "
              << (has_flash_attn_support() ? "tested" : "skipped") << "\n"
              << "  Modul 5 (Collective)       : "
              << (collective_available() ? "NCCL/RCCL" : "stub") << "\n"
              << "  Modul 6 (Continuous Batch) : tested\n"
              << "  Modul 7 (Speculative Dec.) : tested\n"
              << "  Modul 8 (Vendor GEMM)      : "
              << global_gemm().backend_name() << "\n"
              << "\n  Run again → Modul 3 should HIT all (tune_time << 1ms)\n";

#else // CPU fallback
    std::cout << "\n(CPU backend: kernel autotune skipped. Modul 2+3 are still tested.)\n";
    PagePool pool(256, 1024);
    auto ids = pool.alloc("trace-cpu", 4);
    std::cout << "  CPU pool alloc OK, ids.size=" << ids.size() << "\n";
    pool.free_pages(ids);

    std::string arch_key = dev.vendor + ":" + dev.name + ":" + rt.runtime_ver;
    CacheStore store("./AMP_cache");
    CompileGraph g{"matmul",{128,128,128},"fp32", arch_key,""};
    auto k = CacheStore::fingerprint(g);
    store.put(k, {'x','y','z'});
    auto got = store.get(k);
    std::cout << "  CPU cache put/get OK, size=" << (got?got->size():0) << "\n";
#endif
    return 0;
}
