// bench.cpp - head-to-head baseline vs optimized
#include "portable.hpp"
#include "pool.hpp"
#include "pool_fast.hpp"
#include "cache.hpp"
#include "cache_fast.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <random>
#include <iomanip>
#include <cstring>

using namespace AMP;
using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// =====================================================================
// Bench 1: pool contention, N threads alloc/free in a tight loop
// =====================================================================
template<typename Pool>
double bench_pool_contention(Pool& pool, int n_threads, int ops_per_thread,
                             bool is_fast) {
    std::atomic<uint64_t> ops{0};
    auto worker = [&](int tid) {
        std::mt19937 rng(tid * 7919);
        std::uniform_int_distribution<int> sz(2, 6);
        for (int i = 0; i < ops_per_thread; ++i) {
            int n = sz(rng);
            std::vector<int> ids;
            if constexpr (std::is_same_v<Pool, PagePoolFast>) {
                ids = pool.alloc(n, tid);
            } else {
                ids = pool.alloc("t" + std::to_string(tid), n);
            }
            if (!ids.empty()) {
                pool.free_pages(ids);
                ops.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };
    auto t0 = clk::now();
    std::vector<std::thread> th;
    for (int i = 0; i < n_threads; ++i) th.emplace_back(worker, i);
    for (auto& t : th) t.join();
    double dt_ms = ms_since(t0);
    return ops.load() / (dt_ms / 1000.0);  // ops/sec
}

// =====================================================================
// Bench 2: cache get latency, hot path
// =====================================================================
struct CacheBench { double cold_ms, warm_ms; double speedup; };

CacheBench bench_cache_baseline(int n_keys, int n_reads) {
    CacheStore c("./bench_cache_baseline");
    // populate
    std::vector<std::string> keys;
    for (int i = 0; i < n_keys; ++i) {
        CompileGraph g{"op" + std::to_string(i), {i}, "fp32", "nv:H100", ""};
        auto k = CacheStore::fingerprint(g);
        c.put(k, std::vector<char>(256, (char)i));
        keys.push_back(k);
    }
    // measure: each get hits disk
    auto t0 = clk::now();
    volatile size_t sum = 0;
    for (int i = 0; i < n_reads; ++i) {
        auto v = c.get(keys[i % n_keys]);
        if (v) sum += v->size();
    }
    double dt = ms_since(t0);
    return {dt, dt, 1.0};   // baseline = no warm tier
}

CacheBench bench_cache_fast(int n_keys, int n_reads) {
    CacheStoreFast c("./bench_cache_fast");
    std::vector<std::string> keys;
    for (int i = 0; i < n_keys; ++i) {
        CompileGraph g{"op" + std::to_string(i), {i}, "fp32", "nv:H100", ""};
        auto k = CacheStore::fingerprint(g);
        c.put(k, std::vector<char>(256, (char)i));
        keys.push_back(k);
    }
    // cold: first time, miss L1, hit L2 disk, promote
    auto t0 = clk::now();
    for (auto& k : keys) (void)c.get(k);
    double cold = ms_since(t0);
    // warm: all L1 hits
    auto t1 = clk::now();
    volatile size_t sum = 0;
    for (int i = 0; i < n_reads; ++i) {
        auto v = c.get(keys[i % n_keys]);
        if (v) sum += v->size();
    }
    double warm = ms_since(t1);
    return {cold, warm, 0};
}

int main() {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=========================================================\n"
              << "BENCHMARK: baseline vs optimized (REAL measurements)\n"
              << "Backend: " << AMP_VENDOR << "\n"
              << "=========================================================\n";

    // ---- Pool contention ----
    std::cout << "\n[Bench 1: PagePool — alloc/free, contention test]\n";
    std::cout << "  Threads | Baseline (mutex)  | Optimized (lock-free)  | Speedup\n";
    std::cout << "  --------+-------------------+------------------------+--------\n";
    int OPS = 50'000;
    for (int nt : {1, 2, 4, 8}) {
        PagePool      pb(1024, 4096);
        PagePoolFast  pf(1024, 4096, 16);
        // warmup
        bench_pool_contention(pb, nt, 1000, false);
        bench_pool_contention(pf, nt, 1000, true);
        double base = bench_pool_contention(pb, nt, OPS, false);
        double fast = bench_pool_contention(pf, nt, OPS, true);
        std::cout << "  " << std::setw(6) << nt << "  | "
                  << std::setw(13) << base/1e6 << " Mops | "
                  << std::setw(15) << fast/1e6 << " Mops    | "
                  << std::setw(5) << (fast/base) << "x\n";
    }

    // ---- Cache lookup ----
    std::cout << "\n[Bench 2: CompileCache — get() latency]\n";
    auto cb = bench_cache_baseline(64, 10000);
    auto cf = bench_cache_fast(64, 10000);
    std::cout << "  Baseline (disk only)  : "
              << std::setw(8) << cb.cold_ms << " ms / 10k reads  ("
              << (cb.cold_ms*1000.0/10000) << " μs/op)\n";
    std::cout << "  Fast L1 cold (promote): "
              << std::setw(8) << cf.cold_ms << " ms / 64 reads\n";
    std::cout << "  Fast L1 warm (memory) : "
              << std::setw(8) << cf.warm_ms << " ms / 10k reads  ("
              << (cf.warm_ms*1000.0/10000) << " μs/op)  -> "
              << (cb.cold_ms/cf.warm_ms) << "x faster\n";

    // ---- Context vs competitors ----
    std::cout << "\n=========================================================\n"
              << "COMPARISON VS COMPETITORS (claims, need to verify on hardware):\n"
              << "  vs vLLM v0.11 KVCacheBlock (mutex+linked list):\n"
              << "    Pool ops/sec @ 8 threads -> expected 3-8x faster\n"
              << "  vs JAX persistent_cache (disk only):\n"
              << "    Cache get() hot path -> 100x+ faster (L1 in-memory)\n"
              << "  vs TensorRT-LLM (NV-only, no cross-vendor):\n"
              << "    Not head-to-head; positioning = heterogeneous fleet\n"
              << "=========================================================\n";
    return 0;
}
