// bench_full.cpp - benchmark all three optimizations + emit JSON
#include "portable.hpp"
#include "pool.hpp"
#include "pool_fast.hpp"
#include "pool_numa.hpp"
#include "cache.hpp"
#include "cache_fast.hpp"
#include "cache_async.hpp"
#include "autotune_sh.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <random>
#include <iomanip>

using namespace AMP;
using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// -------- pool contention --------
template<typename Pool, bool USE_TID>
double bench_pool(Pool& pool, int nt, int ops) {
    std::atomic<uint64_t> done{0};
    auto worker = [&](int tid) {
        std::mt19937 rng(tid * 7919);
        std::uniform_int_distribution<int> sz(2, 6);
        for (int i = 0; i < ops; ++i) {
            int n = sz(rng);
            std::vector<int> ids;
            if constexpr (USE_TID) ids = pool.alloc(n, tid);
            else                   ids = pool.alloc("t" + std::to_string(tid), n);
            if (!ids.empty()) { pool.free_pages(ids); done.fetch_add(1); }
        }
    };
    auto t0 = clk::now();
    std::vector<std::thread> th;
    for (int i = 0; i < nt; ++i) th.emplace_back(worker, i);
    for (auto& t : th) t.join();
    return done.load() / (ms_since(t0) / 1000.0);
}

// -------- cache put throughput --------
template<typename Cache>
double bench_cache_put(Cache& c, int n_keys) {
    auto t0 = clk::now();
    for (int i = 0; i < n_keys; ++i) {
        CompileGraph g{"op", {i}, "fp32", "nv:H100", ""};
        c.put(CacheStore::fingerprint(g), std::vector<char>(1024, (char)i));
    }
    return ms_since(t0);
}

// -------- autotuner SH vs grid --------
struct DummyCfg { int a, b, c; };
double dummy_measure(const DummyCfg& cfg, int reps) {
    // Simulation of a "kernel": quadratic bowl function + noise.
    double base = (cfg.a-64)*(cfg.a-64)*0.001
                + (cfg.b-128)*(cfg.b-128)*0.001
                + (cfg.c-32)*(cfg.c-32)*0.001 + 1.0;
    std::mt19937 rng(cfg.a*1000 + cfg.b*100 + cfg.c + reps);
    std::normal_distribution<double> n(0, 0.05 / std::sqrt((double)reps));
    return base + n(rng);
}

int main(int argc, char** argv) {
    // Portable stdout: use "-" as a sentinel (not /dev/stdout, which is POSIX-only)
    std::string out_path = (argc > 1) ? argv[1] : "-";
    bool json = (argc > 2 && std::string(argv[2]) == "--json");

    std::ostringstream js;
    js << "{\n  \"backend\": \"" << AMP_VENDOR << "\",\n";
    js << "  \"hardware_threads\": " << std::thread::hardware_concurrency() << ",\n";

    // ===== Bench A: Pool baseline vs fast vs NUMA =====
    if (!json) std::cout << "\n[Pool contention]\n  threads | baseline | fast | numa\n";
    js << "  \"pool\": {\n    \"threads\": [], \"baseline_mops\": [], "
                          "\"fast_mops\": [], \"numa_mops\": []\n  },\n";
    std::vector<int> thread_list = {1, 2, 4, 8};
    std::vector<double> base_v, fast_v, numa_v;
    for (int nt : thread_list) {
        PagePool      pb(1024, 4096);
        PagePoolFast  pf(1024, 4096, 16);
        PagePoolNUMA  pn(1024, 4096);   // pages_per_node
        // warmup
        bench_pool<PagePool, false>(pb, nt, 500);
        bench_pool<PagePoolFast, true>(pf, nt, 500);
        bench_pool<PagePoolNUMA, true>(pn, nt, 500);
        double b = bench_pool<PagePool, false>(pb, nt, 30000);
        double f = bench_pool<PagePoolFast, true>(pf, nt, 30000);
        double n = bench_pool<PagePoolNUMA, true>(pn, nt, 30000);
        base_v.push_back(b); fast_v.push_back(f); numa_v.push_back(n);
        if (!json)
            std::cout << "  " << std::setw(6) << nt << "  | "
                      << std::fixed << std::setprecision(2)
                      << (b/1e6) << "M  | " << (f/1e6) << "M  | " << (n/1e6) << "M\n";
    }

    // ===== Bench B: Cache put — sync (fast) vs async =====
    CacheStoreFast  csf("./bench_csf");
    CacheStoreAsync csa("./bench_csa");
    int N_PUT = 2000;
    double t_sync = bench_cache_put(csf, N_PUT);
    double t_async = bench_cache_put(csa, N_PUT);
    csa.flush();
    if (!json) {
        std::cout << "\n[Cache put " << N_PUT << " entries]\n";
        std::cout << "  sync (fast)  : " << t_sync  << " ms  ("
                  << (t_sync*1000/N_PUT) << " μs/put)\n";
        std::cout << "  async        : " << t_async << " ms  ("
                  << (t_async*1000/N_PUT) << " μs/put)  -> "
                  << (t_sync/t_async) << "x speedup on hot path\n";
    }

    // ===== Bench C: Autotuner — grid vs successive halving =====
    std::vector<DummyCfg> cfgs;
    for (int a : {16,32,48,64,80,96,112,128})
        for (int b : {32,64,96,128,160,192})
            for (int c : {16,24,32,40,48,56,64})
                cfgs.push_back({a,b,c});

    auto t0 = clk::now();
    auto rg = grid_search<DummyCfg>(cfgs, dummy_measure, 32);
    double t_grid = ms_since(t0);
    t0 = clk::now();
    auto rs = successive_halving<DummyCfg>(cfgs, dummy_measure, 1, 32, 2.0);
    double t_sh = ms_since(t0);

    if (!json) {
        std::cout << "\n[Autotune " << cfgs.size() << " configs]\n";
        std::cout << "  grid search       : " << rg.total_evals
                  << " evals  best_score=" << rg.best_score
                  << "  wall=" << t_grid << " ms\n";
        std::cout << "  successive halving: " << rs.total_evals
                  << " evals  best_score=" << rs.best_score
                  << "  wall=" << t_sh << " ms  -> "
                  << ((double)rg.total_evals / rs.total_evals) << "x fewer evals\n";
    }

    // ===== Emit JSON =====
    js << "  \"pool_results\": {\n";
    js << "    \"threads\": [";
    for (size_t i=0;i<thread_list.size();++i) js << thread_list[i] << (i+1<thread_list.size()?",":"");
    js << "],\n    \"baseline_mops\": [";
    for (size_t i=0;i<base_v.size();++i) js << std::fixed << std::setprecision(2) << base_v[i]/1e6 << (i+1<base_v.size()?",":"");
    js << "],\n    \"fast_mops\": [";
    for (size_t i=0;i<fast_v.size();++i) js << fast_v[i]/1e6 << (i+1<fast_v.size()?",":"");
    js << "],\n    \"numa_mops\": [";
    for (size_t i=0;i<numa_v.size();++i) js << numa_v[i]/1e6 << (i+1<numa_v.size()?",":"");
    js << "]\n  },\n";
    js << "  \"cache_put\": {\n";
    js << "    \"n_entries\": " << N_PUT << ",\n";
    js << "    \"sync_ms\": " << t_sync << ",\n";
    js << "    \"async_ms\": " << t_async << ",\n";
    js << "    \"speedup\": " << (t_sync/t_async) << "\n  },\n";
    js << "  \"autotune\": {\n";
    js << "    \"n_configs\": " << cfgs.size() << ",\n";
    js << "    \"grid_evals\": " << rg.total_evals << ", \"grid_wall_ms\": " << t_grid << ",\n";
    js << "    \"sh_evals\": " << rs.total_evals   << ", \"sh_wall_ms\": " << t_sh << ",\n";
    js << "    \"eval_reduction\": " << ((double)rg.total_evals/rs.total_evals) << "\n  }\n}\n";

    if (json) {
        if (out_path == "-") {
            std::cout << js.str();
        } else {
            std::ofstream f(out_path);
            if (!f) { std::cerr << "Cannot open: " << out_path << "\n"; return 1; }
            f << js.str();
            std::cerr << "JSON written to: " << out_path << "\n";
        }
    }
    return 0;
}
