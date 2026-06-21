// pool_fast.hpp - Module 2 OPTIMIZED
// Strategy: sharded free list (N shards, hash by thread id) -> reduce contention.
// Hot path: 1 atomic CAS vs std::mutex (~10-100ns vs 1-10μs under contention).
#pragma once
#include "portable.hpp"
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstring>

namespace AMP {

class PagePoolFast {
public:
    struct PageMeta {
        std::atomic<uint64_t> touch_us{0};   // microsec since epoch; 0 = freed
        std::atomic<uint32_t> shard{0};
        // padding to prevent false sharing
        char _pad[64 - sizeof(std::atomic<uint64_t>) - sizeof(std::atomic<uint32_t>)];
    };

    PagePoolFast(size_t n_pages, size_t page_bytes, int n_shards = 16);
    ~PagePoolFast();

    // Hot path: lock-free pop from shard.
    std::vector<int> alloc(int n_pages, int thread_hash);
    void free_pages(const std::vector<int>& ids);
    void touch(const std::vector<int>& ids);
    int leak_scan(int ttl_ms);

    struct Stats { int total, in_use; uint64_t alloc_cnt, free_cnt; };
    Stats stats() const;

private:
    size_t page_bytes_;
    int n_shards_;
    std::vector<void*> dptrs_;
    std::vector<PageMeta> meta_;
    // free list per-shard, lock-free stack via atomic head
    struct ShardHead { std::atomic<int> head{-1}; char _pad[60]; };
    std::vector<ShardHead> heads_;
    std::vector<std::atomic<int>> next_;   // next-pointer per page (for stack)
    std::atomic<uint64_t> alloc_cnt_{0}, free_cnt_{0};

    static uint64_t now_us() {
        using namespace std::chrono;
        return duration_cast<microseconds>(
            steady_clock::now().time_since_epoch()).count();
    }
};

} // namespace AMP
