// pool.hpp - Module 2: paged KV allocator
#pragma once
#include "portable.hpp"
#include <vector>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <atomic>

namespace AMP {

struct Page {
    void* dptr = nullptr;
    size_t bytes = 0;
    std::atomic<int> refcount{0};
    std::string owner;        // trace_id
    std::chrono::steady_clock::time_point alloc_ts;
    std::chrono::steady_clock::time_point touch_ts;
    bool freed = true;
};

class PagePool {
public:
    PagePool(size_t n_pages, size_t page_bytes);
    ~PagePool();

    // Allocate n_pages from the pool. Returns page ids, or empty if OOM.
    std::vector<int> alloc(const std::string& trace_id, int n_pages);
    void free_pages(const std::vector<int>& ids);
    void touch(const std::vector<int>& ids);

    // Background sweep: reclaim pages whose last_touch > ttl_ms.
    int leak_scan(int ttl_ms);

    // Stats
    struct Stats { int total, in_use, free; uint64_t alloc_cnt, free_cnt; };
    Stats stats() const;

private:
    size_t page_bytes_;
    std::vector<Page> pages_;
    std::vector<int> free_list_;
    mutable std::mutex mtx_;
    std::atomic<uint64_t> alloc_cnt_{0}, free_cnt_{0};
};

} // namespace AMP
