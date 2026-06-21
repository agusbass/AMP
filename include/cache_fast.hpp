// cache_fast.hpp - Module 3 OPTIMIZED
// 2-tier: L1 in-memory + L2 disk.
// Hot path L1 hit: ~50ns vs JAX persistent cache which always hits disk ~1ms.
//
// Concurrency strategy (2026 standard):
//   L1 reads  → shared_lock  (concurrent readers not blocked)
//   L1 writes → unique_lock  (eviction, put, L2→L1 promotion)
//   Eviction: "clock" approximate LRU — avoid write-lock-on-every-read.
//     Each entry has an atomic "recently_used" bit.
//     Clock hand scan on eviction: clear bits, evict if already clear.
//     Trade-off: slightly suboptimal eviction vs strict LRU, but the read path
//     becomes O(1) concurrent with no lock contention.
#pragma once
#include "cache.hpp"
#include <unordered_map>
#include <list>
#include <vector>
#include <shared_mutex>
#include <atomic>
#include <mutex>

namespace AMP {

class CacheStoreFast {
public:
    // max_l1_bytes: L1 memory limit (bytes). 0 = unlimited (not recommended for production).
    explicit CacheStoreFast(const std::string& root,
                             size_t max_l1_bytes = 512ULL * 1024 * 1024);

    // Hot path L1: shared_lock (concurrent reads), ~50ns.
    // L2 miss: disk read + promotion (unique_lock).
    std::optional<std::vector<char>> get(const std::string& key);

    // Put to disk + L1; clock eviction if L1 is full.
    void put(const std::string& key, const std::vector<char>& binary);

    struct Stats {
        std::atomic<uint64_t> l1_hits{0}, l2_hits{0}, misses{0}, evictions{0};
    };
    Stats&  stats()      { return stats_; }
    size_t  l1_count()   const;
    size_t  l1_bytes()   const;
    size_t  disk_size()  const { return disk_.size(); }

private:
    CacheStore disk_;

    struct Entry {
        std::vector<char>        data;
        mutable std::atomic<bool> recently_used{true};

        Entry() = default;
        Entry(Entry&& o) noexcept
            : data(std::move(o.data)),
              recently_used(o.recently_used.load(std::memory_order_relaxed)) {}
        Entry& operator=(Entry&& o) noexcept {
            data = std::move(o.data);
            recently_used.store(o.recently_used.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
            return *this;
        }
        Entry(const Entry&) = delete;
        Entry& operator=(const Entry&) = delete;
    };
    std::unordered_map<std::string, Entry> l1_;
    std::list<std::string>                 clock_order_;  // insertion order
    std::unordered_map<std::string, std::list<std::string>::iterator> clock_pos_;

    std::atomic<size_t>    l1_bytes_{0};
    size_t                 max_l1_bytes_;

    // shared_mutex: shared for reads, unique for writes
    mutable std::shared_mutex smtx_;
    Stats stats_;

    // Evict until there is room for `needed` bytes (must be called with unique_lock held).
    void evict_to_fit(size_t needed);

    // Internal put-to-L1 without locking (caller must hold unique_lock).
    void l1_put_locked(const std::string& key, const std::vector<char>& bin);
};

} // namespace AMP
