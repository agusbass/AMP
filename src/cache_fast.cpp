// cache_fast.cpp - LRU L1 + disk L2 cache with shared_mutex
// Read path: shared_lock — concurrent readers don't block each other.
// Write path (eviction, put, promotion): unique_lock.
// Clock eviction: O(n) worst case but the read hot path doesn't need a write lock.
#include "cache_fast.hpp"
#include "profiler.hpp"

namespace AMP {

CacheStoreFast::CacheStoreFast(const std::string& root, size_t max_l1_bytes)
    : disk_(root), max_l1_bytes_(max_l1_bytes) {}

size_t CacheStoreFast::l1_count() const {
    std::shared_lock g(smtx_);
    return l1_.size();
}

size_t CacheStoreFast::l1_bytes() const {
    return l1_bytes_.load(std::memory_order_relaxed);
}

// ---- Hot read path: shared_lock ----
std::optional<std::vector<char>> CacheStoreFast::get(const std::string& key) {
    AMP_RANGE_PUSH("amp::cache::get");

    // 1. L1 lookup — shared lock (concurrent safe, no write needed)
    {
        std::shared_lock g(smtx_);
        auto it = l1_.find(key);
        if (it != l1_.end()) {
            // Mark recently used (atomic — no lock upgrade needed)
            it->second.recently_used.store(true, std::memory_order_relaxed);
            stats_.l1_hits.fetch_add(1, std::memory_order_relaxed);
            AMP_RANGE_POP();
            return it->second.data;
        }
    }

    // 2. L2 disk (without lock — CacheStore is path-isolated thread-safe)
    auto bin = disk_.get(key);
    if (!bin) {
        stats_.misses.fetch_add(1, std::memory_order_relaxed);
        AMP_RANGE_POP();
        return std::nullopt;
    }

    // 3. Promote to L1 (unique_lock for write)
    {
        std::unique_lock g(smtx_);
        // Double-check: another thread may have promoted between our checks
        if (l1_.find(key) == l1_.end()) {
            l1_put_locked(key, *bin);
        }
    }
    stats_.l2_hits.fetch_add(1, std::memory_order_relaxed);
    AMP_RANGE_POP();
    return bin;
}

// ---- Write path ----
void CacheStoreFast::put(const std::string& key, const std::vector<char>& bin) {
    AMP_RANGE_PUSH("amp::cache::put");
    disk_.put(key, bin);
    {
        std::unique_lock g(smtx_);
        auto it = l1_.find(key);
        if (it != l1_.end()) {
            // Update existing — adjust byte count
            l1_bytes_.fetch_sub(it->second.data.size(), std::memory_order_relaxed);
            it->second.data = bin;
            it->second.recently_used.store(true, std::memory_order_relaxed);
            l1_bytes_.fetch_add(bin.size(), std::memory_order_relaxed);
        } else {
            l1_put_locked(key, bin);
        }
    }
    AMP_RANGE_POP();
}

// ---- Internal helpers (caller must hold unique_lock) ----
void CacheStoreFast::l1_put_locked(const std::string& key,
                                    const std::vector<char>& bin) {
    evict_to_fit(bin.size());

    Entry e;
    e.data = bin;
    e.recently_used.store(true, std::memory_order_relaxed);
    clock_order_.push_back(key);
    clock_pos_[key] = std::prev(clock_order_.end());
    l1_[key] = std::move(e);
    l1_bytes_.fetch_add(bin.size(), std::memory_order_relaxed);
}

// Clock eviction: scan clock_order_, evict entries with recently_used=false,
// clear recently_used=true entries (give them a second chance).
void CacheStoreFast::evict_to_fit(size_t needed) {
    if (max_l1_bytes_ == 0) return;
    size_t current = l1_bytes_.load(std::memory_order_relaxed);
    if (current + needed <= max_l1_bytes_) return;

    // Two-pass clock scan: first clear bits, second evict if still clear
    for (int pass = 0; pass < 2 && current + needed > max_l1_bytes_; ++pass) {
        auto it = clock_order_.begin();
        while (it != clock_order_.end() && current + needed > max_l1_bytes_) {
            auto l1_it = l1_.find(*it);
            if (l1_it == l1_.end()) {
                it = clock_order_.erase(it);
                continue;
            }
            if (l1_it->second.recently_used.load(std::memory_order_relaxed)) {
                // First chance: clear the bit, skip
                l1_it->second.recently_used.store(false, std::memory_order_relaxed);
                ++it;
            } else {
                // Evict
                size_t sz = l1_it->second.data.size();
                l1_bytes_.fetch_sub(sz, std::memory_order_relaxed);
                current -= sz;
                clock_pos_.erase(*it);
                l1_.erase(l1_it);
                it = clock_order_.erase(it);
                stats_.evictions.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

} // namespace AMP
