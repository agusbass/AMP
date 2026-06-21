// cache_async.cpp - truly async: L1 sync, disk async via worker thread
#include "cache_async.hpp"

namespace AMP {

CacheStoreAsync::CacheStoreAsync(const std::string& root) : disk_(root) {
    writer_ = std::thread(&CacheStoreAsync::writer_loop, this);
}

CacheStoreAsync::~CacheStoreAsync() {
    stop_ = true;
    cv_.notify_all();
    if (writer_.joinable()) writer_.join();
}

std::optional<std::vector<char>> CacheStoreAsync::get(const std::string& key) {
    // L1 lookup (shared lock — concurrent readers don't block each other)
    {
        std::shared_lock g(l1_mtx_);
        auto it = l1_.find(key);
        if (it != l1_.end()) {
            stats_.l1_hits.fetch_add(1, std::memory_order_relaxed);
            return it->second;
        }
    }
    // L2: disk (sync; infrequent after warm-up)
    auto bin = disk_.get(key);
    if (bin) {
        std::unique_lock g(l1_mtx_);
        l1_[key] = *bin;   // promote to L1
    }
    return bin;
}

void CacheStoreAsync::put(const std::string& key, std::vector<char> bin) {
    // L1 update sync (caller doesn't block on disk I/O)
    {
        std::unique_lock g(l1_mtx_);
        l1_[key] = bin;
    }
    // Queue disk write
    {
        std::lock_guard g(qm_);
        q_.push({key, std::move(bin)});
    }
    stats_.writes_queued.fetch_add(1, std::memory_order_relaxed);
    cv_.notify_one();
}

void CacheStoreAsync::flush() {
    std::unique_lock g(qm_);
    cv_.wait(g, [this] { return q_.empty() && !busy_.load(std::memory_order_relaxed); });
}

void CacheStoreAsync::writer_loop() {
    while (true) {
        WriteJob job;
        {
            std::unique_lock g(qm_);
            cv_.wait(g, [this] { return !q_.empty() || stop_; });
            if (q_.empty() && stop_) return;
            job = std::move(q_.front());
            q_.pop();
            busy_.store(true, std::memory_order_relaxed);
        }
        disk_.put(job.key, job.bin);
        stats_.writes_persisted.fetch_add(1, std::memory_order_relaxed);
        {
            // Write fully persisted before signalling flush() waiters.
            std::lock_guard g(qm_);
            busy_.store(false, std::memory_order_relaxed);
            if (q_.empty()) cv_.notify_all();
        }
    }
}

} // namespace AMP
