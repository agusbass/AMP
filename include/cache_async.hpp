// cache_async.hpp - non-blocking disk writes
// L1 (in-memory, shared_mutex) is read synchronously; disk writes are queued to a worker thread.
// No global static — each instance has its own L1.
#pragma once
#include "cache.hpp"
#include <thread>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <unordered_map>

namespace AMP {

class CacheStoreAsync {
public:
    explicit CacheStoreAsync(const std::string& root);
    ~CacheStoreAsync();

    // get: L1 O(1) on hit; disk read + L1 promote on miss.
    std::optional<std::vector<char>> get(const std::string& key);

    // put: synchronous L1 update (fast) + disk write queued (non-blocking).
    void put(const std::string& key, std::vector<char> binary);

    // flush: blocks until the entire disk write queue has been written.
    void flush();

    struct Stats {
        std::atomic<uint64_t> l1_hits{0};
        std::atomic<uint64_t> writes_queued{0};
        std::atomic<uint64_t> writes_persisted{0};
    };
    Stats& stats() { return stats_; }

private:
    struct WriteJob { std::string key; std::vector<char> bin; };

    CacheStore disk_;   // raw disk store (no L1)

    // Per-instance L1 — no global state
    std::unordered_map<std::string, std::vector<char>> l1_;
    mutable std::shared_mutex l1_mtx_;

    std::thread   writer_;
    std::queue<WriteJob> q_;
    std::mutex    qm_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> busy_{false};  // true while a job is being written to disk

    Stats stats_;
    void writer_loop();
};

} // namespace AMP
