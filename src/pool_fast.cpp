// pool_fast.cpp
#include "pool_fast.hpp"

namespace AMP {

PagePoolFast::PagePoolFast(size_t n_pages, size_t page_bytes, int n_shards)
    : page_bytes_(page_bytes), n_shards_(n_shards),
      dptrs_(n_pages), meta_(n_pages),
      heads_(n_shards), next_(n_pages)
{
    for (size_t i = 0; i < n_pages; ++i) {
        AMP_CHECK(AMP_MALLOC(&dptrs_[i], page_bytes));
        meta_[i].shard.store(i % n_shards, std::memory_order_relaxed);
        meta_[i].touch_us.store(0, std::memory_order_relaxed);
    }
    // Build per-shard free stack.
    for (int s = 0; s < n_shards; ++s)
        heads_[s].head.store(-1, std::memory_order_relaxed);
    for (size_t i = 0; i < n_pages; ++i) {
        int s = i % n_shards;
        int old = heads_[s].head.load(std::memory_order_relaxed);
        next_[i].store(old, std::memory_order_relaxed);
        heads_[s].head.store((int)i, std::memory_order_relaxed);
    }
}

PagePoolFast::~PagePoolFast() {
    for (auto p : dptrs_) if (p) AMP_FREE(p);
}

std::vector<int> PagePoolFast::alloc(int n, int thread_hash) {
    std::vector<int> out;
    out.reserve(n);
    uint64_t ts = now_us();
    int s = ((unsigned)thread_hash) % n_shards_;
    int tries = n_shards_;
    while ((int)out.size() < n && tries-- > 0) {
        auto& h = heads_[s].head;
        int top = h.load(std::memory_order_acquire);
        while (top != -1) {
            int nxt = next_[top].load(std::memory_order_relaxed);
            if (h.compare_exchange_weak(top, nxt,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                meta_[top].touch_us.store(ts, std::memory_order_release);
                out.push_back(top);
                if ((int)out.size() == n) break;
                top = h.load(std::memory_order_acquire);
            }
        }
        s = (s + 1) % n_shards_;     // next shard
    }
    if ((int)out.size() < n) {
        // failed — return what was already taken
        free_pages(out);
        return {};
    }
    alloc_cnt_.fetch_add(n, std::memory_order_relaxed);
    return out;
}

void PagePoolFast::free_pages(const std::vector<int>& ids) {
    for (int id : ids) {
        meta_[id].touch_us.store(0, std::memory_order_release);
        int s = meta_[id].shard.load(std::memory_order_relaxed);
        auto& h = heads_[s].head;
        int top = h.load(std::memory_order_relaxed);
        do {
            next_[id].store(top, std::memory_order_relaxed);
        } while (!h.compare_exchange_weak(top, id,
                    std::memory_order_release, std::memory_order_relaxed));
    }
    free_cnt_.fetch_add(ids.size(), std::memory_order_relaxed);
}

void PagePoolFast::touch(const std::vector<int>& ids) {
    uint64_t ts = now_us();
    for (int id : ids) meta_[id].touch_us.store(ts, std::memory_order_release);
}

int PagePoolFast::leak_scan(int ttl_ms) {
    uint64_t now = now_us();
    uint64_t ttl_us = (uint64_t)ttl_ms * 1000;
    std::vector<int> orphan;
    for (size_t i = 0; i < meta_.size(); ++i) {
        uint64_t t = meta_[i].touch_us.load(std::memory_order_acquire);
        if (t != 0 && now - t > ttl_us) orphan.push_back((int)i);
    }
    free_pages(orphan);
    return (int)orphan.size();
}

PagePoolFast::Stats PagePoolFast::stats() const {
    int in_use = 0;
    for (auto& m : meta_)
        if (m.touch_us.load(std::memory_order_acquire) != 0) in_use++;
    return {(int)meta_.size(), in_use,
            alloc_cnt_.load(), free_cnt_.load()};
}

} // namespace AMP
