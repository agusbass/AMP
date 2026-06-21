// pool.cpp
#include "pool.hpp"
#include <algorithm>

namespace AMP {

PagePool::PagePool(size_t n_pages, size_t page_bytes)
    : page_bytes_(page_bytes), pages_(n_pages)
{
    for (size_t i = 0; i < n_pages; ++i) {
        Page& p = pages_[i];
        AMP_CHECK(AMP_MALLOC(&p.dptr, page_bytes));
        p.bytes = page_bytes;
        p.freed = true;
        free_list_.push_back(static_cast<int>(i));
    }
}

PagePool::~PagePool() {
    for (auto& p : pages_) {
        if (p.dptr) AMP_FREE(p.dptr);
    }
}

std::vector<int> PagePool::alloc(const std::string& trace_id, int n) {
    std::lock_guard<std::mutex> g(mtx_);
    if ((int)free_list_.size() < n) return {};
    std::vector<int> out;
    out.reserve(n);
    auto now = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        int id = free_list_.back();
        free_list_.pop_back();
        Page& p = pages_[id];
        p.refcount = 1;
        p.owner = trace_id;
        p.alloc_ts = p.touch_ts = now;
        p.freed = false;
        out.push_back(id);
    }
    alloc_cnt_ += n;
    return out;
}

void PagePool::free_pages(const std::vector<int>& ids) {
    std::lock_guard<std::mutex> g(mtx_);
    for (int id : ids) {
        Page& p = pages_[id];
        if (!p.freed) {
            p.refcount = 0;
            p.owner.clear();
            p.freed = true;
            free_list_.push_back(id);
            free_cnt_++;
        }
    }
}

void PagePool::touch(const std::vector<int>& ids) {
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> g(mtx_);
    for (int id : ids) pages_[id].touch_ts = now;
}

int PagePool::leak_scan(int ttl_ms) {
    std::lock_guard<std::mutex> g(mtx_);
    auto now = std::chrono::steady_clock::now();
    std::vector<int> orphan;
    for (size_t i = 0; i < pages_.size(); ++i) {
        Page& p = pages_[i];
        if (p.freed) continue;
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - p.touch_ts).count();
        if (age > ttl_ms) orphan.push_back((int)i);
    }
    for (int id : orphan) {
        Page& p = pages_[id];
        p.refcount = 0; p.owner.clear(); p.freed = true;
        free_list_.push_back(id);
        free_cnt_++;
    }
    return (int)orphan.size();
}

PagePool::Stats PagePool::stats() const {
    std::lock_guard<std::mutex> g(mtx_);
    int in_use = 0;
    for (auto& p : pages_) if (!p.freed) in_use++;
    return {(int)pages_.size(), in_use, (int)free_list_.size(),
            alloc_cnt_.load(), free_cnt_.load()};
}

} // namespace AMP
