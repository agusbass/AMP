// pool_numa.cpp
#include "pool_numa.hpp"
#include <memory>

namespace AMP {

PagePoolNUMA::PagePoolNUMA(size_t pages_per_node, size_t page_bytes)
    : pages_per_node_(pages_per_node)
{
#if defined(AMP_HAVE_NUMA)
    n_nodes_ = numa_available() != -1 ? numa_max_node() + 1 : 1;
#else
    n_nodes_ = 1;
#endif
    per_node_.reserve(n_nodes_);
    for (int i = 0; i < n_nodes_; ++i) {
#if defined(AMP_HAVE_NUMA)
        // Bind future allocations to node i before creating the pool
        if (numa_available() != -1) numa_run_on_node(i);
#endif
        per_node_.emplace_back(std::make_unique<PagePoolFast>(
            pages_per_node, page_bytes, 16));
    }
}

std::vector<int> PagePoolNUMA::alloc(int n, int tid) {
    int node = current_numa_node();
    if (node < 0 || node >= n_nodes_) node = tid % n_nodes_;
    auto local = per_node_[node]->alloc(n, tid);
    std::vector<int> out;
    out.reserve(local.size());
    for (int id : local) out.push_back(global_id(node, id));
    if (out.empty()) {
        // fallback: try another node
        for (int alt = 0; alt < n_nodes_; ++alt) {
            if (alt == node) continue;
            auto l = per_node_[alt]->alloc(n, tid);
            if (!l.empty()) {
                for (int id : l) out.push_back(global_id(alt, id));
                return out;
            }
        }
    }
    return out;
}

void PagePoolNUMA::free_pages(const std::vector<int>& gids) {
    std::vector<std::vector<int>> per(n_nodes_);
    for (int g : gids) { auto [n, l] = split(g); per[n].push_back(l); }
    for (int n = 0; n < n_nodes_; ++n)
        if (!per[n].empty()) per_node_[n]->free_pages(per[n]);
}

void PagePoolNUMA::touch(const std::vector<int>& gids) {
    std::vector<std::vector<int>> per(n_nodes_);
    for (int g : gids) { auto [n, l] = split(g); per[n].push_back(l); }
    for (int n = 0; n < n_nodes_; ++n)
        if (!per[n].empty()) per_node_[n]->touch(per[n]);
}

int PagePoolNUMA::leak_scan(int ttl_ms) {
    int total = 0;
    for (auto& p : per_node_) total += p->leak_scan(ttl_ms);
    return total;
}

PagePoolFast::Stats PagePoolNUMA::stats() const {
    PagePoolFast::Stats agg{0, 0, 0, 0};
    for (auto& p : per_node_) {
        auto s = p->stats();
        agg.total += s.total; agg.in_use += s.in_use;
        agg.alloc_cnt += s.alloc_cnt; agg.free_cnt += s.free_cnt;
    }
    return agg;
}

} // namespace AMP
