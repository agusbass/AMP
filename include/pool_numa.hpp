// pool_numa.hpp - NUMA-aware variant of PagePoolFast
// Strategy: bind shards to NUMA nodes. Threads allocate from their
// NUMA-local shard -> avoids cross-socket cache misses & QPI traffic.
//
// Build with: -DAMP_HAVE_NUMA -lnuma  (Linux)
// Without the flag: degrades gracefully to plain PagePoolFast.
#pragma once
#include "pool_fast.hpp"
#include <memory>
#include <utility>

#if defined(AMP_HAVE_NUMA)
  #include <numa.h>
  #include <sched.h>   // sched_getcpu()
#endif

namespace AMP {

class PagePoolNUMA {
public:
    PagePoolNUMA(size_t pages_per_node, size_t page_bytes);
    std::vector<int> alloc(int n_pages, int thread_id);
    void free_pages(const std::vector<int>& global_ids);
    void touch(const std::vector<int>& ids);
    int leak_scan(int ttl_ms);
    int n_nodes() const { return n_nodes_; }
    PagePoolFast::Stats stats() const;

private:
    int n_nodes_;
    std::vector<std::unique_ptr<PagePoolFast>> per_node_;
    size_t pages_per_node_;

    int current_numa_node() const {
#if defined(AMP_HAVE_NUMA)
        int cpu = sched_getcpu();
        return numa_node_of_cpu(cpu);
#else
        return 0;
#endif
    }
    int global_id(int node, int local_id) const {
        return node * (int)pages_per_node_ + local_id;
    }
    std::pair<int,int> split(int gid) const {
        return {gid / (int)pages_per_node_, gid % (int)pages_per_node_};
    }
};

} // namespace AMP
