// collective.hpp - multi-GPU collective communication abstraction
// Backend:
//   CUDA → NCCL (nccl.h, -lnccl)
//   HIP  → RCCL (rccl.h, -lrccl)
//   SYCL → oneCCL (oneapi/ccl.hpp, -lccl)
//   CPU  → stub (single-process no-op)
//
// Build with:
//   -DAMP_HAVE_NCCL  (CUDA + NCCL)
//   -DAMP_HAVE_RCCL  (HIP + RCCL)
//   -DAMP_HAVE_ONECCL (SYCL + oneCCL)
//
// Without the flag: compiles but all ops throw UnsupportedError.
#pragma once
#include "portable.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

#if defined(AMP_HAVE_NCCL) && defined(AMP_BACKEND_CUDA)
  #include <nccl.h>
#endif
#if defined(AMP_HAVE_RCCL) && defined(AMP_BACKEND_HIP)
  #include <rccl/rccl.h>
#endif
#if defined(AMP_HAVE_ONECCL) && defined(AMP_BACKEND_SYCL)
  #include <oneapi/ccl.hpp>
#endif

namespace AMP {

// ---- CommGroup: one communicator per process ----
struct CommGroup {
    void*  handle      = nullptr;  // ncclComm_t / rcclComm_t / ccl::communicator*
    int    rank        = 0;
    int    world       = 1;
    int    device_id   = 0;
    bool   initialized = false;
};

// ---- Opaque unique ID for bootstrap (broadcast via MPI/file/env) ----
struct CommUniqueId {
    static constexpr int kBytes = 128;
    uint8_t data[kBytes] = {};
};

// ---- Reduce op ----
enum class ReduceOp { SUM, MAX, MIN, PROD };

// ---- DType for collective (superset of DataType) ----
// Mapping to ncclDataType_t / ccl_datatype_t is done in collective.cpp
using CollDType = DataType;

// ================================================================
// Lifecycle
// ================================================================

// Create a unique ID on rank 0, then broadcast it to all ranks before comm_init_rank.
CommUniqueId comm_get_unique_id();

// Init communicator for one rank in a separate process (multi-process).
// nccl_id must be identical across all ranks (broadcast out-of-band).
void comm_init_rank(CommGroup& g, int rank, int world, int device_id,
                    const CommUniqueId& uid);

// Init all GPUs in a single process (single-process multi-GPU).
// Fills groups[0..n_gpu-1].
void comm_init_all(std::vector<CommGroup>& groups);

void comm_destroy(CommGroup& g);

// ================================================================
// Collective ops — all async on `stream`
// The caller is responsible for syncing the stream after all ops complete.
// ================================================================

// AllReduce: sum/max/min/prod across all ranks, result identical on all ranks
void allreduce(CommGroup& g, void* buf, size_t count,
               CollDType dtype, ReduceOp op, gpu_stream_t stream);

// Broadcast: root → all ranks
void broadcast(CommGroup& g, void* buf, size_t count,
               CollDType dtype, int root, gpu_stream_t stream);

// AllGather: each rank sends sendbuf, all ranks receive recvbuf (world * count)
void allgather(CommGroup& g,
               const void* sendbuf, void* recvbuf,
               size_t count, CollDType dtype, gpu_stream_t stream);

// ReduceScatter: allreduce then scatter
void reduce_scatter(CommGroup& g,
                    const void* sendbuf, void* recvbuf,
                    size_t recv_count, CollDType dtype, ReduceOp op,
                    gpu_stream_t stream);

// Point-to-point
void send(CommGroup& g, const void* buf, size_t count,
          CollDType dtype, int peer, gpu_stream_t stream);
void recv(CommGroup& g, void* buf, size_t count,
          CollDType dtype, int peer, gpu_stream_t stream);

// Barrier (all ranks wait)
void barrier(CommGroup& g, gpu_stream_t stream);

// ================================================================
// Helpers
// ================================================================

// Number of GPUs available on this node
int local_device_count();

// Whether the collective backend is available (linked)
bool collective_available();

} // namespace AMP
