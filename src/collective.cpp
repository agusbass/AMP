// collective.cpp - NCCL/RCCL/oneCCL wrappers
#include "collective.hpp"
#include <stdexcept>
#include <cstring>

namespace AMP {

// ================================================================
// Backend-specific helpers
// ================================================================

#if (defined(AMP_HAVE_NCCL) && defined(AMP_BACKEND_CUDA)) || \
    (defined(AMP_HAVE_RCCL) && defined(AMP_BACKEND_HIP))

// NCCL and RCCL have identical APIs — abstracted via typedef

#if defined(AMP_BACKEND_CUDA)
  #define AMP_NCCLCHECK(call) do { \
    ncclResult_t _r = (call); \
    if (_r != ncclSuccess) \
        throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(_r)); \
  } while(0)
  using nccl_comm_t       = ncclComm_t;
  using nccl_unique_id_t  = ncclUniqueId;
  #define NCCL_GET_UNIQUE_ID ncclGetUniqueId
  #define NCCL_COMM_INIT_RANK ncclCommInitRank
  #define NCCL_COMM_INIT_ALL  ncclCommInitAll
  #define NCCL_COMM_DESTROY   ncclCommDestroy
  #define NCCL_ALLREDUCE      ncclAllReduce
  #define NCCL_BROADCAST      ncclBroadcast
  #define NCCL_ALLGATHER      ncclAllGather
  #define NCCL_REDUCE_SCATTER ncclReduceScatter
  #define NCCL_SEND           ncclSend
  #define NCCL_RECV           ncclRecv
  #define NCCL_SUCCESS        ncclSuccess
  inline ncclDataType_t to_nccl_dtype(CollDType dt) {
      switch (dt) {
          case DataType::FP32:  return ncclFloat32;
          case DataType::BF16:  return ncclBfloat16;
          case DataType::FP16:  return ncclFloat16;
          case DataType::INT8:  return ncclInt8;
          case DataType::INT32: return ncclInt32;
          default: throw std::runtime_error("unsupported dtype for NCCL");
      }
  }
  inline ncclRedOp_t to_nccl_op(ReduceOp op) {
      switch (op) {
          case ReduceOp::SUM:  return ncclSum;
          case ReduceOp::MAX:  return ncclMax;
          case ReduceOp::MIN:  return ncclMin;
          case ReduceOp::PROD: return ncclProd;
          default: return ncclSum;
      }
  }
#else  // HIP RCCL
  #define AMP_NCCLCHECK(call) do { \
    ncclResult_t _r = (call); \
    if (_r != ncclSuccess) \
        throw std::runtime_error(std::string("RCCL error: ") + ncclGetErrorString(_r)); \
  } while(0)
  using nccl_comm_t       = ncclComm_t;
  using nccl_unique_id_t  = ncclUniqueId;
  #define NCCL_GET_UNIQUE_ID ncclGetUniqueId
  #define NCCL_COMM_INIT_RANK ncclCommInitRank
  #define NCCL_COMM_INIT_ALL  ncclCommInitAll
  #define NCCL_COMM_DESTROY   ncclCommDestroy
  #define NCCL_ALLREDUCE      ncclAllReduce
  #define NCCL_BROADCAST      ncclBroadcast
  #define NCCL_ALLGATHER      ncclAllGather
  #define NCCL_REDUCE_SCATTER ncclReduceScatter
  #define NCCL_SEND           ncclSend
  #define NCCL_RECV           ncclRecv
  #define NCCL_SUCCESS        ncclSuccess
  inline ncclDataType_t to_nccl_dtype(CollDType dt) {
      switch (dt) {
          case DataType::FP32:  return ncclFloat;
          case DataType::BF16:  return ncclBfloat16;
          case DataType::FP16:  return ncclHalf;
          case DataType::INT8:  return ncclInt8;
          case DataType::INT32: return ncclInt32;
          default: throw std::runtime_error("unsupported dtype for RCCL");
      }
  }
  inline ncclRedOp_t to_nccl_op(ReduceOp op) {
      switch (op) {
          case ReduceOp::SUM:  return ncclSum;
          case ReduceOp::MAX:  return ncclMax;
          case ReduceOp::MIN:  return ncclMin;
          case ReduceOp::PROD: return ncclProd;
          default: return ncclSum;
      }
  }
#endif

#define AMP_HAVE_COLLECTIVE 1

static nccl_comm_t* to_comm(void* h) { return reinterpret_cast<nccl_comm_t*>(h); }

CommUniqueId comm_get_unique_id() {
    CommUniqueId uid;
    nccl_unique_id_t nid;
    static_assert(sizeof(nid) <= CommUniqueId::kBytes,
                  "CommUniqueId::kBytes too small for nccl unique id");
    AMP_NCCLCHECK(NCCL_GET_UNIQUE_ID(&nid));
    std::memcpy(uid.data, &nid, sizeof(nid));
    return uid;
}

void comm_init_rank(CommGroup& g, int rank, int world, int device_id,
                    const CommUniqueId& uid) {
#if defined(AMP_BACKEND_CUDA)
    cudaSetDevice(device_id);
#elif defined(AMP_BACKEND_HIP)
    hipSetDevice(device_id);
#endif
    nccl_unique_id_t nid;
    std::memcpy(&nid, uid.data, sizeof(nid));
    auto* comm = new nccl_comm_t;
    AMP_NCCLCHECK(NCCL_COMM_INIT_RANK(comm, world, nid, rank));
    g.handle = comm; g.rank = rank; g.world = world;
    g.device_id = device_id; g.initialized = true;
}

void comm_init_all(std::vector<CommGroup>& groups) {
    int n = local_device_count();
    groups.resize(n);
    std::vector<nccl_comm_t> comms(n);
    AMP_NCCLCHECK(NCCL_COMM_INIT_ALL(n, comms.data(), n, nullptr));
    for (int i = 0; i < n; ++i) {
        auto* c = new nccl_comm_t(comms[i]);
        groups[i].handle = c; groups[i].rank = i; groups[i].world = n;
        groups[i].device_id = i; groups[i].initialized = true;
    }
}

void comm_destroy(CommGroup& g) {
    if (!g.initialized) return;
    AMP_NCCLCHECK(NCCL_COMM_DESTROY(*to_comm(g.handle)));
    delete to_comm(g.handle);
    g.handle = nullptr; g.initialized = false;
}

void allreduce(CommGroup& g, void* buf, size_t count,
               CollDType dtype, ReduceOp op, gpu_stream_t stream) {
    AMP_NCCLCHECK(NCCL_ALLREDUCE(buf, buf, count,
        to_nccl_dtype(dtype), to_nccl_op(op), *to_comm(g.handle), stream));
}

void broadcast(CommGroup& g, void* buf, size_t count,
               CollDType dtype, int root, gpu_stream_t stream) {
    AMP_NCCLCHECK(NCCL_BROADCAST(buf, buf, count,
        to_nccl_dtype(dtype), root, *to_comm(g.handle), stream));
}

void allgather(CommGroup& g,
               const void* sendbuf, void* recvbuf,
               size_t count, CollDType dtype, gpu_stream_t stream) {
    AMP_NCCLCHECK(NCCL_ALLGATHER(sendbuf, recvbuf, count,
        to_nccl_dtype(dtype), *to_comm(g.handle), stream));
}

void reduce_scatter(CommGroup& g,
                    const void* sendbuf, void* recvbuf,
                    size_t recv_count, CollDType dtype, ReduceOp op,
                    gpu_stream_t stream) {
    AMP_NCCLCHECK(NCCL_REDUCE_SCATTER(sendbuf, recvbuf, recv_count,
        to_nccl_dtype(dtype), to_nccl_op(op), *to_comm(g.handle), stream));
}

void send(CommGroup& g, const void* buf, size_t count,
          CollDType dtype, int peer, gpu_stream_t stream) {
    AMP_NCCLCHECK(NCCL_SEND(buf, count,
        to_nccl_dtype(dtype), peer, *to_comm(g.handle), stream));
}

void recv(CommGroup& g, void* buf, size_t count,
          CollDType dtype, int peer, gpu_stream_t stream) {
    AMP_NCCLCHECK(NCCL_RECV(buf, count,
        to_nccl_dtype(dtype), peer, *to_comm(g.handle), stream));
}

void barrier(CommGroup& g, gpu_stream_t stream) {
    // NCCL has no explicit barrier — use allreduce over 1 element
    float tmp = 0.0f;
    float* d_tmp;
#if defined(AMP_BACKEND_CUDA)
    cudaMalloc(&d_tmp, sizeof(float));
    cudaMemset(d_tmp, 0, sizeof(float));
#else
    hipMalloc(&d_tmp, sizeof(float));
    hipMemset(d_tmp, 0, sizeof(float));
#endif
    AMP_NCCLCHECK(NCCL_ALLREDUCE(d_tmp, d_tmp, 1, ncclFloat32,
        ncclSum, *to_comm(g.handle), stream));
#if defined(AMP_BACKEND_CUDA)
    cudaFree(d_tmp);
#else
    hipFree(d_tmp);
#endif
    (void)tmp;
}

int local_device_count() {
    int n = 0;
#if defined(AMP_BACKEND_CUDA)
    cudaGetDeviceCount(&n);
#elif defined(AMP_BACKEND_HIP)
    hipGetDeviceCount(&n);
#endif
    return n > 0 ? n : 1;
}

bool collective_available() { return true; }

// ================================================================
#elif defined(AMP_HAVE_ONECCL) && defined(AMP_BACKEND_SYCL)
// ================================================================
// oneCCL backend

#define AMP_HAVE_COLLECTIVE 1

CommUniqueId comm_get_unique_id() { return CommUniqueId{}; }  // oneCCL doesn't require one

void comm_init_rank(CommGroup& g, int rank, int world, int device_id,
                    const CommUniqueId&) {
    auto* comm = new ccl::communicator(ccl::create_communicator(world, rank,
        ccl::create_kvs(ccl::kvs::address{})));
    g.handle = comm; g.rank = rank; g.world = world;
    g.device_id = device_id; g.initialized = true;
}

void comm_init_all(std::vector<CommGroup>& groups) {
    int n = local_device_count();
    groups.resize(n);
    for (int i = 0; i < n; ++i)
        comm_init_rank(groups[i], i, n, i, CommUniqueId{});
}

void comm_destroy(CommGroup& g) {
    if (!g.initialized) return;
    delete reinterpret_cast<ccl::communicator*>(g.handle);
    g.handle = nullptr; g.initialized = false;
}

// oneCCL helpers (simplified — production code would pass sycl::queue)
static ccl::communicator& to_comm(CommGroup& g) {
    return *reinterpret_cast<ccl::communicator*>(g.handle);
}
static ccl::datatype to_ccl_dtype(CollDType dt) {
    switch (dt) {
        case DataType::FP32:  return ccl::datatype::float32;
        case DataType::BF16:  return ccl::datatype::bfloat16;
        case DataType::FP16:  return ccl::datatype::float16;
        case DataType::INT8:  return ccl::datatype::int8;
        case DataType::INT32: return ccl::datatype::int32;
        default: throw std::runtime_error("unsupported dtype for oneCCL");
    }
}
static ccl::reduction to_ccl_op(ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM:  return ccl::reduction::sum;
        case ReduceOp::MAX:  return ccl::reduction::max;
        case ReduceOp::MIN:  return ccl::reduction::min;
        case ReduceOp::PROD: return ccl::reduction::prod;
        default: return ccl::reduction::sum;
    }
}

void allreduce(CommGroup& g, void* buf, size_t count,
               CollDType dtype, ReduceOp op, gpu_stream_t) {
    ccl::allreduce(buf, buf, count, to_ccl_dtype(dtype), to_ccl_op(op),
                   to_comm(g)).wait();
}

void broadcast(CommGroup& g, void* buf, size_t count,
               CollDType dtype, int root, gpu_stream_t) {
    ccl::broadcast(buf, count, to_ccl_dtype(dtype), root, to_comm(g)).wait();
}

void allgather(CommGroup& g, const void* sbuf, void* rbuf,
               size_t count, CollDType dtype, gpu_stream_t) {
    ccl::allgatherv(sbuf, count, rbuf, std::vector<size_t>(g.world, count),
                    to_ccl_dtype(dtype), to_comm(g)).wait();
}

void reduce_scatter(CommGroup& g, const void* sbuf, void* rbuf,
                    size_t rc, CollDType dtype, ReduceOp op, gpu_stream_t) {
    ccl::reduce_scatter(sbuf, rbuf, rc, to_ccl_dtype(dtype), to_ccl_op(op),
                        to_comm(g)).wait();
}

void send(CommGroup& g, const void* buf, size_t count, CollDType dtype,
          int peer, gpu_stream_t) {
    ccl::send(buf, count, to_ccl_dtype(dtype), peer, to_comm(g)).wait();
}

void recv(CommGroup& g, void* buf, size_t count, CollDType dtype,
          int peer, gpu_stream_t) {
    ccl::recv(buf, count, to_ccl_dtype(dtype), peer, to_comm(g)).wait();
}

void barrier(CommGroup& g, gpu_stream_t) {
    ccl::barrier(to_comm(g)).wait();
}

int local_device_count() {
    try {
        return (int)sycl::platform::get_platforms().size();  // rough estimate
    } catch (...) { return 1; }
}

bool collective_available() { return true; }

// ================================================================
#else
// ================================================================
// Stub: no collective backend linked

CommUniqueId comm_get_unique_id() { return CommUniqueId{}; }

void comm_init_rank(CommGroup& g, int rank, int world, int device_id,
                    const CommUniqueId&) {
    g.rank = rank; g.world = world; g.device_id = device_id;
    g.initialized = true;
}

void comm_init_all(std::vector<CommGroup>& groups) {
    groups.resize(1);
    comm_init_rank(groups[0], 0, 1, 0, CommUniqueId{});
}

void comm_destroy(CommGroup& g) { g.initialized = false; }

static void check_collective() {
    throw std::runtime_error(
        "No collective backend linked. "
        "Build with -DAMP_HAVE_NCCL, -DAMP_HAVE_RCCL, or -DAMP_HAVE_ONECCL.");
}

void allreduce(CommGroup& g, void*, size_t, CollDType, ReduceOp, gpu_stream_t) {
    if (g.world > 1) check_collective();
}
void broadcast(CommGroup& g, void*, size_t, CollDType, int, gpu_stream_t) {
    if (g.world > 1) check_collective();
}
void allgather(CommGroup& g, const void*, void*, size_t, CollDType, gpu_stream_t) {
    if (g.world > 1) check_collective();
}
void reduce_scatter(CommGroup& g, const void*, void*, size_t, CollDType, ReduceOp, gpu_stream_t) {
    if (g.world > 1) check_collective();
}
void send(CommGroup& g, const void*, size_t, CollDType, int, gpu_stream_t) {
    if (g.world > 1) check_collective();
}
void recv(CommGroup& g, void*, size_t, CollDType, int, gpu_stream_t) {
    if (g.world > 1) check_collective();
}
void barrier(CommGroup&, gpu_stream_t) {}

int local_device_count() {
#if defined(AMP_BACKEND_CUDA)
    int n = 0; cudaGetDeviceCount(&n); return n > 0 ? n : 1;
#elif defined(AMP_BACKEND_HIP)
    int n = 0; hipGetDeviceCount(&n); return n > 0 ? n : 1;
#else
    return 1;
#endif
}

bool collective_available() { return false; }

#endif

} // namespace AMP
