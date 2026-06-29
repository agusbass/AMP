// portable.hpp - vendor abstraction layer
// Build flags:
//   -DAMP_BACKEND_CUDA  (NVIDIA)
//   -DAMP_BACKEND_HIP   (AMD)
//   -DAMP_BACKEND_SYCL  (Intel oneAPI / Gaudi)
//   -DAMP_BACKEND_CPU   (fallback, for sanity testing without a GPU)
// Optional:
//   -DAMP_HAVE_FP8      (auto-set if CUDA>=12.1 SM>=8.9 or ROCm>=6.1)
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

// Universal data type enum — used in matmul, quant, collective
enum class DataType : uint8_t {
    FP32    = 0,
    BF16    = 1,
    FP16    = 2,
    FP8E4M3 = 3,  // 8-bit unit, exponent 4, mantissa 3
    FP8E5M2 = 4,  // 8-bit unit, exponent 5, mantissa 2
    INT8    = 5,
    INT4    = 6,  // packed 2 per byte
    INT32   = 7,
};

#if defined(AMP_BACKEND_CUDA)
  #include <cuda_runtime.h>
  #include <cuda.h>          // cuDriverGetVersion
  #include <nvrtc.h>
  // FP8: CUDA 12.1+ header (SM>=8.9 at runtime, SM>=9.0 for Hopper TMA)
  #if defined(__CUDACC__) && defined(CUDA_VERSION) && CUDA_VERSION >= 12010
    #include <cuda_fp8.h>
    #define AMP_HAVE_FP8 1
    using AMP_fp8_e4m3 = __nv_fp8_e4m3;
    using AMP_fp8_e5m2 = __nv_fp8_e5m2;
  #endif
  // FP16 host type
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  using AMP_bf16 = __nv_bfloat16;
  using AMP_fp16 = __half;
  using gpu_error_t   = cudaError_t;
  using gpu_stream_t  = cudaStream_t;
  #define AMP_VENDOR "nvidia"
  // cudaGetLastError() before malloc: clear sticky error from the CUDA 13 init sequence
  #define AMP_MALLOC(p,n)       (cudaGetLastError(), cudaMalloc(p,n))
  #define AMP_FREE(p)           cudaFree(p)
  #define AMP_MEMCPY_HD(d,s,n)  cudaMemcpy(d,s,n,cudaMemcpyHostToDevice)
  #define AMP_MEMCPY_DH(d,s,n)  cudaMemcpy(d,s,n,cudaMemcpyDeviceToHost)
  #define AMP_MEMCPY_ASYNC_HD(d,s,n,st)  cudaMemcpyAsync(d,s,n,cudaMemcpyHostToDevice,st)
  #define AMP_MEMCPY_ASYNC_DH(d,s,n,st)  cudaMemcpyAsync(d,s,n,cudaMemcpyDeviceToHost,st)
  #define AMP_STREAM_CREATE(s)  cudaStreamCreate(s)
  #define AMP_STREAM_DESTROY(s) cudaStreamDestroy(s)
  #define AMP_STREAM_SYNC(s)    cudaStreamSynchronize(s)
  #define AMP_SYNC()            cudaDeviceSynchronize()
  #define AMP_OK                cudaSuccess
  #define AMP_BF16_TO_FLOAT(x)   __bfloat162float(x)
  #define AMP_FLOAT_TO_BF16(x)   __float2bfloat16(x)
  #define AMP_FUNC_SET_ATTR(fn,attr,val) cudaFuncSetAttribute(fn,attr,val)
  #define AMP_FUNC_ATTR_MAX_DYN_SHMEM    cudaFuncAttributeMaxDynamicSharedMemorySize
  #define AMP_GET_LAST_ERROR()   cudaGetLastError()
  #define AMP_GET_ERROR_STRING(e) cudaGetErrorString(e)

#elif defined(AMP_BACKEND_HIP)
  #include <hip/hip_runtime.h>
  #include <hip/hiprtc.h>
  // FP8: ROCm 6.1+ (gfx940/941/942 = MI300X)
  #if __has_include(<hip/hip_fp8.h>)
    #include <hip/hip_fp8.h>
    #define AMP_HAVE_FP8 1
    using AMP_fp8_e4m3 = __hip_fp8_e4m3;
    using AMP_fp8_e5m2 = __hip_fp8_e5m2;
  #endif
  #include <hip/hip_bfloat16.h>
  #include <hip/hip_fp16.h>
  using AMP_bf16 = hip_bfloat16;
  using AMP_fp16 = __half;
  using gpu_error_t   = hipError_t;
  using gpu_stream_t  = hipStream_t;
  #define AMP_VENDOR "amd"
  #define AMP_MALLOC(p,n)       hipMalloc(p,n)
  #define AMP_FREE(p)           hipFree(p)
  #define AMP_MEMCPY_HD(d,s,n)  hipMemcpy(d,s,n,hipMemcpyHostToDevice)
  #define AMP_MEMCPY_DH(d,s,n)  hipMemcpy(d,s,n,hipMemcpyDeviceToHost)
  #define AMP_MEMCPY_ASYNC_HD(d,s,n,st)  hipMemcpyAsync(d,s,n,hipMemcpyHostToDevice,st)
  #define AMP_MEMCPY_ASYNC_DH(d,s,n,st)  hipMemcpyAsync(d,s,n,hipMemcpyDeviceToHost,st)
  #define AMP_STREAM_CREATE(s)  hipStreamCreate(s)
  #define AMP_STREAM_DESTROY(s) hipStreamDestroy(s)
  #define AMP_STREAM_SYNC(s)    hipStreamSynchronize(s)
  #define AMP_SYNC()            hipDeviceSynchronize()
  #define AMP_OK                hipSuccess
  #define AMP_BF16_TO_FLOAT(x)   (float(x))
  #define AMP_FLOAT_TO_BF16(x)   (hip_bfloat16(x))
  #define AMP_FUNC_SET_ATTR(fn,attr,val) hipFuncSetAttribute(fn,attr,val)
  #define AMP_FUNC_ATTR_MAX_DYN_SHMEM    hipFuncAttributeMaxDynamicSharedMemorySize
  #define AMP_GET_LAST_ERROR()   hipGetLastError()
  #define AMP_GET_ERROR_STRING(e) hipGetErrorString(e)

#elif defined(AMP_BACKEND_SYCL)
  #include <sycl/sycl.hpp>
  // XMX matrix extension for joint_matrix GEMM (Intel Gaudi/Arc Xe-HPC)
  #if __has_include(<sycl/ext/intel/experimental/matrix/matrix.hpp>)
    #include <sycl/ext/intel/experimental/matrix/matrix.hpp>
    #define AMP_HAVE_XMX 1
  #endif
  using gpu_error_t   = int;
  using gpu_stream_t  = sycl::queue*;
  using AMP_fp16      = sycl::half;
  // bfloat16 lives under sycl::ext::oneapi in current oneAPI releases;
  // kernels/matmul_sycl.cpp used to reference a plain sycl::bfloat16 that
  // doesn't exist at all, confirmed by an actual CI build against a real
  // oneAPI DPC++ toolchain ("no type named 'bfloat16' in namespace 'sycl'").
  #if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #include <sycl/ext/oneapi/bfloat16.hpp>
    using AMP_bf16 = sycl::ext::oneapi::bfloat16;
  #else
    using AMP_bf16 = sycl::half;
  #endif
  #define AMP_VENDOR "intel"
  namespace AMP { namespace detail {
    inline sycl::queue& default_q() {
      static sycl::queue q{sycl::gpu_selector_v,
          sycl::property_list{sycl::property::queue::enable_profiling{}}};
      return q;
    }
    inline int sycl_malloc(void** p, size_t n) {
      *p = sycl::malloc_device(n, default_q());
      return *p ? 0 : -1;
    }
    inline int sycl_free(void* p)  { sycl::free(p, default_q()); return 0; }
    inline int sycl_memcpy(void* d, const void* s, size_t n)
      { default_q().memcpy(d, s, n).wait(); return 0; }
    inline int sycl_memcpy_async(void* d, const void* s, size_t n, sycl::queue* q)
      { (q ? *q : default_q()).memcpy(d, s, n); return 0; }
  }}
  #define AMP_MALLOC(p,n)                  AMP::detail::sycl_malloc(p,n)
  #define AMP_FREE(p)                      AMP::detail::sycl_free(p)
  #define AMP_MEMCPY_HD(d,s,n)             AMP::detail::sycl_memcpy(d,s,n)
  #define AMP_MEMCPY_DH(d,s,n)             AMP::detail::sycl_memcpy(d,s,n)
  #define AMP_MEMCPY_ASYNC_HD(d,s,n,st)   AMP::detail::sycl_memcpy_async(d,s,n,st)
  #define AMP_MEMCPY_ASYNC_DH(d,s,n,st)   AMP::detail::sycl_memcpy_async(d,s,n,st)
  #define AMP_STREAM_CREATE(s)   do { *(s) = new sycl::queue{sycl::gpu_selector_v}; } while(0)
  #define AMP_STREAM_DESTROY(s)  do { delete (s); } while(0)
  #define AMP_STREAM_SYNC(s)     ((s)->wait(), 0)
  #define AMP_SYNC()             (AMP::detail::default_q().wait(), 0)
  #define AMP_OK                 0

#else  // AMP_BACKEND_CPU
  using gpu_error_t   = int;
  using gpu_stream_t  = void*;
  using AMP_fp16      = uint16_t;  // opaque on CPU fallback
  #define AMP_VENDOR "cpu"
  #define AMP_MALLOC(p,n)                (*(p)=std::malloc(n), 0)
  #define AMP_FREE(p)                    (std::free(p), 0)
  #define AMP_MEMCPY_HD(d,s,n)           (std::memcpy(d,s,n), 0)
  #define AMP_MEMCPY_DH(d,s,n)           (std::memcpy(d,s,n), 0)
  #define AMP_MEMCPY_ASYNC_HD(d,s,n,st)  (std::memcpy(d,s,n), 0)
  #define AMP_MEMCPY_ASYNC_DH(d,s,n,st)  (std::memcpy(d,s,n), 0)
  #define AMP_STREAM_CREATE(s)           (*(s)=nullptr, 0)
  #define AMP_STREAM_DESTROY(s)          0
  #define AMP_STREAM_SYNC(s)             0
  #define AMP_SYNC()                     0
  #define AMP_OK                         0
#endif

// AMP_CHECK: host-side error gate. GPU kernels never call this directly.
#if defined(AMP_BACKEND_CUDA)
  #define AMP_ERR_STR(e) cudaGetErrorString(e)
#elif defined(AMP_BACKEND_HIP)
  #define AMP_ERR_STR(e) hipGetErrorString(e)
#else
  #define AMP_ERR_STR(e) "(no error string)"
#endif

#define AMP_CHECK(call) do { \
    auto _e = (call); \
    if (_e != AMP_OK) throw std::runtime_error( \
        std::string("AMP error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
        " — " + AMP_ERR_STR(_e)); \
} while(0)

// Capability flags — filled in by detect_device()
struct DeviceCaps {
    bool has_bf16      = false;  // BF16 WMMA/MFMA/XMX
    bool has_fp8       = false;  // FP8 E4M3/E5M2 WMMA (SM>=8.9 / MI300X / Gaudi3)
    bool has_fp16      = false;  // FP16 tensor core
    bool has_int8      = false;  // INT8 tensor core (DP4A/IDOT/DPAS)
    bool has_flash_attn = false; // enough SRAM for tiled FA2 (>=80KB shared mem)
    int  shared_mem_kb = 0;      // max shared memory per SM/CU/EU
    int  n_devices     = 1;      // number of physical GPUs in the node (for multi-GPU)
};

// Runtime architecture detection
struct DeviceInfo {
    std::string vendor;      // "nvidia" | "amd" | "intel" | "cpu"
    std::string name;        // "H100" | "MI300X" | "PVC" | "cpu"
    int simd_width;          // warp/wave/subgroup
    int compute_units;       // SM/CU/EU
    size_t global_mem;
    std::string matrix_unit; // "tensor_core" | "mfma" | "xmx" | "none"
    int sm_major, sm_minor;  // SM/GFX version (0 for CPU/SYCL if unknown)
};

// Runtime version info — used to version cache entries across driver upgrades.
struct RuntimeInfo {
    std::string runtime_ver; // "12.6" (CUDA) | "6.1" (ROCm) | "2024.2" (oneAPI)
    std::string driver_ver;  // "560.35" (NVIDIA) | same as runtime for HIP/SYCL
};

DeviceInfo  detect_device(int device_id = 0);
RuntimeInfo detect_runtime();
DeviceCaps  detect_caps(int device_id = 0);
int         device_count();
