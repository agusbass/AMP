// gemm_vendor.hpp - cuBLASLt (CUDA) / rocBLAS (HIP) vendor GEMM backend
//
// Why this matters:
//   Hand-rolled WMMA kernels (matmul.cu) fall far short of vendor library peak.
//   cuBLASLt on H100 FP8: ~3000 TFLOPS. Hand-rolled: ~400-600 TFLOPS.
//   For production, always use the vendor library as the GEMM backend.
//
// Dispatch hierarchy (selected automatically):
//   1. cuBLASLt  (CUDA, all dtypes including FP8 with per-tensor scale)
//   2. rocBLAS   (HIP, BF16/FP8 on MI300X)
//   3. oneMKL    (SYCL, BF16 on Xe-HPC)
//   4. Fallback  → hand-rolled kernel in matmul.cu (CPU or older GPU)
//
// Build:
//   CUDA: -DAMP_HAVE_CUBLAS  → link -lcublasLt
//   HIP:  -DAMP_HAVE_ROCBLAS → link -lrocblas
//   SYCL: -DAMP_HAVE_ONEMKL  → link -lmkl_sycl
#pragma once
#include "portable.hpp"
#include <cstddef>

#if defined(AMP_HAVE_CUBLAS) && defined(AMP_BACKEND_CUDA)
  #include <cublasLt.h>
#endif
#if defined(AMP_HAVE_ROCBLAS) && defined(AMP_BACKEND_HIP)
  #include <rocblas/rocblas.h>
#endif

namespace AMP {

// ---- Epilogue fusions (cuBLASLt feature, ignored on fallback) ----
enum class GemmEpilogue {
    NONE   = 0,  // C = alpha * A*B + beta * C
    RELU   = 1,  // C = ReLU(alpha * A*B + beta * C)
    BIAS   = 2,  // C = alpha * A*B + bias
    GELU   = 3,  // C = GELU(alpha * A*B)
};

// ---- GEMM descriptor ----
struct GemmDesc {
    // Matrix dimensions
    int M = 0, N = 0, K = 0;

    // Data types
    DataType dtype_a  = DataType::BF16;
    DataType dtype_b  = DataType::BF16;
    DataType dtype_c  = DataType::FP32;

    // Scaling (FP8 + INT8 quantized paths)
    float alpha     = 1.0f;
    float beta      = 0.0f;
    // Per-tensor scale pointers on device (nullptr = no scaling)
    const float* d_scale_a = nullptr;
    const float* d_scale_b = nullptr;
    float*       d_scale_c = nullptr;  // output scale (FP8 output)
    float*       d_amax_c  = nullptr;  // abs-max of C output (for next scale)

    // Layout: row-major assumed. transa/transb = false → no transpose.
    bool transa = false;
    bool transb = false;

    // Epilogue
    GemmEpilogue epilogue = GemmEpilogue::NONE;
    const void*  d_bias   = nullptr;  // device ptr for BIAS epilogue
};

// ---- VendorGemm: lifecycle + execute ----
class VendorGemm {
public:
    VendorGemm();
    ~VendorGemm();

    // Disable copy — handle is not copyable
    VendorGemm(const VendorGemm&) = delete;
    VendorGemm& operator=(const VendorGemm&) = delete;

    // Execute GEMM: C = alpha * op(A) * op(B) + beta * C
    // dA, dB, dC: raw device pointers (dtype from desc)
    void gemm(const GemmDesc& desc,
              const void* dA, const void* dB, void* dC,
              gpu_stream_t stream = 0);

    // Batch GEMM: each entry in the array performs 1 GEMM
    // dA[i], dB[i], dC[i]: array of device pointers
    void batch_gemm(const GemmDesc& desc,
                    const void* const* dA_array,
                    const void* const* dB_array,
                    void* const*       dC_array,
                    int batch_count,
                    gpu_stream_t stream = 0);

    // Workspace: cuBLASLt needs workspace for some algorithms
    void set_workspace(void* ptr, size_t bytes);

    // Backend info
    bool uses_vendor_lib() const { return vendor_available_; }
    const char* backend_name() const;

private:
    bool vendor_available_ = false;
    void* workspace_       = nullptr;
    size_t workspace_size_ = 0;

#if defined(AMP_HAVE_CUBLAS) && defined(AMP_BACKEND_CUDA)
    cublasLtHandle_t lt_handle_ = nullptr;
    void cublas_gemm(const GemmDesc& desc,
                     const void* dA, const void* dB, void* dC,
                     gpu_stream_t stream);
#endif
#if defined(AMP_HAVE_ROCBLAS) && defined(AMP_BACKEND_HIP)
    rocblas_handle rocblas_handle_ = nullptr;
    void rocblas_gemm_impl(const GemmDesc& desc,
                           const void* dA, const void* dB, void* dC,
                           gpu_stream_t stream);
#endif
    void fallback_gemm(const GemmDesc& desc,
                       const void* dA, const void* dB, void* dC,
                       gpu_stream_t stream);
};

// ---- Process-global singleton (lazy init, thread-safe) ----
VendorGemm& global_gemm();

// ---- Convenience wrapper ----
inline void vendor_gemm(const GemmDesc& desc,
                        const void* dA, const void* dB, void* dC,
                        gpu_stream_t stream = 0) {
    global_gemm().gemm(desc, dA, dB, dC, stream);
}

} // namespace AMP
