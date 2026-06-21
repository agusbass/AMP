// gemm_vendor.cpp - cuBLASLt / rocBLAS / fallback GEMM backend
#include "gemm_vendor.hpp"
#include "profiler.hpp"
#include <stdexcept>
#include <mutex>
#include <memory>
#include <cstring>

#if defined(AMP_BACKEND_CUDA) && !defined(AMP_HAVE_CUBLAS)
  // cuBLASLt is available alongside the CUDA toolkit — no separate flag needed
  // but only active if -DAMP_HAVE_CUBLAS is set, to keep it opt-in
#endif

namespace AMP {

// ================================================================
// dtype helpers
// ================================================================

#if defined(AMP_HAVE_CUBLAS) && defined(AMP_BACKEND_CUDA)

static cudaDataType_t to_cuda_dtype(DataType dt) {
    switch (dt) {
        case DataType::FP32:    return CUDA_R_32F;
        case DataType::BF16:    return CUDA_R_16BF;
        case DataType::FP16:    return CUDA_R_16F;
        case DataType::INT8:    return CUDA_R_8I;
        case DataType::INT32:   return CUDA_R_32I;
#if defined(AMP_HAVE_FP8)
        case DataType::FP8E4M3: return CUDA_R_8F_E4M3;
        case DataType::FP8E5M2: return CUDA_R_8F_E5M2;
#endif
        default: throw std::runtime_error("unsupported dtype for cuBLASLt");
    }
}

static cublasComputeType_t compute_type_for(DataType a, DataType c) {
    if (c == DataType::FP32) return CUBLAS_COMPUTE_32F;
    if (a == DataType::FP16 && c == DataType::FP16) return CUBLAS_COMPUTE_16F;
    if (a == DataType::INT8)  return CUBLAS_COMPUTE_32I;
    return CUBLAS_COMPUTE_32F;
}

static cublasLtEpilogue_t to_cublas_epilogue(GemmEpilogue ep) {
    switch (ep) {
        case GemmEpilogue::RELU: return CUBLASLT_EPILOGUE_RELU;
        case GemmEpilogue::BIAS: return CUBLASLT_EPILOGUE_BIAS;
        case GemmEpilogue::GELU: return CUBLASLT_EPILOGUE_GELU;
        default:                 return CUBLASLT_EPILOGUE_DEFAULT;
    }
}

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) \
        throw std::runtime_error("cuBLASLt error at " __FILE__ ":" + \
                                  std::to_string(__LINE__)); \
} while(0)

#endif  // AMP_HAVE_CUBLAS

#if defined(AMP_HAVE_ROCBLAS) && defined(AMP_BACKEND_HIP)

static rocblas_datatype to_rocblas_dtype(DataType dt) {
    switch (dt) {
        case DataType::FP32: return rocblas_datatype_f32_r;
        case DataType::BF16: return rocblas_datatype_bf16_r;
        case DataType::FP16: return rocblas_datatype_f16_r;
        case DataType::INT8: return rocblas_datatype_i8_r;
        case DataType::INT32:return rocblas_datatype_i32_r;
        default: throw std::runtime_error("unsupported dtype for rocBLAS");
    }
}

#define ROCBLAS_CHECK(call) do { \
    rocblas_status _s = (call); \
    if (_s != rocblas_status_success) \
        throw std::runtime_error("rocBLAS error: " + std::to_string(_s)); \
} while(0)

#endif  // AMP_HAVE_ROCBLAS

// ================================================================
// VendorGemm implementation
// ================================================================

VendorGemm::VendorGemm() {
#if defined(AMP_HAVE_CUBLAS) && defined(AMP_BACKEND_CUDA)
    cublasStatus_t s = cublasLtCreate(&lt_handle_);
    vendor_available_ = (s == CUBLAS_STATUS_SUCCESS);
    if (vendor_available_) {
        // cuBLASLt needs a workspace for FP8 and BF16 large tiles
        workspace_size_ = 32ULL * 1024 * 1024;  // 32MB
        if (cudaMalloc(&workspace_, workspace_size_) != cudaSuccess) {
            workspace_ = nullptr;
            workspace_size_ = 0;
        }
    }
#elif defined(AMP_HAVE_ROCBLAS) && defined(AMP_BACKEND_HIP)
    rocblas_status s = rocblas_create_handle(&rocblas_handle_);
    vendor_available_ = (s == rocblas_status_success);
#else
    vendor_available_ = false;
#endif
}

VendorGemm::~VendorGemm() {
#if defined(AMP_HAVE_CUBLAS) && defined(AMP_BACKEND_CUDA)
    if (workspace_) cudaFree(workspace_);
    if (lt_handle_) cublasLtDestroy(lt_handle_);
#elif defined(AMP_HAVE_ROCBLAS) && defined(AMP_BACKEND_HIP)
    if (rocblas_handle_) rocblas_destroy_handle(rocblas_handle_);
#endif
}

void VendorGemm::set_workspace(void* ptr, size_t bytes) {
    workspace_ = ptr; workspace_size_ = bytes;
}

const char* VendorGemm::backend_name() const {
#if defined(AMP_HAVE_CUBLAS) && defined(AMP_BACKEND_CUDA)
    return vendor_available_ ? "cublasLt" : "AMP-fallback";
#elif defined(AMP_HAVE_ROCBLAS) && defined(AMP_BACKEND_HIP)
    return vendor_available_ ? "rocBLAS" : "AMP-fallback";
#else
    return "AMP-fallback";
#endif
}

void VendorGemm::gemm(const GemmDesc& desc,
                       const void* dA, const void* dB, void* dC,
                       gpu_stream_t stream)
{
    AMP_RANGE_PUSH("amp::vendor_gemm");
    if (vendor_available_) {
#if defined(AMP_HAVE_CUBLAS) && defined(AMP_BACKEND_CUDA)
        cublas_gemm(desc, dA, dB, dC, stream);
#elif defined(AMP_HAVE_ROCBLAS) && defined(AMP_BACKEND_HIP)
        rocblas_gemm_impl(desc, dA, dB, dC, stream);
#else
        fallback_gemm(desc, dA, dB, dC, stream);
#endif
    } else {
        fallback_gemm(desc, dA, dB, dC, stream);
    }
    AMP_RANGE_POP();
}

// ================================================================
// cuBLASLt backend
// ================================================================
#if defined(AMP_HAVE_CUBLAS) && defined(AMP_BACKEND_CUDA)

void VendorGemm::cublas_gemm(const GemmDesc& d,
                               const void* dA, const void* dB, void* dC,
                               gpu_stream_t stream)
{
    cublasLtMatmulDesc_t   op_desc  = nullptr;
    cublasLtMatrixLayout_t lay_A    = nullptr;
    cublasLtMatrixLayout_t lay_B    = nullptr;
    cublasLtMatrixLayout_t lay_C    = nullptr;

    auto dtype_a = to_cuda_dtype(d.dtype_a);
    auto dtype_b = to_cuda_dtype(d.dtype_b);
    auto dtype_c = to_cuda_dtype(d.dtype_c);
    auto ctype   = compute_type_for(d.dtype_a, d.dtype_c);

    // Operation descriptor
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, ctype, CUDA_R_32F));

    // Transpose ops
    cublasOperation_t op_a = d.transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = d.transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));

    // FP8 per-tensor scaling
    if (d.d_scale_a) CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d.d_scale_a, sizeof(d.d_scale_a)));
    if (d.d_scale_b) CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d.d_scale_b, sizeof(d.d_scale_b)));
    if (d.d_scale_c) CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d.d_scale_c, sizeof(d.d_scale_c)));
    if (d.d_amax_c)  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d.d_amax_c, sizeof(d.d_amax_c)));

    // Epilogue
    auto epilogue = to_cublas_epilogue(d.epilogue);
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    if (d.epilogue == GemmEpilogue::BIAS && d.d_bias) {
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc,
            CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d.d_bias, sizeof(d.d_bias)));
    }

    // Matrix layouts — cuBLASLt default is col-major, we explicitly set ROW-MAJOR
    int64_t lda = d.transa ? d.M : d.K;
    int64_t ldb = d.transb ? d.K : d.N;
    int64_t ldc = d.N;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lay_A, dtype_a, d.transa?d.K:d.M, d.transa?d.M:d.K, lda));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lay_B, dtype_b, d.transb?d.N:d.K, d.transb?d.K:d.N, ldb));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lay_C, dtype_c, d.M, d.N, ldc));
    // Set ROW-MAJOR order (C array layout)
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(lay_A, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(lay_B, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(lay_C, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));

    // Algorithm heuristic (let cuBLASLt pick best)
    cublasLtMatmulHeuristicResult_t heurResult;
    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    if (workspace_) {
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size_, sizeof(workspace_size_)));
    }

    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(lt_handle_, op_desc,
        lay_A, lay_B, lay_C, lay_C, pref, 1,
        &heurResult, &returnedResults);

    CUBLAS_CHECK(cublasLtMatmul(lt_handle_, op_desc,
        &d.alpha, dA, lay_A, dB, lay_B,
        &d.beta,  dC, lay_C, dC, lay_C,
        returnedResults > 0 ? &heurResult.algo : nullptr,
        workspace_, workspace_size_, stream));

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lay_A);
    cublasLtMatrixLayoutDestroy(lay_B);
    cublasLtMatrixLayoutDestroy(lay_C);
    cublasLtMatmulDescDestroy(op_desc);
}

#endif  // AMP_HAVE_CUBLAS

// ================================================================
// rocBLAS backend
// ================================================================
#if defined(AMP_HAVE_ROCBLAS) && defined(AMP_BACKEND_HIP)

void VendorGemm::rocblas_gemm_impl(const GemmDesc& d,
                                    const void* dA, const void* dB, void* dC,
                                    gpu_stream_t stream)
{
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_handle_, stream));

    rocblas_datatype dt_a  = to_rocblas_dtype(d.dtype_a);
    rocblas_datatype dt_b  = to_rocblas_dtype(d.dtype_b);
    rocblas_datatype dt_c  = to_rocblas_dtype(d.dtype_c);
    rocblas_datatype dt_co = to_rocblas_dtype(DataType::FP32); // compute type

    rocblas_operation op_a = d.transa ? rocblas_operation_transpose : rocblas_operation_none;
    rocblas_operation op_b = d.transb ? rocblas_operation_transpose : rocblas_operation_none;

    int lda = d.transa ? d.M : d.K;
    int ldb = d.transb ? d.K : d.N;
    int ldc = d.N;

    float alpha = d.alpha, beta = d.beta;

    // lda/ldb/ldc above are the leading dimensions of A/B/C as stored
    // row-major. rocblas_gemm_ex (like cuBLAS) is column-major-only, so we
    // apply the standard row-major-via-column-major trick: C_rm = A_rm*B_rm
    // is computed as C_rm^T = B_rm^T * A_rm^T, i.e. swap A<->B and M<->N
    // (the row-major buffers are unchanged; only the call's operand order
    // and dimensions swap). Without this swap, rocBLAS rejects the call
    // with rocblas_status_invalid_size whenever M != K.
    ROCBLAS_CHECK(rocblas_gemm_ex(
        rocblas_handle_,
        op_b, op_a,
        d.N, d.M, d.K,
        &alpha,
        dB, dt_b, ldb,
        dA, dt_a, lda,
        &beta,
        dC, dt_c, ldc,
        dC, dt_c, ldc,
        dt_co,
        rocblas_gemm_algo_standard, 0, 0));
}

#endif  // AMP_HAVE_ROCBLAS

// ================================================================
// Fallback: delegate to the hand-rolled kernel in matmul.cu
// ================================================================
void VendorGemm::fallback_gemm(const GemmDesc& d,
                                 const void* dA, const void* dB, void* dC,
                                 gpu_stream_t /*stream*/)
{
    // CPU-only naive fallback (dev/sanity, not for production)
    if (d.dtype_a != DataType::FP32) return;
    auto* A = reinterpret_cast<const float*>(dA);
    auto* B = reinterpret_cast<const float*>(dB);
    auto* C = reinterpret_cast<float*>(dC);
    for (int i = 0; i < d.M; ++i)
        for (int j = 0; j < d.N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < d.K; ++k) s += A[i*d.K+k] * B[k*d.N+j];
            C[i*d.N+j] = d.alpha * s + d.beta * C[i*d.N+j];
        }
}

void VendorGemm::batch_gemm(const GemmDesc& desc,
                              const void* const* dA_array,
                              const void* const* dB_array,
                              void* const*       dC_array,
                              int batch_count,
                              gpu_stream_t stream)
{
    // Simple loop — cuBLASLt batched matmul could be further optimized
    for (int b = 0; b < batch_count; ++b)
        gemm(desc, dA_array[b], dB_array[b], dC_array[b], stream);
}

// ================================================================
// Global singleton
// ================================================================
VendorGemm& global_gemm() {
    static std::once_flag flag;
    static std::unique_ptr<VendorGemm> instance;
    std::call_once(flag, [] { instance = std::make_unique<VendorGemm>(); });
    return *instance;
}

} // namespace AMP
