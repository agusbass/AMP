// user_plugin_example.cu - a complete, runnable example of wrapping YOUR
// OWN GEMM kernel so amp_validate_kernel can validate it.
//
// This file compiles unmodified with either nvcc (CUDA) or hipcc (HIP) --
// the kernel body below is intentionally naive (not tuned, no shared
// memory) since the point is the wrapper shape, not kernel performance.
// Replace `my_naive_gemm_kernel` with a call into your actual kernel and
// everything else stays the same.
//
// Build:
//   nvcc  -shared -Xcompiler -fPIC user_plugin_example.cu -o libexample_cuda.so
//   hipcc -shared -fPIC          user_plugin_example.cu -o libexample_hip.so
//
// Validate:
//   ./amp_validate_kernel libexample_cuda.so cuda_dump.json 512 512 512
//   ./amp_validate_kernel libexample_hip.so  hip_dump.json  512 512 512
//   python3 scripts/parity_check.py cuda_dump.json hip_dump.json

#if defined(__HIPCC__)
  #include <hip/hip_runtime.h>
  #define GPU_MALLOC      hipMalloc
  #define GPU_FREE        hipFree
  #define GPU_MEMCPY_H2D(d,s,n) hipMemcpy(d,s,n,hipMemcpyHostToDevice)
  #define GPU_MEMCPY_D2H(d,s,n) hipMemcpy(d,s,n,hipMemcpyDeviceToHost)
  #define GPU_SYNC        hipDeviceSynchronize
  #define GPU_SUCCESS     hipSuccess
#else
  #include <cuda_runtime.h>
  #define GPU_MALLOC      cudaMalloc
  #define GPU_FREE        cudaFree
  #define GPU_MEMCPY_H2D(d,s,n) cudaMemcpy(d,s,n,cudaMemcpyHostToDevice)
  #define GPU_MEMCPY_D2H(d,s,n) cudaMemcpy(d,s,n,cudaMemcpyDeviceToHost)
  #define GPU_SYNC        cudaDeviceSynchronize
  #define GPU_SUCCESS     cudaSuccess
#endif

// --- Replace this kernel with YOUR existing kernel ---
__global__ void my_naive_gemm_kernel(const float* A, const float* B, float* C,
                                      int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// --- This is the part amp_validate_kernel actually calls (extern "C", the
//     exact signature from include/amp_plugin.h) ---
extern "C" int amp_user_gemm_fp32(const float* hA, const float* hB, float* hC,
                                   int M, int N, int K) {
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    if (GPU_MALLOC(&dA, (size_t)M * K * sizeof(float)) != GPU_SUCCESS) return 1;
    if (GPU_MALLOC(&dB, (size_t)K * N * sizeof(float)) != GPU_SUCCESS) return 1;
    if (GPU_MALLOC(&dC, (size_t)M * N * sizeof(float)) != GPU_SUCCESS) return 1;

    GPU_MEMCPY_H2D(dA, hA, (size_t)M * K * sizeof(float));
    GPU_MEMCPY_H2D(dB, hB, (size_t)K * N * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    my_naive_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);

    if (GPU_SYNC() != GPU_SUCCESS) { GPU_FREE(dA); GPU_FREE(dB); GPU_FREE(dC); return 1; }

    GPU_MEMCPY_D2H(hC, dC, (size_t)M * N * sizeof(float));

    GPU_FREE(dA); GPU_FREE(dB); GPU_FREE(dC);
    return 0;
}
