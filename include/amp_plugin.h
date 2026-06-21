/* amp_plugin.h - the contract for validating YOUR OWN kernel with AMP.
 *
 * AMP's bundled kernels (kernels/matmul.cu, flash_attn.cu) are a reference
 * implementation, not the point of this tool. The point is letting you
 * point amp_validate_kernel at YOUR kernel -- the one you already have,
 * possibly already HIPIFY'd -- without rewriting it to fit AMP's internal
 * structure.
 *
 * To validate your own GEMM kernel (CUDA or HIP, AMP doesn't care which):
 *
 *   1. Write one wrapper function matching the signature below. It can
 *      call your existing kernel launch exactly as it already does --
 *      this is a thin adapter, not a rewrite.
 *   2. Compile it as a shared library:
 *        nvcc -shared -Xcompiler -fPIC my_wrapper.cu -o libmykernel_cuda.so
 *        hipcc -shared -fPIC my_wrapper.cu -o libmykernel_hip.so
 *   3. ./amp_validate_kernel libmykernel_cuda.so cuda_dump.json 1024 1024 1024
 *      ./amp_validate_kernel libmykernel_hip.so  hip_dump.json  1024 1024 1024
 *   4. python3 scripts/parity_check.py cuda_dump.json hip_dump.json
 *
 * amp_validate_kernel itself links against neither CUDA nor HIP -- it's a
 * plain host C++ binary that dlopen()s your library, so the exact same
 * tool validates either side without needing both toolkits installed.
 * See examples/user_plugin_example.cu for a complete, runnable wrapper.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* A, B, C are HOST (CPU) memory, row-major: A is M*K, B is K*N, C is M*N
 * (C[i*N+j] = sum_k A[i*K+k]*B[k*N+j]). Your implementation owns all
 * device-side work: allocate device buffers, copy A/B in, launch your
 * kernel, copy C back, synchronize, and free your device buffers --
 * all of that must be complete before this function returns, since the
 * caller times wall-clock around the call for the GFLOPS figure in the
 * dump.
 *
 * Return 0 on success, nonzero on failure (caller treats this as a hard
 * error, not a numerical mismatch -- numerical correctness is checked
 * separately by amp_validate_kernel against a CPU reference).
 */
typedef int (*amp_user_gemm_fp32_fn)(const float* A, const float* B, float* C,
                                      int M, int N, int K);

/* The symbol amp_validate_kernel looks up via dlsym() must have exactly
 * this name and this C linkage. */
#define AMP_PLUGIN_GEMM_FP32_SYMBOL "amp_user_gemm_fp32"

#ifdef __cplusplus
}
#endif
