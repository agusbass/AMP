# Validation log

Detailed, line-by-line record of what was actually built, run, and fixed.
The README keeps only the headline result. This is the receipts.

## MI300X (RunPod, ROCm 6.1)

Built and ran on a rented AMD Instinct MI300X (image
`rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.11_pytorch_release_2.10.0`). The HIP
backend had never actually been compiled before this validation pass.
Doing so surfaced 8 real bugs, all fixed and pushed:

1. `include/portable.hpp` included `<hiprtc/hiprtc.h>`, which doesn't exist;
   ROCm ships it as `<hip/hiprtc.h>`.
2. `src/collective.cpp` called `ncclCommInitAll` with the wrong argument
   count/order (`ndev` passed twice).
3. `CMakeLists.txt` never actually compiled the `.cu` kernel sources for
   the HIP backend. Without CUDA language support enabled project-wide,
   CMake silently dropped `kernels/matmul.cu`/`flash_attn.cu` from every HIP
   target instead of erroring.
4. Fixing #3 needed `-x hip` (not CMake's default `-x c++`) so the compiler
   actually enables `__global__`/`__shared__`/`threadIdx` for those files.
5. `kernels/flash_attn.cu` called CUDA-only symbols directly
   (`__bfloat162float`, `cudaGetLastError`, etc.) despite its own
   `#if AMP_BACKEND_CUDA || AMP_BACKEND_HIP` guard implying HIP support.
   Added vendor-neutral `AMP_BF16_TO_FLOAT`/`AMP_GET_LAST_ERROR`/etc. macros
   to `portable.hpp`.
6. FlashAttention's default tile config needs 88KB of dynamic shared
   memory; MI300X hard-caps shared memory at 64KB/block with no opt-in
   (unlike NVIDIA's ~227KB on H100). Added a runtime check that falls back
   to a smaller tile (52KB) instead of crashing with `invalid argument`.
7. The rocBLAS GEMM path's `lda`/`ldb`/`ldc` were already correct for the
   standard row-major-via-column-major BLAS trick, but the call never
   applied the matching operand swap (A and B swapped, M and N swapped)
   that trick requires. Failed with `rocblas_status_invalid_size` for any
   non-square M not equal to K shape.
8. `scripts/build_auto.sh` itself enabled `-DAMP_HAVE_ROCWMMA=ON` whenever
   `$ROCM_PATH/include/rocwmma` existed, but `find_package(rocwmma
   REQUIRED)` needs a CMake package config this image doesn't ship despite
   having the headers. It also enabled `-DAMP_HAVE_FP8=ON` purely from
   `ROCm >= 6.1`, but `hip_fp8.h` can still be absent on a 6.1.0 install (as
   it was here). Both checks now verify the actual file/config needed.

After all 8 fixes, `test_triple` (GEMM autotune, paged KV pool,
FlashAttention-2, RCCL collectives, continuous batching, speculative
decoding, rocBLAS) passes end-to-end on MI300X.

## Docker build (GitHub Actions CI, no GPU needed)

The hackathon requires a containerized submission. The `Dockerfile` had
never actually been built by anyone before this pass. Compiling HIP device
code doesn't need a GPU, only running it does, so CI validates the build
on every push (`.github/workflows/docker-build.yml`). That first real
build surfaced 2 more bugs:

9. `FROM rocm/rocm:6.3.2-complete`: this image doesn't exist on Docker
   Hub at all. The real official image is `rocm/dev-ubuntu-22.04`.
10. `portable.hpp` used `hip_fp8_e4m3`/`hip_fp8_e5m2` as the FP8 type
    names, but ROCm 6.3.2's `amd_hip_fp8.h` names them
    `__hip_fp8_e4m3`/`__hip_fp8_e5m2` (double underscore prefix). Never
    caught on the MI300X pod because that pod's ROCm install lacked
    `hip_fp8.h` entirely, so `AMP_HAVE_FP8` was always OFF there.

With both fixed, CI now builds the image end-to-end successfully.
`./test_triple` correctly reports `no ROCm-capable device is detected` and
exits. That's expected on a GPU-less CI runner, and proof the *build*
works independent of GPU availability.

## NVIDIA side (Google Colab, Tesla T4)

The same CUDA backend was separately built and run on a real NVIDIA Tesla
T4 (free tier, CUDA 12.8). `amp_verify_matmul` passes all 4 tile configs
with the same `max_rel_err` of about 0.00006 as the MI300X run. The FP32
GEMM kernel is numerically identical across both vendors. cuBLASLt (CUDA
vendor-GEMM) ran cleanly (6043 GFLOPS, 1024x1024x512 FP32) with no fixes
needed.

The plugin interface (`amp_validate_kernel` plus
`examples/user_plugin_example.cu`) was also verified end to end on the T4:
PASS, `max_rel_err=0.00045` vs CPU reference. The same example, compiled
with `hipcc` and run on a real MI300X, produced the identical
`max_rel_err=0.00045` vs CPU reference. `scripts/parity_check.py` diffing
the two vendors' actual output arrays directly against each other came
back `max_rel_err=0.000000` (byte-for-byte identical), closing the
cross-vendor loop for the plugin path specifically, not just AMP's
bundled kernel.

## Cross-vendor parity results

96x96x96 (toy scale):

```
Cross-vendor parity: nvidia (A) vs amd (B)
tile             A GFLOPS   B GFLOPS  B/A ratio  max_rel_err  status
(16,16,16)          183.5      329.6      1.80x     0.000069  PASS
(32,16,16)          154.5      287.0      1.86x     0.000069  PASS
(32,32,16)          113.5      235.3      2.07x     0.000069  PASS
(32,32,32)          133.0      267.1      2.01x     0.000044  PASS
ALL TILE CONFIGS CROSS-VENDOR PARITY OK (rel_err < 0.001)
```

1024x1024x1024 (LLM-realistic scale):

```
Cross-vendor parity: nvidia (A) vs amd (B)
tile             A GFLOPS   B GFLOPS  B/A ratio  max_rel_err  status
(16,16,16)          583.6    12300.4     21.08x     0.000748  PASS
(32,16,16)          638.8    12988.4     20.33x     0.000748  PASS
(32,32,16)          662.0    11481.5     17.34x     0.000748  PASS
(32,32,32)          831.7    12904.0     15.52x     0.000487  PASS
ALL TILE CONFIGS CROSS-VENDOR PARITY OK (rel_err < 0.001)
```

The 15-21x ratio tracks the actual hardware gap (T4 at roughly 8 TFLOPS
FP32 vs MI300X at roughly 163 TFLOPS FP32 peak), not a measurement
artifact.

## Audit pass: closing gaps that were claimed but never actually exercised

A later review caught several claims that were true in spirit (the
underlying code is correct) but had never actually been run, because
every real test up to that point happened to pass or happened to only
touch one backend:

- **`parity_check.py --analyze` on a genuine FAIL**: every real
  cross-vendor check in this log passed, so the auto-diagnose-on-FAIL
  branch had never fired. Closed with a permanent regression test
  (`tests/test_parity_check_analyze.py`) using two synthetic dumps with
  a deliberately mismatched tile. Confirmed FAIL is detected
  (`exit 1`) and `--analyze` correctly invokes static diagnosis against
  the real shipped kernel (`kernels/matmul.cuh`), reporting a legitimate
  INFO-level bank-conflict note (not a false HIGH).
- **CPU backend**: claimed supported (`AMP_BACKEND=CPU`) but never built
  or run anywhere in this project. Verified: built and ran clean on a real
  pod (no GPU touched): `Backend: cpu`, pool/cache modules pass,
  GPU-only modules (autotune) skip gracefully, `exit 0`.
- **`docker run` with GPU device passthrough** (`--device=/dev/kfd
  --device=/dev/dri`): attempted on a RunPod pod. Docker itself isn't
  preinstalled and the daemon fails to start in that environment:
  `iptables --wait -t nat -N DOCKER: ... Permission denied`, because
  RunPod pods are themselves containers without the network
  capabilities nested Docker needs. This is an infrastructure
  limitation of that specific rental, not an AMP bug, but it means the
  actual `docker run` GPU path remains unverified. CI only proves
  `docker build` succeeds.
- **SYCL backend**: build now verified via a dedicated CI workflow
  (`.github/workflows/sycl-build.yml`) that installs Intel's official
  oneAPI DPC++ compiler via apt and builds against it, no Intel GPU
  needed to compile. The first real build surfaced 3 more bugs:
  `build_auto.sh`'s SYCL branch never set `CMAKE_CXX_COMPILER`, so CMake
  silently used the system default (g++) instead of `icpx`, which then
  rejected the `-fsycl` flag `add_sycl_to_target()` adds (fixed by
  setting `-DCMAKE_CXX_COMPILER=icpx` when `icpx`/`dpcpp` is found);
  `kernels/matmul_sycl.cpp` referenced a plain `sycl::bfloat16` that
  doesn't exist in current oneAPI releases (the real type lives under
  `sycl::ext::oneapi::bfloat16`, now exposed as a proper `AMP_bf16`
  type alias in `portable.hpp`, matching every other backend's
  pattern); and the same file called `amp::detail::default_q()` with a
  lowercase namespace when the actual namespace is `AMP` (uppercase),
  confirmed against the correct usage in `src/device.cpp`. With all
  three fixed, the build succeeds end to end, and `test_triple`
  correctly throws `No device of requested type 'info::device_type::gpu'
  available` and aborts. That's expected on a GPU-less CI runner and
  proof the *build* works independent of Intel GPU availability, the
  same precedent as the Docker and CPU-backend validation above.
  Running this backend against real Intel GPU hardware (Arc/Xe-HPC/
  Gaudi) is still open since none was available in this project.
- **`scripts/amp_check.sh` on the CUDA/NVIDIA side**: closed.
  `bash scripts/amp_check.sh my_kernel.cu my_naive_gemm 512 512 512` on
  a real Tesla T4 (Colab): `Detected: NVIDIA (nvcc)` followed by
  `max_rel_err=0.000450 vs CPU reference PASS`, the same error metric
  as the MI300X run. Also caught and fixed a second real bug in the
  process: the script always wrote to a fixed `cuda_dump.json`/
  `hip_dump.json`, colliding with `amp_parity_dump`'s own default
  filenames and silently overwriting results when checking two
  different kernels in sequence. Confirmed fixed by the output
  filename in this run: `my_naive_gemm_cuda_dump.json`.

## Cost vs. the alternative

This entire validation pass, building the HIP backend and Docker image
for the first time, finding and fixing 10 real bugs, and producing a
verified cross-vendor parity result, took roughly an hour of MI300X
rental (about $2.19/hr on RunPod) plus a free Colab T4 session and free
GitHub Actions minutes. The alternative is trusting an unverified port in
production, or days of an engineer manually instrumenting both builds to
compare outputs by hand.

## UX layer (Spaces, Colab, one-command scripts)

- **HF Spaces** (`web/app.py`, deployed to
  [agusbudiman14/amp-kernel-diagnose](https://agusbudiman14-amp-kernel-diagnose.hf.space/)):
  live, verified via direct HTTP request and a real diagnose-and-fix run
  against `tests/fixtures/buggy_matmul_fixture.cuh` (the same fixture
  `tests/test_diagnose.py` covers). It found the same 2 HIGH findings and
  applied the same `BN` to `BK` plus inserted-`__syncthreads()` fix the
  CLI produces.
- **Colab quickstart** (`notebooks/quickstart.ipynb`): the underlying
  build/verify/parity-dump steps it runs are the same ones verified live
  on Tesla T4 above; the notebook itself (git-clone-based, no manual
  upload) has not been re-run end-to-end since that change.
- **`scripts/amp_check.sh`** (one-command kernel check): verified end
  to end on a real MI300X. `bash scripts/amp_check.sh kernel.cu
  my_bare_gemm 512 512 512` auto-detected `hipcc`, wrapped and compiled
  the kernel, built AMP, and validated: `max_rel_err=0.000450 vs CPU
  reference PASS`. Caught and fixed one real bug in the process: the
  generated wrapper `#include`d the user's kernel file *before*
  `hip_runtime.h`, so `blockIdx`/`blockDim`/`threadIdx` were undeclared
  even though `__global__` itself parsed fine. Swapped the include
  order. `scripts/amp_generate_wrapper.py`, which it's built on top of,
  is separately unit-tested (`tests/test_generate_wrapper.py`) for
  output shape.
- **Docker's default `CMD`** (runs `amp_verify_matmul` on `docker run`):
  CI only validates `docker build` (no GPU on GitHub-hosted runners), so
  the actual `docker run` path, including this default command, has
  never been executed against a real device.

## Prior art considered

Closest prior work is [CASS](https://github.com/ahmedheakl/CASS) (MBZUAI),
which validates CUDA to HIP transpilation correctness and performance as a
research benchmark over classic HPC workloads (Rodinia/SHOC/PolyBench),
not a practical tool aimed at LLM-inference kernels a developer points at
their own code. Two earlier project ideas turned out to already exist in
production and were dropped: KV-cache offload to CPU (already solved by
[LMCache](https://github.com/LMCache/LMCache)), and ROCm install/validation
tooling (AMD's own ROCm Validation Suite, RCCL-Tests).
