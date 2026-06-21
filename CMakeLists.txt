cmake_minimum_required(VERSION 3.20)
project(amp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Backend selection ----
set(AMP_BACKEND "CPU" CACHE STRING "GPU backend: CUDA | HIP | SYCL | CPU")
set_property(CACHE AMP_BACKEND PROPERTY STRINGS CUDA HIP SYCL CPU)

# ---- Optional features ----
option(AMP_ENABLE_PROFILER  "NVTX/ROCTx/ITT profiler annotations"       OFF)
option(AMP_HAVE_NUMA        "NUMA-aware pool (Linux + libnuma)"          OFF)
option(AMP_HAVE_ROCWMMA     "BF16 WMMA via rocWMMA (HIP + ROCm 6+)"     OFF)
option(AMP_HAVE_FP8         "FP8 E4M3/E5M2 kernels (CUDA 12.1+ / ROCm 6.1+)" OFF)
option(AMP_HAVE_XMX         "Intel XMX joint_matrix GEMM (oneAPI 2024+)" OFF)
option(AMP_HAVE_NCCL        "NCCL collective communication (CUDA)"       OFF)
option(AMP_HAVE_RCCL        "RCCL collective communication (HIP/AMD)"    OFF)
option(AMP_HAVE_ONECCL      "oneCCL collective communication (SYCL)"     OFF)
option(AMP_HAVE_CUBLAS      "cuBLASLt vendor GEMM backend (CUDA)"        OFF)
option(AMP_HAVE_ROCBLAS     "rocBLAS vendor GEMM backend (HIP)"          OFF)

include_directories(include)

set(COMMON_SRC
    src/device.cpp
    src/pool.cpp
    src/pool_fast.cpp
    src/pool_numa.cpp
    src/cache.cpp
    src/cache_fast.cpp
    src/cache_async.cpp
    src/collective.cpp
    src/gemm_vendor.cpp
    src/scheduler.cpp
    src/speculative.cpp
)

# ==========================================================================
# Host-only regression tests (no GPU required, run on every backend).
# Declared here, BEFORE any AMP_BACKEND_*/AMP_HAVE_* compile definition is
# added below — otherwise portable.hpp's CUDA/HIP/SYCL branches would try to
# include cuda_runtime.h/hip_runtime.h on a target that never needs the GPU.
# ==========================================================================
find_package(Threads REQUIRED)
add_executable(amp_verify_fixes tests/verify_fixes.cpp)
add_executable(amp_verify_cache_async tests/verify_cache_async.cpp
    src/cache.cpp src/cache_async.cpp)
target_link_libraries(amp_verify_cache_async PRIVATE Threads::Threads)

# amp_validate_kernel: validates a USER-supplied GEMM kernel (their own
# CUDA or HIP code, dlopen'd as a shared library) against a CPU reference
# and dumps the same JSON schema amp_parity_dump uses -- so a user can run
# this once on a CUDA machine and once on a HIP machine against their own
# kernel, then diff with scripts/parity_check.py, without needing AMP's
# own CUDA/HIP build at all. See include/amp_plugin.h.
#
# Uses dlfcn.h, which is POSIX-only -- every real deployment target for
# this (RunPod, Colab, AMD Developer Cloud) is Linux, so this is gated on
# UNIX rather than ported to LoadLibrary/GetProcAddress for Windows.
if(UNIX)
    add_executable(amp_validate_kernel tests/amp_validate_kernel.cpp)
    target_link_libraries(amp_validate_kernel PRIVATE ${CMAKE_DL_LIBS})
endif()

# ---- Compile definitions ----
if(AMP_ENABLE_PROFILER)
    add_compile_definitions(AMP_ENABLE_PROFILER)
endif()
if(AMP_HAVE_NUMA)
    add_compile_definitions(AMP_HAVE_NUMA)
endif()
if(AMP_HAVE_FP8)
    add_compile_definitions(AMP_HAVE_FP8)
endif()
if(AMP_HAVE_XMX)
    add_compile_definitions(AMP_HAVE_XMX)
endif()
if(AMP_HAVE_NCCL)
    add_compile_definitions(AMP_HAVE_NCCL)
endif()
if(AMP_HAVE_RCCL)
    add_compile_definitions(AMP_HAVE_RCCL)
endif()
if(AMP_HAVE_ONECCL)
    add_compile_definitions(AMP_HAVE_ONECCL)
endif()
if(AMP_HAVE_CUBLAS)
    add_compile_definitions(AMP_HAVE_CUBLAS)
endif()
if(AMP_HAVE_ROCBLAS)
    add_compile_definitions(AMP_HAVE_ROCBLAS)
endif()
if(AMP_HAVE_NUMA)
    link_libraries(numa)
endif()

# ==========================================================================
# CUDA backend
# ==========================================================================
if(AMP_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    add_compile_definitions(AMP_BACKEND_CUDA)

    # FP8 requires CUDA 12.1+
    if(AMP_HAVE_FP8)
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.1")
            message(WARNING "AMP_HAVE_FP8 requires CUDA 12.1+, disabling")
            remove_definitions(-DAMP_HAVE_FP8)
        endif()
    endif()

    # Compile common + kernel files as CUDA
    set_source_files_properties(${COMMON_SRC} PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(kernels/matmul.cu  PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(kernels/flash_attn.cu PROPERTIES LANGUAGE CUDA)

    # test_triple
    set_source_files_properties(tests/test_triple.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(test_triple
        tests/test_triple.cpp
        kernels/matmul.cu
        kernels/flash_attn.cu
        ${COMMON_SRC})
    target_link_libraries(test_triple PRIVATE nvrtc cuda)

    # bench
    set_source_files_properties(tests/bench.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(bench tests/bench.cpp ${COMMON_SRC})
    target_link_libraries(bench PRIVATE cuda)

    # bench_full
    set_source_files_properties(tests/bench_full.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(bench_full
        tests/bench_full.cpp
        kernels/matmul.cu
        ${COMMON_SRC})
    target_link_libraries(bench_full PRIVATE nvrtc cuda)

    # amp_verify_matmul: numerical correctness check (not just speed) for the
    # FP32 tile shared-memory fix, across all 4 tile configs.
    set_source_files_properties(tests/verify_matmul_correctness.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(amp_verify_matmul tests/verify_matmul_correctness.cpp kernels/matmul.cu src/gemm_vendor.cpp)
    target_link_libraries(amp_verify_matmul PRIVATE nvrtc cuda)

    # amp_parity_dump: dumps GEMM output + GFLOPS to JSON for cross-vendor
    # diffing (see tests/parity_dump.cpp + scripts/parity_check.py).
    set_source_files_properties(tests/parity_dump.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(amp_parity_dump tests/parity_dump.cpp kernels/matmul.cu src/gemm_vendor.cpp)
    target_link_libraries(amp_parity_dump PRIVATE nvrtc cuda)

    # Profiler (NVTX3 header-only in CUDA 12+, or legacy nvToolsExt)
    if(AMP_ENABLE_PROFILER)
        find_library(NVTX_LIB nvToolsExt HINTS ${CUDAToolkit_LIBRARY_DIR})
        if(NVTX_LIB)
            foreach(t test_triple bench bench_full amp_verify_matmul amp_parity_dump)
                target_link_libraries(${t} PRIVATE ${NVTX_LIB})
            endforeach()
        endif()
    endif()

    # NCCL
    if(AMP_HAVE_NCCL)
        find_library(NCCL_LIB nccl)
        if(NCCL_LIB)
            foreach(t test_triple bench bench_full amp_verify_matmul amp_parity_dump)
                target_link_libraries(${t} PRIVATE ${NCCL_LIB})
            endforeach()
        else()
            message(WARNING "libnccl not found — collective stubs will be used")
        endif()
    endif()

    # cuBLASLt (included in CUDA toolkit, just needs linking)
    if(AMP_HAVE_CUBLAS)
        find_library(CUBLASLT_LIB cublasLt HINTS ${CUDAToolkit_LIBRARY_DIR})
        if(CUBLASLT_LIB)
            foreach(t test_triple bench bench_full amp_verify_matmul amp_parity_dump)
                target_link_libraries(${t} PRIVATE ${CUBLASLT_LIB})
            endforeach()
            message(STATUS "cuBLASLt found: ${CUBLASLT_LIB}")
        else()
            # Try via CUDAToolkit target
            find_package(CUDAToolkit QUIET)
            if(CUDAToolkit_FOUND)
                foreach(t test_triple bench bench_full amp_verify_matmul amp_parity_dump)
                    target_link_libraries(${t} PRIVATE CUDA::cublasLt)
                endforeach()
            else()
                message(WARNING "cublasLt not found — vendor GEMM fallback")
            endif()
        endif()
    endif()

# ==========================================================================
# HIP (AMD ROCm) backend
# ==========================================================================
elseif(AMP_BACKEND STREQUAL "HIP")
    set(CMAKE_CXX_COMPILER "hipcc")
    add_compile_definitions(AMP_BACKEND_HIP)

    if(AMP_HAVE_ROCWMMA)
        add_compile_definitions(AMP_HAVE_ROCWMMA)
        find_package(rocwmma REQUIRED)
    endif()

    # Without CUDA language support enabled, CMake won't compile .cu sources
    # under a CXX-only project at all. LANGUAGE CXX makes it try, but that
    # forces "-x c++" which drops HIP's device-side language extensions
    # (__global__/__shared__/threadIdx); override back to "-x hip" so hipcc
    # actually compiles these as HIP/GPU code.
    set_source_files_properties(kernels/matmul.cu kernels/flash_attn.cu
        PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-x;hip")

    add_executable(test_triple
        tests/test_triple.cpp
        kernels/matmul.cu
        kernels/flash_attn.cu
        ${COMMON_SRC})
    target_link_libraries(test_triple PRIVATE hiprtc)

    add_executable(bench      tests/bench.cpp ${COMMON_SRC})
    add_executable(bench_full tests/bench_full.cpp kernels/matmul.cu ${COMMON_SRC})
    target_link_libraries(bench_full PRIVATE hiprtc)

    # amp_verify_matmul: numerical correctness check (not just speed) for the
    # FP32 tile shared-memory fix, across all 4 tile configs.
    add_executable(amp_verify_matmul tests/verify_matmul_correctness.cpp kernels/matmul.cu src/gemm_vendor.cpp)
    target_link_libraries(amp_verify_matmul PRIVATE hiprtc)

    # amp_parity_dump: dumps GEMM output + GFLOPS to JSON for cross-vendor
    # diffing (see tests/parity_dump.cpp + scripts/parity_check.py).
    add_executable(amp_parity_dump tests/parity_dump.cpp kernels/matmul.cu src/gemm_vendor.cpp)
    target_link_libraries(amp_parity_dump PRIVATE hiprtc)

    if(AMP_HAVE_ROCWMMA)
        foreach(t test_triple bench_full)
            target_link_libraries(${t} PRIVATE roc::rocwmma)
        endforeach()
    endif()

    if(AMP_ENABLE_PROFILER)
        find_library(ROCTX_LIB roctx64)
        if(ROCTX_LIB)
            foreach(t test_triple bench bench_full amp_verify_matmul amp_parity_dump)
                target_link_libraries(${t} PRIVATE ${ROCTX_LIB})
            endforeach()
        endif()
    endif()

    # rocBLAS
    if(AMP_HAVE_ROCBLAS)
        find_library(ROCBLAS_LIB rocblas)
        if(ROCBLAS_LIB)
            foreach(t test_triple bench bench_full amp_verify_matmul amp_parity_dump)
                target_link_libraries(${t} PRIVATE ${ROCBLAS_LIB})
            endforeach()
        else()
            message(WARNING "librocblas not found — vendor GEMM fallback")
        endif()
    endif()

    # RCCL
    if(AMP_HAVE_RCCL)
        find_library(RCCL_LIB rccl)
        if(RCCL_LIB)
            foreach(t test_triple bench bench_full amp_verify_matmul amp_parity_dump)
                target_link_libraries(${t} PRIVATE ${RCCL_LIB})
            endforeach()
        else()
            message(WARNING "librccl not found — collective stubs will be used")
        endif()
    endif()

# ==========================================================================
# SYCL (Intel oneAPI / Gaudi) backend
# ==========================================================================
elseif(AMP_BACKEND STREQUAL "SYCL")
    # Prerequisite: source /opt/intel/oneapi/setvars.sh before running cmake
    find_package(IntelSYCL REQUIRED)
    add_compile_definitions(AMP_BACKEND_SYCL)

    # SYCL GEMM source
    set(SYCL_EXTRA_SRC kernels/matmul_sycl.cpp)

    add_executable(test_triple tests/test_triple.cpp ${COMMON_SRC} ${SYCL_EXTRA_SRC})
    add_executable(bench       tests/bench.cpp ${COMMON_SRC})
    add_executable(bench_full  tests/bench_full.cpp ${COMMON_SRC} ${SYCL_EXTRA_SRC})

    foreach(t test_triple bench bench_full)
        add_sycl_to_target(TARGET ${t} SOURCES)
    endforeach()

    if(AMP_ENABLE_PROFILER)
        find_library(ITT_LIB ittnotify)
        if(ITT_LIB)
            foreach(t test_triple bench bench_full)
                target_link_libraries(${t} PRIVATE ${ITT_LIB})
            endforeach()
        endif()
    endif()

    # oneCCL
    if(AMP_HAVE_ONECCL)
        find_package(oneCCL REQUIRED)
        foreach(t test_triple bench bench_full)
            target_link_libraries(${t} PRIVATE oneCCL::oneCCL)
        endforeach()
    endif()

# ==========================================================================
# CPU fallback (no GPU, sanity test)
# ==========================================================================
else()
    add_compile_definitions(AMP_BACKEND_CPU)
    add_executable(test_triple tests/test_triple.cpp ${COMMON_SRC})
    add_executable(bench       tests/bench.cpp ${COMMON_SRC})
    add_executable(bench_full  tests/bench_full.cpp ${COMMON_SRC})
endif()

# ==========================================================================
# Status summary
# ==========================================================================
message(STATUS "AMP backend      : ${AMP_BACKEND}")
message(STATUS "Profiler         : ${AMP_ENABLE_PROFILER}")
message(STATUS "NUMA             : ${AMP_HAVE_NUMA}")
message(STATUS "rocWMMA (BF16)   : ${AMP_HAVE_ROCWMMA}")
message(STATUS "FP8 kernels      : ${AMP_HAVE_FP8}")
message(STATUS "Intel XMX        : ${AMP_HAVE_XMX}")
message(STATUS "NCCL             : ${AMP_HAVE_NCCL}")
message(STATUS "RCCL             : ${AMP_HAVE_RCCL}")
message(STATUS "oneCCL           : ${AMP_HAVE_ONECCL}")
message(STATUS "cuBLASLt         : ${AMP_HAVE_CUBLAS}")
message(STATUS "rocBLAS          : ${AMP_HAVE_ROCBLAS}")
