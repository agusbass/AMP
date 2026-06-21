#!/bin/bash
# AMP — Auto-detect GPU vendor and build with the right backend/flags.
# Usage: bash scripts/build_auto.sh [extra cmake args...]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "=== AMP — Auto Backend Detection ==="

BACKEND=""
EXTRA_FLAGS=()

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    BACKEND="CUDA"
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' )"
    echo "✅ NVIDIA GPU detected (compute capability sm_${CC:-unknown})"
    [ -n "$CC" ] && EXTRA_FLAGS+=("-DCMAKE_CUDA_ARCHITECTURES=$CC")
    command -v nvcc >/dev/null 2>&1 && EXTRA_FLAGS+=("-DAMP_HAVE_CUBLAS=ON")
    if [ -n "$CC" ] && [ "$CC" -ge 89 ]; then
        EXTRA_FLAGS+=("-DAMP_HAVE_FP8=ON")
    fi
    ldconfig -p 2>/dev/null | grep -q libnccl && EXTRA_FLAGS+=("-DAMP_HAVE_NCCL=ON")

elif command -v rocm-smi >/dev/null 2>&1 || [ -d /opt/rocm ]; then
    BACKEND="HIP"
    export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    EXTRA_FLAGS+=("-DCMAKE_PREFIX_PATH=$ROCM_PATH")
    echo "✅ AMD ROCm detected (ROCM_PATH=$ROCM_PATH)"
    # Check for the actual CMake package config, not just the header dir --
    # several ROCm images (e.g. rocwmma-dev on ROCm 6.1.0) ship the headers
    # but no rocwmmaConfig.cmake, which makes find_package(rocwmma REQUIRED)
    # in CMakeLists.txt fail outright. Confirmed missing on a real MI300X
    # pod despite /opt/rocm/include/rocwmma existing.
    if ls "$ROCM_PATH"/lib/cmake/rocwmma/rocwmma*-config.cmake >/dev/null 2>&1 || \
       ls "$ROCM_PATH"/lib/cmake/rocwmma/rocwmmaConfig.cmake >/dev/null 2>&1; then
        EXTRA_FLAGS+=("-DAMP_HAVE_ROCWMMA=ON")
    fi
    ldconfig -p 2>/dev/null | grep -q librocblas && EXTRA_FLAGS+=("-DAMP_HAVE_ROCBLAS=ON")
    ldconfig -p 2>/dev/null | grep -q librccl && EXTRA_FLAGS+=("-DAMP_HAVE_RCCL=ON")
    # Check for the actual header, not just "ROCm >= 6.1" -- confirmed on
    # real hardware that hip_fp8.h can be absent even on ROCm 6.1.0.
    if [ -f "$ROCM_PATH/include/hip/hip_fp8.h" ]; then
        EXTRA_FLAGS+=("-DAMP_HAVE_FP8=ON")
    fi

elif command -v sycl-ls >/dev/null 2>&1; then
    BACKEND="SYCL"
    echo "✅ Intel oneAPI/SYCL detected"
    command -v sycl-ls >/dev/null 2>&1 && sycl-ls 2>/dev/null | grep -qi gpu && EXTRA_FLAGS+=("-DAMP_HAVE_XMX=ON")

else
    BACKEND="CPU"
    echo "⚠️  No GPU vendor detected — falling back to CPU sanity build"
fi

echo ""
echo "Backend     : $BACKEND"
echo "Extra flags : ${EXTRA_FLAGS[*]:-(none)}"
echo ""

mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
cmake "$PROJECT_DIR" -DAMP_BACKEND="$BACKEND" "${EXTRA_FLAGS[@]}" -DCMAKE_BUILD_TYPE=Release "$@"
make -j"$(nproc)"

echo ""
echo "✅ AMP build complete (backend: $BACKEND)"
echo "  ./test_triple    — Integration test"
echo "  ./bench_full     — Full benchmark suite"
