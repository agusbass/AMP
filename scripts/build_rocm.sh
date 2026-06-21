#!/bin/bash
# AMP — ROCm Build Script for AMD Developer Cloud (MI300X)
# Usage: bash scripts/build_rocm.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== AMP — AMD Native Migration ROCm Build ==="
echo "Project dir: $PROJECT_DIR"

# 1. Check ROCm installation
if [ -z "$ROCM_PATH" ]; then
    if [ -d "/opt/rocm" ]; then
        export ROCM_PATH=/opt/rocm
    else
        echo "❌ ROCm not found in /opt/rocm. Source your ROCm installation first."
        echo "   e.g., source /opt/rocm/bin/rocm_env.sh"
        exit 1
    fi
fi
echo "✅ ROCm path: $ROCM_PATH"

# 2. Detect MI300X GPU
echo ""
echo "=== Detecting GPUs ==="
$ROCM_PATH/bin/rocm-smi --showproductname 2>/dev/null || echo "⚠️ rocm-smi not available"

# 3. Create build directory
BUILD_DIR="$PROJECT_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 4. CMake configure with HIP backend (AMD MI300X optimizations)
echo ""
echo "=== CMake Configure ==="
cmake "$PROJECT_DIR" \
    -DAMP_BACKEND=HIP \
    -DAMP_HAVE_ROCWMMA=ON \
    -DAMP_HAVE_FP8=ON \
    -DAMP_HAVE_RCCL=ON \
    -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
    -DCMAKE_BUILD_TYPE=Release

# 5. Build
echo ""
echo "=== Build ==="
make -j$(nproc)

echo ""
echo "✅ AMP ROCm build complete!"
echo ""
echo "=== Run Tests ==="
echo "  ./test_triple    — Integration test"
echo "  ./bench          — Pool + Cache benchmark"
echo "  ./bench_full     — Full benchmark suite"
echo ""
echo "=== Quick test ==="
./test_triple || echo "⚠️ Test warning (GPU may not be available)"
