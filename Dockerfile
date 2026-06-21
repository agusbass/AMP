# AMP — AMD Migration Platform
# Docker image for the AMD Developer Hackathon ACT II (Track 3 Unicorn)
# Target: AMD ROCm 6.x on Instinct MI300X

FROM rocm/dev-ubuntu-22.04:6.3.2-complete AS base

LABEL maintainer="AMD Hackathon Participant"
LABEL description="AMP — AMD Migration Platform: Multi-vendor GPU Runtime"
LABEL hackathon="AMD Developer Hackathon ACT II"
LABEL track="3 - Unicorn"

WORKDIR /opt/amp

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy AMP source
COPY . .

# Build via build_auto.sh rather than re-deriving cmake flags here --
# hardcoding -DAMP_HAVE_ROCWMMA=ON/-DAMP_HAVE_FP8=ON unconditionally is
# exactly the bug found and fixed on a real MI300X pod in this project's
# validation log (find_package(rocwmma REQUIRED) and hip_fp8.h aren't
# guaranteed present just because the base image is ROCm); build_auto.sh
# checks for the actual file/config needed instead of assuming it.
RUN bash scripts/build_auto.sh

# Verify build
RUN cd build && echo "=== Test Triple ===" && ./test_triple || echo "Test finished (GPU may not be available at build time)"

# Default command: an instant, no-setup proof it works, using AMP's own
# bundled reference kernel (no GPU upload, no kernel of your own needed
# yet) -- numerical correctness vs CPU reference in well under 60s on a
# real AMD GPU (`docker run --device=/dev/kfd --device=/dev/dri ...`).
# To validate YOUR OWN kernel instead, see "Validate YOUR OWN kernel" in
# the README and run amp_validate_kernel directly.
CMD ["/bin/bash", "-c", "cd /opt/amp/build && echo '=== AMP quick proof: FP32 GEMM vs CPU reference ===' && ./amp_verify_matmul && echo && echo 'Ready. Try: ./amp_validate_kernel <your_kernel.so> dump.json M N K' && echo 'See https://github.com/agusbass/AMP#validate-your-own-kernel-not-just-amps-example'"]
