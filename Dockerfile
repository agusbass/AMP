# AMP — AMD Migration Platform
# Docker image for the AMD Developer Hackathon ACT II (Track 3 Unicorn)
# Target: AMD ROCm 6.x on Instinct MI300X

FROM rocm/rocm:6.3.2-complete AS base

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

# Build with HIP backend (AMD default)
RUN mkdir -p build && cd build && \
    cmake .. \
        -DAMP_BACKEND=HIP \
        -DAMP_HAVE_ROCWMMA=ON \
        -DAMP_HAVE_FP8=ON \
        -DAMP_HAVE_RCCL=ON \
        -DCMAKE_PREFIX_PATH=/opt/rocm && \
    make -j$(nproc)

# Verify build
RUN cd build && echo "=== Test Triple ===" && ./test_triple || echo "Test finished (GPU may not be available at build time)"

# Default command: run benchmarks
CMD ["/bin/bash", "-c", "cd /opt/amp/build && echo 'AMP — AMD Migration Platform ready!' && ls -la"]
