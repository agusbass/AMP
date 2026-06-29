# 🛠️ AMP — AMD Migration Platform

**Validates that your CUDA kernel and its HIP port produce the same numbers — so trying AMD Instinct is a same-day decision, not a multi-week migration project.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agusbass/AMP/blob/master/notebooks/quickstart.ipynb)
[![Try the diagnosis UI](https://img.shields.io/badge/🤗%20Spaces-Try%20it%20live-blue)](https://agusbudiman14-amp-kernel-diagnose.hf.space/)

- **Colab** → real GPU build + cross-vendor parity check, in your browser.
- **Spaces** → paste a kernel, get a diagnosis + fix instantly, no GPU needed.

```
Cross-vendor parity: nvidia (Tesla T4) vs amd (MI300X), M=N=K=1024

tile             A GFLOPS   B GFLOPS  B/A ratio  max_rel_err  status
(16,16,16)          583.6    12300.4     21.08x     0.000748  PASS
(32,16,16)          638.8    12988.4     20.33x     0.000748  PASS
(32,32,16)          662.0    11481.5     17.34x     0.000748  PASS
(32,32,32)          831.7    12904.0     15.52x     0.000487  PASS

ALL TILE CONFIGS CROSS-VENDOR PARITY OK (rel_err < 0.001)
```

Real hardware, not a mock. 10 real bugs found and fixed getting here —
[full log](docs/VALIDATION.md).

## What it does

1. **Validate YOUR kernel** (CUDA or HIP) against a CPU reference and a same-shape run from the other vendor.
2. **Diagnose failures with no GPU** — static analysis + mechanical fix for the bug class that survives "compiles and doesn't crash."
3. **Check model compatibility** (FlashAttention-2 head_dim/GQA) before touching hardware.

```
  Your CUDA kernel              Your HIP kernel
        |                              |
        v                              v
   1 wrapper fn  <--- or generate one: amp_generate_wrapper.py
        |                              |
        v                              v
  nvcc -shared -o *.so         hipcc -shared -o *.so
        |                              |
        v                              v
   amp_validate_kernel (vs CPU ref, both sides)
        |                              |
        v                              v
   cuda_dump.json                hip_dump.json
        \_____________   _____________/
                      \ /
          scripts/parity_check.py  -->  PASS, or FAIL --analyze
                                              |
                                              v
                                    amp_diagnose.py (static, no GPU)
                                    amp_suggest_fix.py (you confirm)
```

## Quick start

```bash
bash scripts/build_auto.sh   # auto-detects CUDA/HIP/SYCL/CPU, zero manual flags
cd build && ./amp_verify_matmul   # AMP's own kernel vs CPU reference

# Validate YOUR kernel -- one command, same one on both machines:
cd ..
bash scripts/amp_check.sh my_kernel.cu my_kernel_function_name 1024 1024 1024   # CUDA machine
bash scripts/amp_check.sh my_kernel.cu my_kernel_function_name 1024 1024 1024   # AMD machine
python3 scripts/parity_check.py cuda_dump.json hip_dump.json --analyze

# Non-standard signature? python3 scripts/amp_generate_wrapper.py my_kernel_function_name

# Or just: AMP's reference kernel proves itself in under 60s
docker build -t amp . && docker run --device=/dev/kfd --device=/dev/dri amp
```

## Auto-diagnosis (no GPU)

```bash
python3 scripts/amp_diagnose.py kernels/matmul.cu kernels/matmul.cuh --tile 32,16,32
python3 scripts/amp_suggest_fix.py kernels/matmul.cu kernels/matmul.cuh --tile 32,16,32 --json > fix.json
python3 scripts/amp_fix.py fix.json --yes   # diff shown first; nothing applied without --yes
```

Regex-based by design — zero dependencies, no GPU/compiler, runs in
milliseconds. Catches the bug class that survives "compiles and doesn't
crash." Why this tradeoff, and what it doesn't catch:
[docs/VALIDATION.md](docs/VALIDATION.md).

## Why this, why now

AMD ships HIPIFY (mechanical transpile) and the ROCm Validation Suite
(install health) — neither proves the transpiled kernel is *numerically
correct* or how far off it is in performance. That gap is this tool.
Business case and prior art: [docs/VALIDATION.md](docs/VALIDATION.md).

## What's still open

- Plugin-path parity (user's own kernel) verified on NVIDIA; HIP side pending MI300X capacity.
- `scripts/amp_pipeline.sh` is syntax-checked, not yet run end-to-end on real hardware.
- Docker's default `CMD` (auto-runs `amp_verify_matmul` on `docker run`) only has its *build* verified by CI; the GPU run itself hasn't been.
- FP8 kernel path untested (validation images lacked `hip_fp8.h`).
