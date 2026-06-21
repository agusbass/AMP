# 🛠️ AMP — AMD Migration Platform

**Validates that your CUDA kernel and its HIP port produce the same numbers — so trying AMD Instinct is a same-day decision, not a multi-week migration project.**

```
Cross-vendor parity: nvidia (Tesla T4) vs amd (MI300X), M=N=K=1024

tile             A GFLOPS   B GFLOPS  B/A ratio  max_rel_err  status
(16,16,16)          583.6    12300.4     21.08x     0.000748  PASS
(32,16,16)          638.8    12988.4     20.33x     0.000748  PASS
(32,32,16)          662.0    11481.5     17.34x     0.000748  PASS
(32,32,32)          831.7    12904.0     15.52x     0.000487  PASS

ALL TILE CONFIGS CROSS-VENDOR PARITY OK (rel_err < 0.001)
```

Real output from real NVIDIA and real AMD hardware, not a mock. Getting
here surfaced and fixed 10 real bugs across the build system, kernels, and
Docker image — full receipts in **[docs/VALIDATION.md](docs/VALIDATION.md)**.

## What it does

1. **Validate YOUR kernel** (CUDA or HIP, yours — not just AMP's example) against a CPU reference and a same-shape run from the other vendor.
2. **Diagnose failures without a GPU** — static analysis flags the kernel bug class that survives "compiles and doesn't crash" (wrong output on some tile configs), and suggests a mechanical fix.
3. **Check model compatibility** before you touch hardware — does your FlashAttention-2 kernel even support a model's head_dim/GQA shape?

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
# Build (auto-detects CUDA/HIP/SYCL/CPU, zero manual flags)
bash scripts/build_auto.sh
cd build

# AMP's own kernel, vs CPU reference (no setup needed)
./amp_verify_matmul

# Validate YOUR kernel instead -- one command, same one on both machines
# (auto-detects nvcc/hipcc, wraps, builds, validates):
cd ..
bash scripts/amp_check.sh my_kernel.cu my_kernel_function_name 1024 1024 1024   # CUDA machine
bash scripts/amp_check.sh my_kernel.cu my_kernel_function_name 1024 1024 1024   # AMD machine
python3 scripts/parity_check.py cuda_dump.json hip_dump.json --analyze

# Non-standard kernel signature? Generate a wrapper to hand-edit instead:
# python3 scripts/amp_generate_wrapper.py my_kernel_function_name

# Or just docker run -- AMP's reference kernel proves itself in under 60s
docker build -t amp . && docker run --device=/dev/kfd --device=/dev/dri amp
```

## Auto-diagnosis (no GPU)

```bash
python3 scripts/amp_diagnose.py kernels/matmul.cu kernels/matmul.cuh --tile 32,16,32
python3 scripts/amp_suggest_fix.py kernels/matmul.cu kernels/matmul.cuh --tile 32,16,32 --json > fix.json
python3 scripts/amp_fix.py fix.json --yes   # always shows a diff first; nothing applied without --yes
```

Regex/heuristic-based by design (zero dependencies, no compiler, no GPU,
runs in milliseconds) — not a full C++ parser. Catches the two bug classes
that the most dangerous failure mode (silently-wrong-output, not a crash)
is built from, with zero false positives on the shipped kernel. See
[docs/VALIDATION.md](docs/VALIDATION.md) for why this tradeoff and what it
doesn't catch.

## Why this, why now

NVIDIA controls ~80% of the AI accelerator market and a 20+ year CUDA
ecosystem — that's not in dispute. But the 2026 GPU supply crunch makes
"wait for NVIDIA" not always an option, and AMD Instinct is a credible
alternative blocked mostly by **practical setup cost**: right CMake flags,
which ROCm features exist, and whether a CUDA→HIP port is even numerically
correct before you trust it with production traffic.

AMD ships HIPIFY (mechanical transpile) and the ROCm Validation Suite
(install health) — neither answers "is the kernel HIPIFY transpiled for me
*numerically correct*, and how far off in performance?" That gap is this
tool. Full business case, prior-art comparison, and cost-vs-alternative
math: **[docs/VALIDATION.md](docs/VALIDATION.md)**.

## What's still open

- Cross-vendor parity for the plugin path (user's own kernel) is verified
  on NVIDIA; the HIP-side run is pending MI300X capacity.
- `scripts/amp_pipeline.sh` (build → verify → diagnose → fix, chained) is
  syntax-checked, not yet run end-to-end on real hardware.
- FP8 kernel path untested (the validation ROCm images used didn't all
  have `hip_fp8.h`).
