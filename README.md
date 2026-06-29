# 🛠️ AMP: CUDA ↔ ROCm Parity Check

*AMP stands for Accelerator Migration Parity.*

**Validates that your CUDA kernel and its HIP port produce the same numbers, so trying AMD Instinct is a same-day decision instead of a multi-week migration project.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agusbass/AMP/blob/master/notebooks/quickstart.ipynb)
[![Try the diagnosis UI](https://img.shields.io/badge/🤗%20Spaces-Try%20it%20live-blue)](https://agusbudiman14-amp-kernel-diagnose.hf.space/)

- **Colab**: real GPU build plus a cross-vendor parity check, right in your browser.
- **Spaces**: paste a kernel, get a diagnosis and a fix instantly, no GPU needed.

```
Cross-vendor parity: nvidia (Tesla T4) vs amd (MI300X), M=N=K=1024

tile             A GFLOPS   B GFLOPS  B/A ratio  max_rel_err  status
(16,16,16)          583.6    12300.4     21.08x     0.000748  PASS
(32,16,16)          638.8    12988.4     20.33x     0.000748  PASS
(32,32,16)          662.0    11481.5     17.34x     0.000748  PASS
(32,32,32)          831.7    12904.0     15.52x     0.000487  PASS

ALL TILE CONFIGS CROSS-VENDOR PARITY OK (rel_err < 0.001)
```

That's real hardware output, not a mock. Getting here took 10 real bugs
found and fixed along the way. Full log: [docs/VALIDATION.md](docs/VALIDATION.md).

## What it does

1. **Validate YOUR kernel** (CUDA or HIP) against a CPU reference and a same-shape run from the other vendor.
2. **Diagnose failures with no GPU.** Static analysis plus a mechanical fix for the bug class that survives "compiles and doesn't crash."
3. **Check model compatibility** (FlashAttention-2 head_dim/GQA) before you touch hardware at all.

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

This is regex-based by design: zero dependencies, no GPU or compiler
needed, runs in milliseconds. It catches the bug class that survives
"compiles and doesn't crash." For why this tradeoff and what it
doesn't catch, see [docs/VALIDATION.md](docs/VALIDATION.md).

## Why this, why now

AMD ships HIPIFY for mechanical transpiling and the ROCm Validation
Suite for install health. Neither one proves the transpiled kernel is
*numerically correct* or tells you how far off it is in performance.
That gap is what this tool fills.

HIPIFY does exactly what it's documented to do: rewrite `cuda*` calls
to `hip*`, nothing more. It has no way to know that NVIDIA's 32-thread
warp and AMD CDNA's 64-thread wavefront mean a warp-level
shuffle or reduction that's correct on one vendor can be silently
wrong on the other (see AMD's own [HIP porting
guide](https://rocm.docs.amd.com/projects/HIP/en/docs-5.7.0/user_guide/hip_porting_guide.html)).
That's the exact bug class `kernels/flash_attn.cu`'s
`warp_reduce_max` and `warp_reduce_sum` already guard against with
vendor-specific shuffle intrinsics. A transpile can compile clean and
still be wrong in a way nothing short of actually running it would
catch. That's the check this tool performs.

Not every gap shows up as a wrong number either. `nvcc` and `amdgcn`
allocate registers differently, so a kernel that fits entirely in
registers on one vendor can spill to slow scratch memory on the other,
quietly cutting throughput with no error and no numerical mismatch.
That's what `amp_parity_dump`'s GFLOPS-ratio column is for: a passing
`max_rel_err` sitting next to a far-worse-than-expected ratio is
exactly what register spilling looks like in this tool's output.

The gap is widening rather than closing, too. AI coding agents can now
port a CUDA backend to ROCm directly, skipping HIPIFY entirely. [One
widely-shared example ported a full backend in about 30
minutes](https://techstrong.ai/features/claude-code-ports-nvidia-cuda-to-amd-rocm-in-30-minutes/),
and the same coverage flags the obvious follow-up question: *"a
30-minute port doesn't prove [it] can handle production-grade,
performance-critical GPU workloads."* Faster porting just means
unverified ports happen more often, not less often. AMP is the check
that catches what a fast port, whether human-written or AI-assisted,
gets numerically wrong. For the full business case and prior art, see
[docs/VALIDATION.md](docs/VALIDATION.md).

## What's still open

- `scripts/amp_pipeline.sh` is syntax-checked but hasn't been run end-to-end on real hardware yet.
- `docker run` with real GPU device passthrough is unverified. It was attempted on a RunPod pod and blocked by that environment's nested-container networking limits (CI only proves `docker build` succeeds).
- The SYCL backend's build is now verified via CI against a real Intel oneAPI DPC++ toolchain (see [docs/VALIDATION.md](docs/VALIDATION.md)). Running it against real Intel GPU hardware is still open since none was available in this project.
- The FP8 kernel path is untested because the validation images used didn't have `hip_fp8.h`.
