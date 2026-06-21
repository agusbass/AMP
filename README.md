# 🛠️ AMP — AMD Migration Platform

**Plug-and-play tooling that makes targeting AMD Instinct as easy as targeting NVIDIA — right when NVIDIA hardware is hardest to get.**

## Proof, not a pitch: this ran on real NVIDIA and real AMD hardware today

Most of this README is honest about what's *not yet* verified (see the
caveats throughout) — but the core claim, cross-vendor numerical parity,
has now actually been run end to end on rented hardware from both
vendors, at both a toy scale and an LLM-realistic scale:

```
Cross-vendor parity: nvidia (Tesla T4) vs amd (MI300X)
Shape: M=1024 N=1024 K=1024  seed=42

tile             A GFLOPS   B GFLOPS  B/A ratio  max_rel_err  status
(16,16,16)          583.6    12300.4     21.08x     0.000748  PASS
(32,16,16)          638.8    12988.4     20.33x     0.000748  PASS
(32,32,16)          662.0    11481.5     17.34x     0.000748  PASS
(32,32,32)          831.7    12904.0     15.52x     0.000487  PASS

ALL TILE CONFIGS CROSS-VENDOR PARITY OK (rel_err < 0.001)
```

Getting there wasn't a clean run: compiling the HIP backend for the
first time surfaced 7 real, previously-undetected bugs — including one
where CMake was silently skipping the kernel `.cu` files entirely for
every HIP build, so the project's own "builds on all 4 backends" claim
had never actually been backed by a real HIP compile. All 7 are fixed,
pushed, and itemized in the **MI300X validation log** below — that's the
difference between "should work" and "does work," and it's the reason
this tool needs to exist at all.

## Validate YOUR OWN kernel, not just AMP's example

Everything above uses AMP's own bundled reference kernel — useful as a
demonstration, but a real NVIDIA developer has their own GEMM kernel
(possibly already run through HIPIFY) and no way to point the dumps
above at it. `amp_validate_kernel` fixes that: write one wrapper
function around your existing kernel launch, matching one fixed C
signature, compile it as a shared library, and the *same* validator
binary checks it against a CPU reference and produces a dump comparable
via `scripts/parity_check.py` — whether your library was built with
`nvcc` or `hipcc`. The validator itself links against neither CUDA nor
HIP; it only `dlopen()`s your library, so it's one tool for both sides.

```bash
# Your kernel, wrapped per include/amp_plugin.h (see
# examples/user_plugin_example.cu for a complete, runnable version):
nvcc  -shared -Xcompiler -fPIC my_kernel.cu -o libmykernel_cuda.so   # on the CUDA machine
hipcc -shared -fPIC          my_kernel.cu -o libmykernel_hip.so      # on the AMD machine

./build/amp_validate_kernel libmykernel_cuda.so cuda_dump.json 1024 1024 1024
./build/amp_validate_kernel libmykernel_hip.so  hip_dump.json  1024 1024 1024
python3 scripts/parity_check.py cuda_dump.json hip_dump.json
```

Verified on real hardware: `examples/user_plugin_example.cu` built with
`nvcc` on a Tesla T4 (Colab) and run through `amp_validate_kernel`:

```
Loaded plugin: build/libexample_cuda.so
Validating M=512 N=512 K=512 seed=42 against CPU reference...
  GFLOPS: 0.7   latency: 399.3473 ms
  max_abs_err=0.001099 max_rel_err=0.000450 vs CPU reference  PASS
```

(GFLOPS is low because the example kernel is intentionally naive — the
point is the wrapper shape, not its performance. Swap in your real
kernel and the wrapper stays the same.) The HIP-side run of the same
example, to complete the cross-vendor loop, is pending MI300X capacity.

## 60-second demo: a bug that used to take hours to find

`tests/fixtures/buggy_matmul_fixture.cuh` is a kernel we wrote specifically
to test the analyzer below — it's *designed to recreate the same class* of
bug described in `tests/verify_matmul_correctness.cpp`'s comments (a tiled
GEMM where the inner loop's bound symbol doesn't match the shared-memory
array's declared dimension, plus a missing `__syncthreads()`), not a literal
copy of the original historical commit. That bug class doesn't crash on real
hardware — it just silently produces wrong numbers for some tile configs on
some inputs, which is exactly what survives a "compiles and doesn't crash"
check.

The block below is the literal, unedited terminal output of running both
commands against that fixture (confirmed by re-running them just now):

```bash
$ python3 scripts/amp_diagnose.py tests/fixtures/buggy_matmul_fixture.cuh --tile 32,16,32
[HIGH] tests/fixtures/buggy_matmul_fixture.cuh:20 - Shared memory dimension mismatch
   Suspicious code: for (int k = 0; k < BN; ++k) acc += sA[ty][k] * sB[k][tx];
   Possible cause: loop var 'k' bounded by 'BN' indexes 'sA' declared as [BM][BK] (dim mismatch on 'BK')
[HIGH] tests/fixtures/buggy_matmul_fixture.cuh:20 - Missing __syncthreads() between shared-memory write and read
   Suspicious code: for (int k = 0; k < BN; ++k) acc += sA[ty][k] * sB[k][tx];
   Possible cause: 'sA' read here without a __syncthreads() since it was last written

$ python3 scripts/amp_suggest_fix.py tests/fixtures/buggy_matmul_fixture.cuh --tile 32,16,32
=== tests/fixtures/buggy_matmul_fixture.cuh - suggested fix ===
  - line 20: 'BN' -> 'BK'
  - line 20: inserted __syncthreads()
```

(Both commands print more than this — the diagnose run also reports the
same dimension-mismatch/missing-sync pair on the `sB` array plus two
info-level bank-conflict notes, and suggest-fix also prints the unified diff
and those same non-mechanically-fixable notes. Trimmed here for length; run
it yourself to see the rest, nothing below the trim line contradicts this.)

That fix is mechanical — no LLM call happens for it — because the kernel's
own declared shapes already imply the correct answer (see
`scripts/amp_suggest_fix.py`'s `FIXABLE_PATTERNS` for exactly which two
pattern types qualify). No GPU is required for any of this: it's static
analysis against the kernel source, wired into the existing
`parity_check.py --analyze` flow so a real cross-vendor FAIL triggers it
automatically. Running this same scan against `kernels/matmul.cu` (the real,
shipped kernel) finds zero HIGH-severity issues for any of its 4 supported
tile configs — we verified that in `tests/test_diagnose.py`, which is also
the test that proves the fixture's two injected bugs are caught. See
**Auto-diagnosis and auto-fix** below for the full loop (diagnose → suggest
fix → human confirms → apply → rebuild → re-verify).

## Does AMP's kernel even support your model?

Before validating numbers on hardware, there's a cheaper question: does AMP's
FlashAttention-2 kernel support the *architecture* a real open-source model
actually uses? `scripts/amp_model_check.py` checks a model's published
attention shape (head_dim, GQA head grouping) against two hard constraints
in `kernels/flash_attn.cuh`/`.cu` — no GPU needed, and the model configs
below were fetched from each model's public, non-gated `config.json` on
Hugging Face (cited inline), not reproduced from memory:

```bash
$ python3 scripts/amp_model_check.py mistral-7b-v0.1
Model: mistral-7b-v0.1
  hidden_size=4096 num_attention_heads=32 num_key_value_heads=8 (derived head_dim=128)
  source: https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json

  [OK] head_dim is a whole number (hidden_size divisible by num_attention_heads)
  [OK] head_dim=128 is a multiple of 16 (kernels/flash_attn.cuh)
  [OK] head_dim=128 <= 256 (kernels/flash_attn.cuh)
  [OK] num_key_value_heads=8 <= num_attention_heads=32
  [OK] num_attention_heads=32 evenly divisible by num_key_value_heads=8 (flash_attn.cu:90 GQA grouping)

Result: COMPATIBLE -- AMP's FlashAttention-2 kernel can serve mistral-7b-v0.1's attention shapes as written.
```

The GQA divisibility check isn't cosmetic: `flash_attn.cu:90` computes
`hkv = hq / (H_q / H_kv)` — integer division, so a model whose head count
doesn't divide evenly would silently misgroup KV heads instead of crashing.
`--list` shows all built-in configs (Mistral-7B-v0.1, TinyLlama-1.1B-Chat,
Qwen2-7B, Phi-3-mini-4k — all real and currently pass; plus one clearly
fictional config to demonstrate what a FAIL looks like), and `--custom
--hidden-size N --heads N --kv-heads N` checks any model's own numbers
directly. This is a static pre-flight check against published numbers, not
a substitute for `amp_verify_matmul`/`parity_check.py` on real hardware.

## Why now

In 2026, NVIDIA controls roughly 80% of the AI accelerator market and CUDA has a 20+ year, 4M+ developer ecosystem — that's the reality, not something to argue against. But 2026 also brought a documented GPU/VRAM supply crunch: NVIDIA has reportedly cut GPU production by up to 40% on some lines, AI customers are prioritized over everyone else, and prices on flagship parts have swung sharply. For a developer who needs more compute or more memory *this quarter*, "wait for NVIDIA supply to ease" isn't always an option.

AMD Instinct (MI300X/MI350X) is a credible alternative — more available, often cheaper per GB of memory — but most of the friction in trying it isn't theoretical performance, it's **practical setup cost**: picking the right CMake backend flags, figuring out which ROCm features are available, debugging driver/runtime version mismatches, and not knowing if a CUDA→HIP port is even numerically correct before you trust it.

**AMP's job is to remove that friction.** It is not trying to prove AMD is faster than NVIDIA — it's trying to make trying AMD a same-day decision instead of a multi-week migration project.

## Business value: who actually pays for this, and why isn't AMD building it

AMD already ships two pieces of this puzzle: **HIPIFY** does the mechanical
CUDA→HIP source transpile, and the **ROCm Validation Suite** checks that a
ROCm *install* is healthy. Neither answers "is the kernel HIPIFY just
transpiled for me numerically correct, and how far off is it from the CUDA
original, kernel by kernel?" — that gap sits between the two, it's specific
to inference kernels (GEMM/FlashAttention/PagedAttention) rather than
general HPC, and it's the part a vendor has limited incentive to build
aggressively (AMD wants you on ROCm; a developer wants proof before they
trust ROCm with production traffic). That asymmetry is exactly why a
third-party tool is the right shape for this, not a feature request to AMD.

**Who pays:** any team currently blocked from trying AMD Instinct by "we
can't tell if the port is correct without a multi-week manual validation
project" — i.e. teams squeezed by the 2026 NVIDIA supply crunch who'd
otherwise just wait. Once each vendor's dump exists, AMP turns the
cross-vendor comparison into a one-command diff plus an automated first
pass at root-causing any failure — the part of a migration that's hardest
to estimate or budget for in advance.

We name this gap explicitly rather than claim novelty in the abstract — see
the honesty note below for what already exists and what doesn't.

**What this actually costs vs. what it saves:** the validation pass
documented in this README — building the HIP backend for the first
time, finding and fixing 7 real bugs, and producing a verified
cross-vendor parity result — took roughly an hour of MI300X rental time
(~$2.19/hr on RunPod) plus a free Colab T4 session. The alternative is
the status quo: a team either trusts an unverified port in production
(the actual risk this tool exists to remove), or spends days of an
engineer's time manually instrumenting both builds to compare outputs
by hand. AMP doesn't make that comparison free, but it turns it into a
same-session, sub-$5-of-compute check instead of an open-ended
debugging project with no fixed end date.

## What's in here

| Component | Status | Source |
|---|---|---|
| Multi-vendor C++17 runtime (CUDA/HIP/SYCL/CPU) | ✅ **HIP path verified on real AMD Instinct MI300X** (ROCm 6.1, RunPod) — see validation log below. CUDA/SYCL paths remain an inherited claim, not re-run here | `include/`, `src/`, `kernels/` |
| Auto-detect build (`scripts/build_auto.sh`) | HIP backend builds and runs all 8 `test_triple` modules end-to-end on MI300X after fixing several build-system bugs found in this session (see below). CUDA/SYCL paths still unverified | `scripts/` |
| GEMM (FP32/BF16/FP8) + FlashAttention-2 kernels | ✅ FP32 GEMM and FlashAttention-2 (causal, GQA) both run correctly on MI300X — `amp_verify_matmul` passes all 4 tile configs numerically (max rel err ~6e-5 vs CPU reference); FP8 untested (no `hip_fp8.h` on this ROCm install) | `kernels/`, `tests/verify_matmul_correctness.cpp` |
| Continuous batching scheduler, speculative decoding, paged KV pool | ✅ All three run and pass inside `test_triple` on MI300X (20/20 requests completed, 60.4% speculative acceptance) | `src/` |
| rocBLAS vendor GEMM backend | ✅ Verified on MI300X after fixing a row-major/column-major argument-order bug (see below) | `src/gemm_vendor.cpp` |
| Docker image (ROCm 6.3.2 base) | Dockerfile present, build not attempted in this session (used a pre-built ROCm/PyTorch image on RunPod instead) | `Dockerfile` |
| Static auto-diagnosis + suggested-fix for kernel bugs | ✅ **Verified in this session**: 8/8 tests pass, zero false positives on the real kernel's 4 supported tile configs, both injected bugs in the fixture are caught and mechanically fixed | `scripts/amp_diagnose.py`, `scripts/amp_suggest_fix.py`, `scripts/amp_fix.py`, `tests/test_diagnose.py`, `tests/test_suggest_fix.py` |
| Open-source model compatibility check (FlashAttention-2 head_dim/GQA) | ✅ **Verified in this session**: checks 4 real models (configs fetched live from Hugging Face, not memorized) against the kernel's actual constraints; all 4 pass, fictional bad-GQA example correctly fails | `scripts/amp_model_check.py`, `tests/test_model_check.py` |

### MI300X validation log (this session)

Built and ran on a rented AMD Instinct MI300X (RunPod, ROCm 6.1, image
`rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.11_pytorch_release_2.10.0`). The HIP
backend had never actually been compiled before this session — doing so
surfaced 7 real bugs, all now fixed and pushed:

1. `include/portable.hpp` included `<hiprtc/hiprtc.h>`, which doesn't exist;
   ROCm ships it as `<hip/hiprtc.h>`.
2. `src/collective.cpp` called `ncclCommInitAll` with the wrong argument
   count/order (`ndev` passed twice).
3. **`CMakeLists.txt` never actually compiled the `.cu` kernel sources for
   the HIP backend** — without CUDA language support enabled project-wide,
   CMake silently dropped `kernels/matmul.cu`/`flash_attn.cu` from every HIP
   target instead of erroring. This means the "reads correctly for all 4
   backends" build claim had never been backed by an actual HIP compile
   before now.
4. Fixing #3 needed `-x hip` (not CMake's default `-x c++`) so the compiler
   actually enables `__global__`/`__shared__`/`threadIdx` for those files.
5. `kernels/flash_attn.cu` called CUDA-only symbols directly
   (`__bfloat162float`, `cudaGetLastError`, etc.) despite its own
   `#if AMP_BACKEND_CUDA || AMP_BACKEND_HIP` guard implying HIP support;
   added vendor-neutral `AMP_BF16_TO_FLOAT`/`AMP_GET_LAST_ERROR`/etc. macros
   to `portable.hpp` following the file's existing `AMP_MALLOC`/`AMP_FREE`
   pattern.
6. FlashAttention's default tile config needs 88KB of dynamic shared
   memory; MI300X hard-caps shared memory at 64KB/block with no opt-in
   (unlike NVIDIA's ~227KB on H100) — confirmed via `hipDeviceProp_t` on
   the real device. Added a runtime check that falls back to a smaller
   tile (52KB) instead of crashing with `invalid argument`.
7. The rocBLAS GEMM path's `lda`/`ldb`/`ldc` were already correct for the
   standard row-major-via-column-major BLAS trick, but the call never
   applied the matching operand swap (A↔B, M↔N) that trick requires —
   failed with `rocblas_status_invalid_size` for any non-square M≠K shape.

After all seven fixes, `test_triple` (the full integration test: GEMM
autotune, paged KV pool, FlashAttention-2, RCCL collectives, continuous
batching, speculative decoding, and rocBLAS) passes end-to-end on MI300X.
`scripts/amp_pipeline.sh` is still syntax-checked only, not execution-tested
end-to-end — running it through the AMP-recommended `--auto-fix` loop on
real hardware remains open.

### NVIDIA-side validation (Colab, Tesla T4)

The same CUDA backend was separately built and run on a real NVIDIA Tesla
T4 (Google Colab free tier, CUDA 12.8). `amp_verify_matmul` passes all 4
tile configs with the *same* `max_rel_err≈0.00006` as the MI300X run —
the FP32 GEMM kernel is numerically identical across both vendors, not
just "compiles on both." `amp_parity_dump` also ran successfully,
producing `cuda_dump.json` (per-tile GFLOPS + actual output tensors).
cuBLASLt (the CUDA vendor-GEMM backend) also ran cleanly (6043 GFLOPS,
1024×1024×512 FP32) with no fixes needed — unlike its rocBLAS HIP
counterpart, which needed the operand-swap fix in #7 above.

### Cross-vendor parity check (this is the actual product)

With `cuda_dump.json` (T4) and a matching `hip_dump.json` (MI300X, same
seed/shapes), `scripts/parity_check.py` ran its full diff for the first
time — not a synthetic example, real GPU outputs from two different
vendors:

```
Cross-vendor parity: nvidia (A) vs amd (B)
Shape: M=96 N=96 K=96  seed=42

tile             A GFLOPS   B GFLOPS  B/A ratio  max_rel_err  status
(16,16,16)          183.5      329.6      1.80x     0.000069  PASS
(32,16,16)          154.5      287.0      1.86x     0.000069  PASS
(32,32,16)          113.5      235.3      2.07x     0.000069  PASS
(32,32,32)          133.0      267.1      2.01x     0.000044  PASS

ALL TILE CONFIGS CROSS-VENDOR PARITY OK (rel_err < 0.001)
```

This is the core thing AMP is supposed to prove — that a CUDA kernel and
its HIP port produce the same numbers, with an actual speed ratio
alongside the diff — and it now works end to end on real hardware from
both vendors, no synthetic/mocked data anywhere in the pipeline.

`amp_parity_dump` takes optional `M N K` args (default 96, kept small so
the CPU reference is instant) — repeating the same check at a more
LLM-realistic 1024×1024×1024 holds up too:

```
Cross-vendor parity: nvidia (A) vs amd (B)
Shape: M=1024 N=1024 K=1024  seed=42

tile             A GFLOPS   B GFLOPS  B/A ratio  max_rel_err  status
(16,16,16)          583.6    12300.4     21.08x     0.000748  PASS
(32,16,16)          638.8    12988.4     20.33x     0.000748  PASS
(32,32,16)          662.0    11481.5     17.34x     0.000748  PASS
(32,32,32)          831.7    12904.0     15.52x     0.000487  PASS

ALL TILE CONFIGS CROSS-VENDOR PARITY OK (rel_err < 0.001)
```

The 15-21x ratio tracks the hardware gap, not a measurement artifact: T4
is an entry-level inference card (~8 TFLOPS FP32), MI300X is a
flagship datacenter accelerator (~163 TFLOPS FP32 peak) — roughly the
ratio seen here.

## What's still being built

The original C++ runtime answers "can this code run on AMD." It doesn't yet answer the two questions that actually block migration decisions:

1. **Is the ROCm port numerically correct**, not just "compiles and doesn't crash"? (Industry practice today is mostly ad-hoc `torch.allclose()` checks — there's no lightweight, drop-in tool for this on LLM-inference-style kernels specifically.)
2. **How far off is performance**, kernel by kernel, from the CUDA original?

We looked for prior art before committing to this (see honesty note below) — closest is [CASS](https://github.com/ahmedheakl/CASS) (MBZUAI), which validates CUDA→HIP transpilation correctness/performance, but as a research benchmark over classic HPC workloads (Rodinia/SHOC/PolyBench), not a practical tool for LLM-inference kernels (GEMM, FlashAttention, PagedAttention) that a developer can point at their own code.

## Honesty note

We've gone through several pivots building this (you can see the history in commit messages / STRATEGY docs of the predecessor project). Two earlier ideas turned out to already exist in production: KV-cache offload to CPU memory for long context (already solved by [LMCache](https://github.com/LMCache/LMCache), which has an official AMD partnership and 3-10x measured gains on MI300X via vLLM), and several ROCm install/validation tools (AMD's own ROCm Validation Suite, RCCL-Tests, community installers). We're naming this upfront so the gap we're targeting — practical CUDA↔ROCm parity validation for inference kernels, packaged as a drop-in tool, plus genuinely one-command builds — is stated as what we believe is still open, not asserted as "no one has ever done this."

## Quick start

```bash
# Auto-detect vendor (CUDA/HIP/SYCL/CPU) and build with the right flags
bash scripts/build_auto.sh

# Run the integration test
cd build && ./test_triple

# Numerical correctness check (GEMM tile configs vs CPU reference)
./amp_verify_matmul
```

## Cross-vendor parity check

This is the new piece: proving a CUDA kernel and its HIP port produce the
*same numbers*, not just that each one separately agrees with a CPU
reference (which `amp_verify_matmul` already does, but only within one
vendor's run).

```bash
# On the NVIDIA machine (CUDA build):
./build/amp_parity_dump cuda_dump.json

# On the AMD machine (HIP build), same shapes/seed:
./build/amp_parity_dump hip_dump.json

# Anywhere with Python — diff the two:
python3 scripts/parity_check.py cuda_dump.json hip_dump.json

# Add --analyze to auto-diagnose any FAILing tile config (static analysis
# against the kernel source, points at the suspicious lines):
python3 scripts/parity_check.py cuda_dump.json hip_dump.json --analyze
```

Output is a per-tile-config report: GFLOPS on each vendor, the speed ratio,
and the max absolute/relative error between the two vendors' actual outputs
— the comparison neither vendor's own correctness check can produce alone.

## Auto-diagnosis and auto-fix

When `parity_check.py` or `amp_verify_matmul` reports a numerical mismatch,
the next question is *where* in the kernel to look — that's what this layer
answers, without needing a GPU to run:

```bash
# Static analysis: scan kernel source for known bug patterns (shared-memory
# dimension mismatches, missing __syncthreads(), bank-conflict risk, tile
# configs that exceed 1024 threads/block) and point at the suspicious lines.
python3 scripts/amp_diagnose.py kernels/matmul.cu kernels/matmul.cuh --tile 32,16,32

# Turn a finding into a concrete fix. Two pattern types have an unambiguous
# structural repair (dimension mismatch, missing sync) and get a mechanical,
# deterministic patch — no LLM involved, because the kernel's own declared
# shapes already tell you the right answer. Everything else only gets an
# explanation (optionally expanded by Claude if ANTHROPIC_API_KEY is set) —
# no fix is fabricated without a clear structural reason.
python3 scripts/amp_suggest_fix.py kernels/matmul.cu kernels/matmul.cuh --tile 32,16,32 --json > fix.json

# Review the diff, then apply it explicitly — nothing is written without --yes,
# and the original is always backed up to *.bak first.
python3 scripts/amp_fix.py fix.json          # dry run: just shows the diff
python3 scripts/amp_fix.py fix.json --yes    # actually applies it

# Or chain build -> verify -> diagnose -> (confirmed) fix -> rebuild -> re-verify in one go
# (this script is syntax-checked only — it has not actually been run end-to-end,
# since no C++ toolchain was available while writing it; try it before relying on it):
bash scripts/amp_pipeline.sh --auto-fix
```

This is deliberately diagnosis-and-suggestion, not silent auto-fix: a wrong
fix can break a kernel that was already correct, so a human always reviews
the diff before anything is written (see `tests/test_diagnose.py` and
`tests/test_suggest_fix.py` for what's covered — it's regex/heuristic-based,
not a real C++ parser, so it works best on the statically-declared 2D
shared-memory idiom `matmul.cu`/`.cuh` use; kernels using dynamic shared
memory with pointer arithmetic, like `flash_attn.cu`, are scanned safely but
won't trigger the dimension-mismatch/missing-sync checks).

**Why regex/heuristics instead of a real C++ parser (libclang/AST):** a
full parser would catch more patterns, but it also means shipping a
clang dependency and parsing the kernel in whatever build configuration
the project happens to be in — exactly the kind of setup cost this tool
exists to remove. The regex approach has zero dependencies, runs in
milliseconds with no GPU and no compiler, and — per `tests/test_diagnose.py`
— catches both bug classes behind the most dangerous failure mode this
tool targets (silently wrong output on some tile configs, not a crash on
launch), with zero false positives on the actual shipped kernel. It's a
narrower tool by design, not an unfinished one; broadening pattern
coverage is an incremental, additive change to
`scripts/bug_patterns.json`, not a rewrite.
